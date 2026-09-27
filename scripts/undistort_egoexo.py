"""
Stage 0 (exo): extracts frames from one GoPro mp4 via ffmpeg and undistorts them
in place (Kannala-Brandt fisheye -> pinhole), resizing 3840x2160 -> 1920x1080.
"""
import os
import glob
from PIL import Image
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from util.egoexo4d_utils import load_csv_to_df, get_gopro_calibs_path, extract_fisheye_params


def get_exocam_info(take_name, cam):
    """Returns the native-4K fisheye intrinsics + distortion for one camera uid."""
    exo_traj_df = load_csv_to_df(get_gopro_calibs_path(take_name))
    return extract_fisheye_params(exo_traj_df[exo_traj_df["cam_uid"] == cam].iloc[0])

def undistort_exocam(image_path, intrinsics, distortion_coeffs, dimension=(3840, 2160)):
    """
    Undistorts one fisheye frame; returns the undistorted image and the new
    pinhole camera matrix it corresponds to.
    """
    DIM = dimension
    dim2 = None
    dim3 = None
    # balance must match read_cameras.py's convert_intrinsics, which computes
    # the pinhole intrinsics the pipeline pairs with these undistorted frames
    balance = 0.8
    # Load the distortion parameters
    distortion_coeffs = distortion_coeffs
    # Load the camera intrinsic parameters
    intrinsics = intrinsics

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dim1 = image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    # assert dim1[0] >= dim1[1], "Assuming width is greater than height for the input image."

    # Change the calibration dim dynamically
    # (e.g., bouldering cam01 and cam04 are verticall)
    if DIM[0] != dim1[0]:
        DIM = (DIM[1], DIM[0])

    assert (
        dim1[0] / dim1[1] == DIM[0] / DIM[1]
    ), "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = (
        intrinsics * dim1[0] / DIM[0]
    )  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to
    # determine the final K used to un-distort image.
    # OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, distortion_coeffs, dim2, np.eye(3), balance=balance
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, distortion_coeffs, np.eye(3), new_K, dim3, cv2.CV_16SC2
    )
    undistorted_image = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return undistorted_image, new_K

def process_frame(frame_path, intrinsics, distortion_coeffs):
    """Process a single frame - undistort and resize."""
    undistorted_image, new_K = undistort_exocam(
        frame_path, intrinsics, distortion_coeffs, (3840, 2160)
    )
    undistorted_image = Image.fromarray(undistorted_image)
    w,h = undistorted_image.size
    target_size = (1920, 1080) if w >= h else (1080, 1920)
    undistorted_image = undistorted_image.resize(target_size, Image.LANCZOS)
    undistorted_image.save(frame_path)
    return frame_path

def main(args):

    video_path = args.video_path
    video_path_to_save = args.video_path_to_save

    os.makedirs(video_path_to_save, exist_ok=True)
    # video_path is .../takes/<take>/frame_aligned_videos/<cam_uid>.mp4
    take_name = video_path.split('/')[-3]
    cam_name = video_path.split('/')[-1][:-4]

    # -loglevel warning -stats trades ffmpeg's build banner for a single
    # self-updating progress line, so long extractions still show life
    print(f"[{take_name}/{cam_name}] extracting frames from video...", flush=True)
    os.system('ffmpeg -loglevel warning -stats -threads %d -i %s -qscale:v 2 %s/%%06d.jpg' % (args.num_procs, video_path, video_path_to_save))

    intrinsics, distortion_coeffs = get_exocam_info(take_name, cam_name)

    frames_list = glob.glob(os.path.join(video_path_to_save, '*.jpg'))
    frames_list.sort()

    # Use multiprocessing to process frames in parallel
    process_func = partial(process_frame, intrinsics=intrinsics, distortion_coeffs=distortion_coeffs)

    with Pool(processes=args.num_procs) as pool:
        list(tqdm(pool.imap(process_func, frames_list), total=len(frames_list),
                  desc=f"[{take_name}/{cam_name}] undistorting", unit="frame"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--video_path_to_save", type=str)
    parser.add_argument("--num_procs", type=int, default=10)
    args = parser.parse_args()

    main(args)
