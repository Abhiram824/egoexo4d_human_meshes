"""
Stage 0 (ego): extracts frames from the Aria RGB mp4 via ffmpeg and undistorts
them in place to a 512x512 pinhole model using the take's VRS calibration.
"""
import os
import glob
from PIL import Image
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from projectaria_tools.core import calibration
from util.egoexo4d_utils import get_calib

def undistort_image(curr_dist_img_path, aria_rgb_calib, pinhole):
    """Undistorts one Aria frame in place (fisheye -> 512x512 pinhole)."""
    curr_dist_image = np.array(Image.open(curr_dist_img_path))
    # the Aria sensor is mounted rotated 90 degrees: undistort in native
    # orientation, then rotate back to upright
    curr_dist_image = (
        cv2.rotate(curr_dist_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    )
    # Undistortion
    undistorted_image = calibration.distort_by_calibration(
        curr_dist_image, pinhole, aria_rgb_calib
    )
    undistorted_image = (
        cv2.rotate(undistorted_image, cv2.ROTATE_90_CLOCKWISE)
    )
    # Save undistorted image
    assert cv2.imwrite(
        curr_dist_img_path, undistorted_image[:, :, ::-1]
    ), f"{curr_dist_img_path} write failed!"


def main(args):

    video_path = args.video_path
    video_path_to_save = args.video_path_to_save

    os.makedirs(video_path_to_save, exist_ok=True)
    os.system('ffmpeg -threads %d -i %s -qscale:v 2 %s/%%06d.jpg' % (args.num_procs, video_path, video_path_to_save))
    # video_path is .../takes/<take>/frame_aligned_videos/<aria>_214-1.mp4
    take_name = video_path.split('/')[-3]

    aria_rgb_calib, pinhole = get_calib(take_name)
    

    frames_list = glob.glob(os.path.join(video_path_to_save, '*.jpg'))
    frames_list.sort()

    for frame_path in tqdm(frames_list):
        undistort_image(frame_path, aria_rgb_calib, pinhole)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--video_path_to_save", type=str)
    parser.add_argument("--num_procs", type=int, default=10)
    args = parser.parse_args()

    main(args)
