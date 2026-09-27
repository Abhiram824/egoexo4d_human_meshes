"""
Builds the pipeline's cameras.npz (intrins/w2c/dist, shape [num_views+1, T, ...])
from EgoExo4D GoPro calibrations, plus per-frame Aria camera matrices if --use-aria.
View 0 is a reference copy of the first exo cam; views 1-4 are the exo cams.
"""

import cv2
import numpy as np
import argparse

import math
from projectaria_tools.core import mps
from projectaria_tools.core.calibration import CameraCalibration, KANNALA_BRANDT_K3
from util.egoexo4d_utils import (
    get_mps, get_calib, get_frame_timestamps, get_camera_matricies,
    load_csv_to_df, get_gopro_calibs_path, extract_fisheye_params,
)

import os


def convert_intrinsics(df_traj):
    """
    Converts one gopro_calibs.csv row's fisheye intrinsics into the pinhole
    camera matrix of the undistorted 3840x2160 frame.
    """
    dimension=(3840, 2160)
    intrinsics, distortion_coeffs = extract_fisheye_params(df_traj)
    cx = intrinsics[0,2]
    cy = intrinsics[1,2]
    # vertical cameras (e.g. bouldering cam01/cam04) have swapped dimensions
    if cy > cx:
        dimension = (dimension[1], dimension[0])

    # balance must match undistort_egoexo.py, which undistorts the actual frames
    balance = 0.8
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsics, distortion_coeffs, dimension, np.eye(3), balance=balance
    )
    return new_K.tolist()


def get_exocam_params(take_name):
    """
    Returns per-camera dicts (world->cam 3x4 extrinsics + undistorted-4K
    intrinsics) for the take's well-localized exo GoPros, in gopro_calibs.csv order.
    """
    exo_traj_path = get_gopro_calibs_path(take_name)
    exo_traj_df = load_csv_to_df(exo_traj_path)
    static_calibrations = mps.read_static_camera_calibrations(exo_traj_path)

    cam_info = []
    for sc in static_calibrations:
        # Skip cameras that were not localized
        if sc.quality != 1.0:
            print(f"Camera: {sc.camera_uid} was not localized. Skipping.")
            continue

        calib = CameraCalibration(
            sc.camera_uid,
            KANNALA_BRANDT_K3,
            sc.intrinsics,
            sc.transform_world_cam,
            sc.width,
            sc.height,
            None,
            math.pi,
            "")
        # transform_world_cam was passed as the "device" pose, so inverting
        # the device->camera transform yields world->camera
        se3_dev_cam = calib.get_transform_device_camera()
        se3_cam_dev = se3_dev_cam.inverse()

        cam_info.append({
            "camera_extrinsics": se3_cam_dev.to_matrix3x4().tolist(),
            "camera_intrinsics": convert_intrinsics(exo_traj_df[exo_traj_df["cam_uid"] == sc.camera_uid].iloc[0]),
        })

    return cam_info

def read_camera(args):
    """
    Writes cameras.npz: static exo calibrations repeated across all frames,
    plus per-frame Aria matrices in view 5 when --use-aria is set.
    """
    video_len = args.video_len
    save_path = args.save_path
    take_name = args.take_name
    json_contents = get_exocam_params(take_name)
    num_views = 5  if args.use_aria else 4

    # arrays hold num_views+1 entries: view 0 is a reference copy of the first exo cam
    intrinsics = np.zeros([num_views+1,video_len,3,3])
    w2c = np.zeros([num_views+1,video_len,4,4])
    dist = np.zeros([num_views+1,video_len,5])

    # view 0: reference view (duplicate of the first exo camera)
    viewid = 0
    for i in range(video_len):
        intrinsics[viewid, i] = json_contents[0]['camera_intrinsics']
        # Scale intrinsics for 1920x1080 resolution (0.5x scale from 3840x2160)
        intrinsics[viewid, i, :2, :] *= 0.5
        w2c[viewid, i, :3, :4] = np.array(json_contents[0]['camera_extrinsics'])
        w2c[viewid, i, 3, 3] = 1
        dist[viewid, i] = np.zeros(5)
    for viewid in range(4):
        for i in range(video_len):
            intrinsics[viewid+1, i] = json_contents[viewid]['camera_intrinsics']
            # Scale intrinsics for 1920x1080 resolution (0.5x scale from 3840x2160)
            intrinsics[viewid+1, i, :2, :] *= 0.5
            w2c[viewid+1, i, :3, :4] = np.array(json_contents[viewid]['camera_extrinsics'])
            w2c[viewid+1, i, 3, 3] = 1
            dist[viewid+1, i] = np.zeros(5)

    if args.use_aria:
        # view 5: egocentric Aria camera, whose pose changes every frame —
        # look up the MPS trajectory at each video frame's capture timestamp
        mps_data_provider= get_mps(take_name)
        rgb_camera_calibration, pinhole = get_calib(take_name)
        # TODO : verify timestamps match video length
        frame2timestamp = get_frame_timestamps(take_name)
        assert video_len == len(frame2timestamp), f"Video length {video_len} does not match frame timestamps length {len(frame2timestamp)}"
        for i,ts in enumerate(frame2timestamp):
            K, R, t = get_camera_matricies(mps_data_provider, ts, rgb_camera_calibration, pinhole,
                                            apply_rotation=True, native_img_height=512)
            intrinsics[5, i] = K
            w2c[5, i, :3, :3] = R
            w2c[5, i, :3, 3] = t.flatten()
            w2c[5, i, 3, 3] = 1
            dist[5, i] = np.zeros(5)  # assume no distortion for pinhole model
    np.savez(save_path, intrins=intrinsics, w2c=w2c, dist=dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_len", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--take-name", type=str)
    parser.add_argument("--use-aria", action='store_true')
    args = parser.parse_args()

    read_camera(args)