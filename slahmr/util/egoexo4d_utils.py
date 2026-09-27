"""
Shared EgoExo4D helpers: takes.json/gopro_calibs.csv lookups, Aria MPS pose and
calibration loading, timestamp sync, and ego-position-guided person selection.
"""
from macros import DATASET_BASE_PATH, CAM_NAME2ID
import cv2
import projectaria_tools.core.mps as mps
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.stream_id import StreamId
import glob
import re
import pandas as pd
import json
import os
import numpy as np
import subprocess

VRS_PATH = os.path.join(DATASET_BASE_PATH, "takes/{video_name}/*noimagestreams.vrs")
FPS = 30.0

# body-keypoint slots (0-24) of the pipeline's 67-kp format, in OpenPose BODY_25 order
OPENPOSE_KP_NAMES = [
    "nose", "neck", "right-shoulder", "right-elbow", "right-wrist",
    "left-shoulder", "left-elbow", "left-wrist", "mid-hip", "right-hip",
    "right-knee", "right-ankle", "left-hip", "left-knee", "left-ankle",
    "right-eye", "left-eye", "right-ear", "left-ear", "left-big-toe",
    "left-small-toe", "left-heel", "right-big-toe", "right-small-toe", "right-heel"
]
OPENPOSE_KP_MAP = {name: idx for idx, name in enumerate(OPENPOSE_KP_NAMES)}


def estimate_eye_pose_bbox(kp):
    """
    Fallback eye-position estimate when no eye/ear keypoints are confident:
    top of the whole-body bbox, offset down slightly to approximate head height.
    """
    valid_kps = kp[kp[:, 2] > 0.1][:, :2]
    if len(valid_kps) == 0:
        return None

    bbox_min = valid_kps.min(axis=0)
    bbox_max = valid_kps.max(axis=0)
    x1, y1 = bbox_min
    x2, y2 = bbox_max
    w, h = x2 - x1, y2 - y1

    cx = (x1 + x2) / 2.0
    cy = y1 + 0.06 * h

    return np.array([cx, cy])


def get_eye_pos(kp):
    """Returns a 2D head-position estimate for a detected person: eyes, else ears, else bbox fallback."""
    left_eye_kp = kp[OPENPOSE_KP_MAP["left-eye"]]
    right_eye_kp = kp[OPENPOSE_KP_MAP["right-eye"]]
    if left_eye_kp[2] > 0.1 and right_eye_kp[2] > 0.1:
        eye_pos = (left_eye_kp[:2] + right_eye_kp[:2]) / 2.0
        return eye_pos

    # use ear kps if eye kps are not available
    left_ear_kp = kp[OPENPOSE_KP_MAP["left-ear"]]
    right_ear_kp = kp[OPENPOSE_KP_MAP["right-ear"]]
    if left_ear_kp[2] > 0.1 and right_ear_kp[2] > 0.1:
        ear_pos = (left_ear_kp[:2] + right_ear_kp[:2]) / 2.0
        return ear_pos

    # use whole bbox center if neither eye nor ear kps are available
    return estimate_eye_pose_bbox(kp)


def get_best_tracklet_for_frame(tracklet_candidates, ego_pos_2d):
    """Picks the detected person whose head position is closest to the projected Aria ego position (rejects bystanders)."""
    best_candidate = None
    best_distance = float('inf')
    for candidate in tracklet_candidates:
        keypoints_2d = np.array(candidate['keypoints']).reshape(-1, 3)
        eye_pos_2d = get_eye_pos(keypoints_2d)
        if eye_pos_2d is None:
            continue
        distance = np.linalg.norm(eye_pos_2d - ego_pos_2d)
        if distance < best_distance:
            best_distance = distance
            best_candidate = candidate
    return best_candidate


def extract_aria_calib_to_dict(input_vrs):
    """
    Shells out to the `vrs` CLI to pull the camera-rgb calibration JSON out of
    a .vrs file, then rescales its focal length/principal point by 2 and
    shifts the principal point by 32px to match the pipeline's half-res,
    32px-cropped Aria frames (see undistort_egocam.py).
    """
    extract_calibration_json_cmd = f"vrs {input_vrs} | grep calib_json"

    p = subprocess.Popen(
        [extract_calibration_json_cmd], shell=True, stdout=subprocess.PIPE
    )
    out, err = p.communicate()
    out = out.decode("utf-8")

    # strip the "... calib_json = " prefix and trailing char the vrs CLI emits
    calib_json_string = out[20:-1]
    calib_dict = json.loads(calib_json_string)
    all_cam_calib = calib_dict["CameraCalibrations"]
    aria_cam_calib = [c for c in all_cam_calib if c["Label"] == "camera-rgb"][0]
    aria_cam_calib["Projection"]["Params"][0] /= 2
    aria_cam_calib["Projection"]["Params"][1] = (
        aria_cam_calib["Projection"]["Params"][1] - 0.5 - 32
    ) / 2
    aria_cam_calib["Projection"]["Params"][2] = (
        aria_cam_calib["Projection"]["Params"][2] - 0.5 - 32
    ) / 2
    return calib_dict

def load_csv_to_df(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as csv_file:
        return pd.read_csv(csv_file)


def get_take_meta(take_name):
    """Returns the take's takes.json metadata entry."""
    takes = json.load(open(os.path.join(DATASET_BASE_PATH, "takes.json")))
    return [take for take in takes if take["take_name"] == take_name][0]


def get_gopro_calibs_path(take_name):
    """Returns the path to the take's trajectory/gopro_calibs.csv."""
    take = get_take_meta(take_name)
    return os.path.join(DATASET_BASE_PATH, take["root_dir"], "trajectory", "gopro_calibs.csv")


def extract_fisheye_params(calib_row):
    """
    Parses one gopro_calibs.csv row into the native-4K fisheye camera matrix
    and Kannala-Brandt (k1-k4) distortion coefficients.
    """
    intrinsics = np.array(
        [
            [float(calib_row[12]), 0, float(calib_row[14])],
            [0, float(calib_row[13]), float(calib_row[15])],
            [0, 0, 1],
        ]
    )
    distortion_coeffs = np.array(
        [
            float(calib_row[16]),
            float(calib_row[17]),
            float(calib_row[18]),
            float(calib_row[19]),
        ]
    )
    return intrinsics, distortion_coeffs


def get_take_cameras(take_name):
    """Returns the take's well-localized (quality == 1.0) exo GoPro uids, in gopro_calibs.csv order."""
    exo_traj_path = get_gopro_calibs_path(take_name)

    cam_info = []
    static_calibrations = mps.read_static_camera_calibrations(exo_traj_path)
    for cam in range(len(static_calibrations)):    # assert the GoPro was correctly localized
        static_calibration = static_calibrations[cam]
        if static_calibration.quality != 1.0:
            continue
        cam_info.append(static_calibration.camera_uid)
    return cam_info


def get_ego_cam_name(take_name):
    """Returns the take's single Aria ego camera id (e.g. "aria01")."""
    take = get_take_meta(take_name)
    ego_cam_names = [x["cam_id"] for x in take["capture"]["cameras"] if x["is_ego"] and x["cam_id"].startswith("aria")]
    assert len(ego_cam_names) == 1, f"Expected exactly one ego cam for take {take_name}, found: {ego_cam_names}"
    return ego_cam_names[0]


def get_mps(take_name):
    """Loads the take's Aria MPS (Machine Perception Services) closed-loop trajectory provider."""
    ego_exo_project_path = os.path.join(DATASET_BASE_PATH, 'takes', take_name)
    mps_data_paths_provider = mps.MpsDataPathsProvider(ego_exo_project_path)
    mps_data_paths = mps_data_paths_provider.get_data_paths()
    mps_data_provider = mps.MpsDataProvider(mps_data_paths)
    return mps_data_provider

def get_calib(take_name):
    """Returns the take's Aria RGB camera calibration plus the 512x512 pinhole model it's undistorted to."""
    ego_exo_project_path = os.path.join(DATASET_BASE_PATH, 'takes', take_name)
    vrs_path = glob.glob(os.path.join(ego_exo_project_path, f"*noimagestreams.vrs"))[0]
    calib_dict = extract_aria_calib_to_dict(vrs_path)
    rgb_camera_calibration = calibration.device_calibration_from_json_string(
        json.dumps(calib_dict)
    ).get_camera_calib("camera-rgb")
    pinhole = calibration.get_linear_camera_calibration(512, 512, 150)
    return rgb_camera_calibration, pinhole


def get_frame_timestamps(take_name):
    """
    Returns each extracted video frame's capture timestamp (ns) by slicing the
    capture-level timesync.csv down to this take's [start_idx, end_idx) range,
    forward-filling any NaN (dropped sync frame) from the last valid timestamp.
    """
    # get the capture name, and load the timesync.csv
    capture_name = re.sub(r"_\d+$", "", take_name)
    timesync = pd.read_csv(os.path.join(DATASET_BASE_PATH, f"captures/{capture_name}/timesync.csv"))
    takes_info = json.load(open(os.path.join(DATASET_BASE_PATH, "takes.json")))
    for take in takes_info:
        name = take["take_name"]
        if name == take_name:
            start_idx = take["timesync_start_idx"]
            end_idx = take["timesync_end_idx"]
            ego_cam_names = [x["cam_id"] for x in take["capture"]["cameras"] if x["is_ego"] and x["cam_id"].startswith("aria")]
            assert len(ego_cam_names) == 1, f"Expected exactly one ego cam name, found {len(ego_cam_names)}"
            ego_cam_name = ego_cam_names[0]
            break
    col = f"{ego_cam_name}_214-1_capture_timestamp_ns"
    take_timestamps = []
    last_valid = None
    for idx in range(start_idx, end_idx):
        val = timesync.iloc[idx][col]
        if pd.isna(val):
            if last_valid is None:
                raise ValueError(f"First frame of {take_name} has NaN timestamp in {col} with no prior valid value to fall back to")
            take_timestamps.append(last_valid)
        else:
            last_valid = int(val)
            take_timestamps.append(last_valid)
    return take_timestamps

def project_to_egocam(pose_vector_in_world, mps_data_provider, frame_num, frame2timestamp, rgb_camera_calibration, pinhole):
    """Projects world-frame 3D points into the Aria egocam's pinhole image at frame_num; only used by __main__ below."""
    sample_timestamp = frame2timestamp[frame_num]
    # ego_reprojection = {}
    pose_info = mps_data_provider.get_closed_loop_pose(sample_timestamp)
    assert pose_info is not None, f"No pose found for timestamp {sample_timestamp}!"
    # cam = ego_cam_name

    T_world_device = pose_info.transform_world_device
    T_device_camera = rgb_camera_calibration.get_transform_device_camera()
    T_world_camera = T_world_device @ T_device_camera
    device_projection = []
    # skip body kps
    for i,pt in enumerate(pose_vector_in_world):
        pt_in_camera = T_world_camera.inverse() @ pt  # This uses Aria's SE3 transform on a 3D vector
        pt_in_device = pinhole.project(pt_in_camera.reshape(3, 1))
        # only vis hands for egocam
        if pt_in_device is None:
            device_projection.append(np.array([0.0, 0.0, 0.0]))
        else:
            device_projection.append(np.array([pt_in_device[0], pt_in_device[1], 1.0]))
    device_projection = np.array(device_projection)
    return device_projection

def get_camera_matricies(mps: mps.MpsDataPathsProvider, sample_timestamp, calibration: calibration.CameraCalibration, pinhole: calibration.CameraCalibration, apply_rotation=False, native_img_height=None):
    """
    Get camera matrices (intrinsics, rotation, translation) for a given timestamp.
    
    Args:
        mps: MPS data provider
        sample_timestamp: Timestamp to get pose for
        calibration: Aria RGB camera calibration
        pinhole: Pinhole camera model for projection
        apply_rotation: If True, applies 90° CW rotation to intrinsics so projected
                        points align with the stored (portrait) Aria image without
                        needing to rotate the image.
        native_img_height: Height of the camera's native (landscape) image, which equals
                           the WIDTH of the stored portrait image (required if apply_rotation=True).
    
    Returns:
        K: 3x3 intrinsic matrix (with rotation baked in if apply_rotation=True)
        R: 3x3 rotation matrix (world to camera)
        t: 3x1 translation vector (world to camera)
    """
    pose_info = mps.get_closed_loop_pose(sample_timestamp)
    assert pose_info is not None, f"No pose found for timestamp {sample_timestamp}!"
    T_world_device = pose_info.transform_world_device
    T_device_camera = calibration.get_transform_device_camera()
    T_world_camera = T_world_device @ T_device_camera

    T_camera_world = T_world_camera.inverse()
    R = T_camera_world.rotation().to_matrix()
    t = T_camera_world.translation().reshape(3,1)
    cx, cy = pinhole.get_principal_point()
    fx, fy = pinhole.get_focal_lengths()
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    if apply_rotation:
        assert native_img_height is not None, "native_img_height is required when apply_rotation=True"
        # The stored Aria image is portrait (rotated 90° CW from camera's native landscape).
        # Camera intrinsics produce coordinates in the native landscape frame.
        # To transform to stored portrait coords, apply 90° CW rotation:
        #   (x_native, y_native) -> (H_native - y_native, x_native)
        # where H_native is the height of the native landscape image 
        # (= width of stored portrait image).
        R_img = np.array([
            [0, -1, native_img_height],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=np.float64)
        K = R_img @ K
    
    return K, R, t

def load_ego_poses(video_name, num_frames):
    """
    Returns the Aria wearer's world-frame head (CPF, central-pupil-frame)
    position per frame, using timesync.csv-derived timestamps. Used to select
    the correct tracked person (get_best_tracklet_for_frame) and to project
    the ego position into each exo view.
    """
    timestamps = get_frame_timestamps(video_name)
    mps_provider = get_mps(video_name)
    
    # get device calibration from VRS file (MpsDataProvider doesn't have get_device_calibration)
    vrs_paths = glob.glob(VRS_PATH.format(video_name=video_name))
    if len(vrs_paths) != 1:
        raise FileNotFoundError(f"VRS file not found or multiple found for video {video_name} in path {VRS_PATH.format(video_name=video_name)}")
    vrs_provider = data_provider.create_vrs_data_provider(vrs_paths[0])
    device_calib = vrs_provider.get_device_calibration()
    T_device_cpf = device_calib.get_transform_device_cpf()

    ego_poses_3d = []
    for i, ts in enumerate(timestamps[:num_frames]):
        frame_i = i + 1  # frames are 1-indexed
        nearest_pose = mps_provider.get_closed_loop_pose(ts)
        if nearest_pose is None:
            raise ValueError(f"No nearest pose found for frame {frame_i} with timestamp {ts}")
        # world -> device
        T_world_device = nearest_pose.transform_world_device
        # compose to get world -> CPF (mid-eye center)
        T_world_cpf = T_world_device @ T_device_cpf

        ego_poses_3d.append({
            'frame': frame_i,
            'translation': np.array(T_world_cpf.translation()),
        })
    return ego_poses_3d

def get_camera_dict(root_path, cam_name, num_frames):
    """
    Loads one camera's first-frame calibration from the pipeline's cameras.npz
    into an OpenCV-style dict (mtx/r/t/dist) for projectPoints.
    """
    cams_npz = np.load(os.path.join(root_path, 'slahmr', 'cameras', "points_triangulated", 'shot-0', 'cameras.npz'))
    cam = {}

    assert cam_name in CAM_NAME2ID, f"Unknown cam_name: {cam_name}"
    cam_id = CAM_NAME2ID[cam_name]
    if cam_name == 'aria':
        # CAM_NAME2ID['aria']=5, but cameras.npz never actually has an aria
        # view (read_cameras.py is always invoked without --use-aria, so the
        # array only has indices 0-4) -- indexing 5 would be out of bounds.
        # This falls back to cam04's slot (index 4) as a placeholder; TODO:
        # fix this hack if a real per-frame aria camera is ever needed here.
        cam_id -= 1
    cam['mtx'] = cams_npz['intrins'][cam_id, 0]
    cam['r'] = cv2.Rodrigues(cams_npz['w2c'][cam_id, 0, :3, :3])[0].T[0][:, None]
    cam['t'] = cams_npz['w2c'][cam_id, 0][:3, 3][:, None]
    cam['dist'] = cams_npz['dist'][cam_id, 0]
    cam['num_frames'] = num_frames
    return cam

if __name__ == "__main__":
    tn = "sfu_cooking025_7"
    mps_data_provider= get_mps(tn)
    rgb_camera_calibration, pinhole = get_calib(tn)
    frame2timestamp = get_frame_timestamps(tn)
    
    K, R, t = get_camera_matricies(mps_data_provider, frame2timestamp[0], rgb_camera_calibration, pinhole)
    print("Camera Matrix K:")
    print(K)
    print("Rotation Matrix R:")
    print(R)
    print("Translation Vector t:")
    print(t)
    print()

    ego_poses_3d = load_ego_poses(tn, 2)
    print("Ego Poses:")
    for pose in ego_poses_3d:
        print(f"Frame: {pose['frame']}, Translation: {pose['translation']}")