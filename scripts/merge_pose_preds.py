"""
Merges per-camera ViTPose body + wholebody predictions into the pipeline's
67-keypoint format (25 OpenPose body + 21 left hand + 21 right hand) and picks
the person closest to the projected Aria ego position, writing one OpenPose-style
<frame>_keypoints.json per frame.
"""

import numpy as np
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from util.egoexo4d_utils import load_ego_poses, get_best_tracklet_for_frame, get_camera_dict
import os
import json
from macros import CAM_NAME2ID

def flatten_batch_results(batch_list):
    """
    Flattens batched ViTPose inference output into (image_path, keypoints) pairs.
    """
    flattened = []

    for batch in batch_list:
        preds = batch.get('preds', [])
        image_paths = batch.get('image_paths', [])

        # Each prediction corresponds to one person in one image
        for pred_idx, keypoints in enumerate(preds):
            img_path = image_paths[pred_idx]
            flattened.append((img_path, keypoints))

    return flattened

def load_vitpose_results(path):
    """
    Loads a ViTPose output JSON and flattens it to (image_path, keypoints) pairs.
    """
    with open(path, "r") as f:
        batch_results = json.load(f)
    flat_results = flatten_batch_results(batch_results)
    return flat_results

def merge_pose_preds(vitpose_results, vitpose_wholebody_results):
    """
    Combines body-model and wholebody-model predictions into per-frame lists of
    67-kp arrays ({frame_key: [(67, 3) per person]}), one entry per detected person.
    """
    merged_results = {}
    for (img_path1, body_keypoints), (img_path2, wholebody_keypoints) in zip(vitpose_results, vitpose_wholebody_results):
        assert img_path1 == img_path2, "Image paths do not match"
        # get filename without extension and parent directory
        key = os.path.splitext(os.path.basename(img_path1))[0]
        if key not in merged_results:
            merged_results[key] = []

        vitpose_mixed = np.zeros([67, 3])
        vitpose_mixed[
            [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
        ] = body_keypoints
        vitpose_mixed[19:25] = wholebody_keypoints[17:23]
        left_hand_keyp = wholebody_keypoints[-42:-21]
        right_hand_keyp = wholebody_keypoints[-21:]
        vitpose_mixed[-21:] = right_hand_keyp
        vitpose_mixed[-42:-21] = left_hand_keyp
        merged_results[key].append(vitpose_mixed)
    return merged_results


def main(args):
    """
    Merges the two ViTPose outputs for one camera and writes one 67-kp
    keypoints JSON per frame, keeping the person nearest the ego position.
    """
    root_dir = Path(args.root_dir)
    seqname = args.seqname
    camid = CAM_NAME2ID[args.cam]
    img_dir = root_dir / "images" / args.cam
    # track_preds dir numbering is camid+1: cam01 -> 002, ..., aria -> 006
    output_dir = root_dir / "slahmr" / "track_preds" / seqname / f"{camid+1:03d}"
    vitpose_wholebody_path = output_dir / "vitpose_wholebody_out.json"
    vitpose_body_path = output_dir / "vitpose_out.json"

    vitpose_results = load_vitpose_results(vitpose_body_path)
    vitpose_wholebody_results = load_vitpose_results(vitpose_wholebody_path)
    pose_results = merge_pose_preds(vitpose_results, vitpose_wholebody_results)


    video_name = root_dir.stem
    images = list(img_dir.glob("*.jpg"))
    video_len = len(images)
    cam = get_camera_dict(args.root_dir, args.cam, video_len)
    ego_poses_3d = load_ego_poses(video_name, video_len)


    for frame_num in tqdm(range(video_len)):
        # frame files are 1-based (ffmpeg extraction); frame_num is 0-based
        key = f"{frame_num+1:06d}"
        if key not in pose_results:
            continue
        if frame_num > len(ego_poses_3d)-1:
            if frame_num - 5 >= len(ego_poses_3d)-1:
                raise ValueError(f"Frame {frame_num} exceeds available ego poses. Stopping further processing.")
            else:
                ego_pose_3d = ego_poses_3d[-1]["translation"]
        else:
            ego_pose_3d = ego_poses_3d[frame_num]["translation"]
        # Project the camera wearer's 3D head position into this view and keep
        # the detected person closest to it (rejects bystanders).
        ego_pose_2d = cv2.projectPoints(ego_pose_3d.reshape(1, 3), cam['r'], cam['t'], cam['mtx'], cam['dist'])[0].reshape(2)
        bodypose_list = pose_results[key]
        bodypose_candidates = [{"keypoints":bodypose_list[i]} for i in range(len(bodypose_list))]
        final_bodypose = get_best_tracklet_for_frame(bodypose_candidates, ego_pose_2d)
        if final_bodypose is None:
            final_bodypose = np.zeros((67, 3))
        else:
            final_bodypose = final_bodypose['keypoints']

        if camid == 5:
            # The aria view never sees the wearer's body: zero out body
            # keypoints and keep a hand only if >3 of its joints are confident.
            final_bodypose[:25,:] = 0.0
            right_hand_keyp = final_bodypose[-21:]
            left_hand_keyp = final_bodypose[-42:-21]
            if np.sum(left_hand_keyp[:,2] > 0.5) <= 3:
                final_bodypose[-42:-21,:] = 0.0
            if np.sum(right_hand_keyp[:,2] > 0.5) <= 3:
                final_bodypose[-21:,:] = 0.0

        # OpenPose-style output consumed by triangulate_points.py
        tracklet_dict = {
            "people": [{"pose_keypoints_2d": final_bodypose.tolist()}]
        }
        outfile = output_dir / f"{frame_num+1:06d}_keypoints.json"
        with open(outfile, "w") as f:
            json.dump(tracklet_dict, f, indent=4)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--cam", type=str, required=True)
    parser.add_argument("--seqname", type=str, required=True)
    args = parser.parse_args()

    main(args)