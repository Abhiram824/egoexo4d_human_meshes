"""
Stage 3: triangulates the 67 per-camera 2D keypoints (from track_preds 002-006)
into 3D world points using each frame's cameras.npz, writing pseudo-2D XYZ
output to track_preds/<seq>/001. Body keypoints use all confident views;
hand keypoints run a RANSAC-style search over view subsets to reject outliers.
"""
import os
import cv2
import json
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations
from multiprocessing import Pool, cpu_count
import argparse
from tqdm import tqdm
from macros import BASE_OUTPUT_PATH

# Hand keypoint index ranges
LEFT_HAND_RANGE = range(25, 46)
RIGHT_HAND_RANGE = range(46, 67)
WRIST_KEYPOINTS = {4,7}
HAND_KEYPOINTS = set(LEFT_HAND_RANGE) | set(RIGHT_HAND_RANGE) | WRIST_KEYPOINTS

# Slices for indexing cost arrays
LEFT_HAND_SLICE = slice(25, 46)
RIGHT_HAND_SLICE = slice(46, 67)
THRESH = 1000  # max least_squares cost accepted for a hand-keypoint subset

def project_point_aria(point_3d, w2c, intrins):
    """Manual pinhole projection for the Aria view (view 5)."""
    point_cam = w2c[:3,:3] @ point_3d + w2c[:3,3]
    point_2d_h = intrins @ point_cam
    return point_2d_h[:2] / point_2d_h[2]

def fun_rosenbrock(x,imgpoints,cams):
    """Residuals (reprojection error per view) for a candidate 3D point x, used by least_squares."""
    residuals=[]
    for viewid in cams:
        if viewid not in imgpoints:
            continue
        # Use direct projection for Aria camera (view 5) since its intrinsic matrix
        # has off-diagonal focal lengths that cv2.projectPoints can't handle
        if viewid == 5:
            imgpoints2 = project_point_aria(x, cams[viewid]['w2c'], cams[viewid]['mtx'])
        else:
            imgpoints2, _ = cv2.projectPoints(x, cams[viewid]['r'], cams[viewid]['t'], cams[viewid]['mtx'], cams[viewid]['dist'])
            imgpoints2 = np.array(imgpoints2).squeeze()
        residuals.extend(list(imgpoints2-imgpoints[viewid]))
    residuals=np.array(residuals)
    return residuals

def triangulate_frame(frame_args):
    """Process a single frame: triangulate all keypoints and save output."""
    frame_i, track_path, seqname, num_views, cams, out_dir = frame_args

    # track_preds dir numbering is view index + 1 (cam01/view1 -> 002, ..., aria/view5 -> 006)
    json_contents = {}
    for i in range(1, num_views+1):
        json_file = os.path.join(track_path, seqname, '%03d' % (i+1), '%06d_keypoints.json' % frame_i)
        if os.path.exists(json_file):
            json_contents[i] = json.load(open(json_file, 'rb'))

    out3D = np.zeros([67, 3])  # stays [0,0,0] (the failure sentinel) if <2 confident views
    kp_costs = np.full(67, np.inf)
    frame_costs = []
    best_subsets = {}  # per-hand-keypoint: which views were kept (used to zero out the rest below)
    for keypoint in range(67):
        imgpoints = {}
        cams_opt = {}
        valid_views = []
        # body keypoints: confidence > 0.6 gate, use every confident view
        for viewid in range(1, num_views+1):
            if viewid in json_contents and json_contents[viewid]['people'][0]['pose_keypoints_2d'][keypoint][2] > 0.6:
                imgpoints[viewid] = np.array(json_contents[viewid]['people'][0]['pose_keypoints_2d'][keypoint][:2]) / 1.
                cams_opt[viewid] = cams[viewid]
                valid_views.append(viewid)
        if len(valid_views) > 1:
            best_cost = np.inf
            best_x = None
            if keypoint in HAND_KEYPOINTS:
                # RANSAC-style: try every subset (size >=2) of confident views,
                # keep the cheapest least_squares solve under THRESH — rejects
                # views whose hand detection is an outlier for this keypoint
                for r in range(2, len(valid_views)+1):
                    for subset in combinations(valid_views, r):
                        sub_imgpoints = {v: imgpoints[v] for v in subset}
                        sub_cams = {v: cams_opt[v] for v in subset}
                        init_ans = np.array([0, 0, 0]).astype('float32')
                        res = least_squares(fun_rosenbrock, init_ans, args=(sub_imgpoints, sub_cams))
                        if res.cost < best_cost and res.cost < THRESH:
                            best_cost = res.cost
                            best_x = res.x
                            best_subsets[keypoint] = set(subset)
                if best_x is not None:
                    frame_costs.append(best_cost)
                    kp_costs[keypoint] = best_cost
                    out3D[keypoint] = best_x
            else:
                # body keypoints: single least_squares solve over all confident views
                init_ans = np.array([0, 0, 0]).astype('float32')
                res = least_squares(fun_rosenbrock, init_ans, args=(imgpoints, cams_opt))
                best_cost = res.cost
                best_x = res.x
                frame_costs.append(best_cost)
                kp_costs[keypoint] = best_cost
                out3D[keypoint] = best_x

    # zero out confidence for any hand keypoint/view not in its winning RANSAC
    # subset, so downstream consumers of the per-camera JSONs see only the
    # views that actually agreed
    for viewid in json_contents:
        for keypoint in HAND_KEYPOINTS:
            if keypoint not in best_subsets or viewid not in best_subsets[keypoint]:
                json_contents[viewid]['people'][0]['pose_keypoints_2d'][keypoint][2] = 0.0
        json_file = os.path.join(track_path, seqname, '%03d' % (viewid+1), '%06d_keypoints.json' % frame_i)
        with open(json_file, 'w') as f:
            json.dump(json_contents[viewid], f, indent=1)

    left_costs = kp_costs[LEFT_HAND_SLICE]
    right_costs = kp_costs[RIGHT_HAND_SLICE]
    left_valid = left_costs[~np.isinf(left_costs)]
    right_valid = right_costs[~np.isinf(right_costs)]
    left_avg = float(np.mean(left_valid)) if len(left_valid) > 0 else None
    right_avg = float(np.mean(right_valid)) if len(right_valid) > 0 else None

    # pseudo-2D output: 3D XYZ stored in the pose_keypoints_2d slot for run_opt.py
    out_name = os.path.join(out_dir, '%06d_keypoints.json' % frame_i)
    kp_dict = {"people": [{"pose_keypoints_2d": out3D.tolist()}]}
    with open(out_name, 'w') as f:
        json.dump(kp_dict, f, indent=1)

    return frame_i, frame_costs, left_avg, right_avg


def triangulate_points(args):
        # sequence of interest
    seqname = args.seqname

    root_path = args.root_path
    track_path = os.path.join(root_path, 'slahmr', 'track_preds')

    video_len = args.video_len
    # this is where the final triangulated data will go
    os.makedirs(os.path.join(track_path, seqname, '001'), exist_ok=True)
    cams_npz = np.load(os.path.join(root_path, 'slahmr', 'cameras', seqname, 'shot-0', 'cameras.npz'))
    num_views = cams_npz['intrins'].shape[0]-1  # excluding view 0 (used for triangulation)
    if video_len == -1:
        video_len = cams_npz['intrins'].shape[1]

    # exo cams (views 1-4) are static: build their camera dicts once, reused every frame
    cams = {}
    for viewid in range(1,5):
        cams[viewid] = {}
        cams[viewid]['mtx'] = cams_npz['intrins'][viewid,0]
        cams[viewid]['r'] = cv2.Rodrigues(cams_npz['w2c'][viewid,0,:3,:3])[0].T[0][:,None]
        cams[viewid]['t'] = cams_npz['w2c'][viewid,0][:3,3][:,None]
        cams[viewid]['dist'] = cams_npz['dist'][viewid,0]
    # Aria camera (needs to be updated every iteration since its moving)
    cams[5] = {}

    # Build per-frame camera dicts (Aria cam changes per frame) and args list
    out_dir = os.path.join(track_path, seqname, '001')
    frame_args_list = []
    for frame_i in range(1, video_len):
        # Deep copy cams so each frame has its own snapshot
        frame_cams = {}
        for viewid in range(1, 5):
            frame_cams[viewid] = dict(cams[viewid])
        if num_views == 5:
            frame_cams[5] = {
                'mtx': cams_npz['intrins'][5, frame_i-1],
                'w2c': cams_npz['w2c'][5, frame_i-1],
                'r': cv2.Rodrigues(cams_npz['w2c'][5, frame_i-1, :3, :3])[0].T[0][:, None],
                't': cams_npz['w2c'][5, frame_i-1][:3, 3][:, None],
                'dist': cams_npz['dist'][5, frame_i-1],
            }
        frame_args_list.append((frame_i, track_path, seqname, num_views, frame_cams, out_dir))

    # Process frames in parallel (each frame is independent)
    num_workers = min(10, len(frame_args_list))
    costs = []
    hand_errors_per_frame = {}
    with Pool(num_workers) as pool:
        for frame_i, frame_costs, left_avg, right_avg in tqdm(
            pool.imap_unordered(triangulate_frame, frame_args_list),
            total=len(frame_args_list),
        ):
            costs.extend(frame_costs)
            hand_errors_per_frame[frame_i] = {
                "left_hand_avg_error": left_avg,
                "right_hand_avg_error": right_avg,
            }

    # Save per-frame hand reprojection errors to 001 folder (sorted by frame)
    hand_errors_out = os.path.join(track_path, seqname, '001', 'hand_reprojection_errors.json')
    hand_errors_per_frame = dict(sorted(hand_errors_per_frame.items(), key=lambda x: x[0]))
    with open(hand_errors_out, 'w') as f:
        json.dump(hand_errors_per_frame, f, indent=1)
   
    # if there are no detection for some frames, just add zero detections for all points
    # (view_i here is the raw dir number, not a view id: 001=triangulated, 002-005/006=cams -
    # covers every dir triangulate_frame could have skipped, including 001 itself)
    out2D = np.zeros([67,3])
    for frame_i in range(1,video_len):
        for view_i in range(1,num_views+2):
            out_name = os.path.join(track_path, seqname, '%03d' % view_i, '%06d_keypoints.json' % frame_i)
            if not os.path.exists(out_name):
                kp_dict = {"people": [{"pose_keypoints_2d": out2D.tolist(),}]}
                with open(out_name, 'w') as f:
                    json.dump(kp_dict, f, indent=1)

    # some final bookkeeping before running the multi-view optimization:
    # run_opt.py expects an images/<seq> dir and a shot_idcs/<seq>.json (both
    # keyed by cam01, which is what stage 0 actually wrote to disk)
    os.system('cp -r %s %s' % (os.path.join(root_path, 'images', 'cam01'), os.path.join(root_path, 'images', seqname)))
    os.system('cp -r %s %s' % (os.path.join(root_path, 'slahmr', 'shot_idcs', 'cam01.json'), os.path.join(root_path, 'slahmr', 'shot_idcs', '%s.json' % seqname)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare tracklet data for multi-view optimization.")
    parser.add_argument("--seqname", type=str, default="points_triangulated", help="Name of the sequence to process.")
    parser.add_argument("--root_path", type=str, required=True, help="Root path to the dataset.")
    parser.add_argument("--video_len", type=int, default=-1, help="Length of the video in frames.")

    args = parser.parse_args()

    if not os.path.exists(args.root_path):
        args.root_path = os.path.join(BASE_OUTPUT_PATH, args.root_path)
    triangulate_points(args)