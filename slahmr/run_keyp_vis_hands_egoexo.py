"""
Debug/QA visualization: overlays 2D keypoints on each camera's frames and
renders them to video. Without --single_view_kps_vis, projects the 3D
triangulated keypoints (track_preds/001) into cam01 only; with the flag,
overlays each camera's own per-view detections (track_preds/002-006) instead,
plus the Aria wearer's ego position projected into every exo view.
"""
import cv2
import json
import torch
import trimesh
import numpy as np
import neural_renderer as nr
import os
import sys
import imageio
from tqdm import tqdm
from util.egoexo4d_utils import load_ego_poses
from macros import BASE_OUTPUT_PATH

"""
Render OpenPose keypoints.
Code was ported to Python from the official C++ implementation https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/utilities/keypoint.cpp
"""
import math
from typing import List, Tuple

# Slices for indexing cost arrays
LEFT_HAND_SLICE = slice(25, 46)
RIGHT_HAND_SLICE = slice(46, 67)

def get_keypoints_rectangle(keypoints: np.array, threshold: float) -> Tuple[float, float, float]:
    """
    Compute rectangle enclosing keypoints above the threshold.
    Args:
        keypoints (np.array): Keypoint array of shape (N, 3).
        threshold (float): Confidence visualization threshold.
    Returns:
        Tuple[float, float, float]: Rectangle width, height and area.
    """
    valid_ind = keypoints[:, -1] > threshold
    if valid_ind.sum() > 0:
        valid_keypoints = keypoints[valid_ind][:, :-1]
        max_x = valid_keypoints[:,0].max()
        max_y = valid_keypoints[:,1].max()
        min_x = valid_keypoints[:,0].min()
        min_y = valid_keypoints[:,1].min()
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        return width, height, area
    else:
        return 0,0,0

def render_keypoints(img: np.array,
                     keypoints: np.array,
                     pairs: List,
                     colors: List,
                     thickness_circle_ratio: float,
                     thickness_line_ratio_wrt_circle: float,
                     pose_scales: List,
                     threshold: float = 0.1,
                     alpha: float = 1.0) -> np.array:
    """
    Render keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        keypoints (np.array): Keypoint array of shape (N, 3).
        pairs (List): List of keypoint pairs per limb.
        colors: (List): List of colors per keypoint.
        thickness_circle_ratio (float): Circle thickness ratio.
        thickness_line_ratio_wrt_circle (float): Line thickness ratio wrt the circle.
        pose_scales (List): List of pose scales.
        threshold (float): Only visualize keypoints with confidence above the threshold.
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    img_orig = img.copy()
    width, height = img.shape[1], img.shape[2]
    area = width * height

    lineType = 8
    shift = 0
    numberColors = len(colors)
    thresholdRectangle = 0.1

    person_width, person_height, person_area = get_keypoints_rectangle(keypoints, thresholdRectangle)
    if person_area > 0:
        ratioAreas = min(1, max(person_width / width, person_height / height))
        thicknessRatio = np.maximum(np.round(math.sqrt(area) * thickness_circle_ratio * ratioAreas), 2)
        thicknessCircle = np.maximum(1, thicknessRatio if ratioAreas > 0.05 else -np.ones_like(thicknessRatio))
        thicknessLine = np.maximum(1, np.round(thicknessRatio * thickness_line_ratio_wrt_circle))
        radius = thicknessRatio / 2

        img = np.ascontiguousarray(img.copy())
        for i, pair in enumerate(pairs):
            index1, index2 = pair
            if keypoints[index1, -1] > threshold and keypoints[index2, -1] > threshold:
                thicknessLineScaled = int(round(min(thicknessLine[index1], thicknessLine[index2]) * pose_scales[0]))
                colorIndex = index2
                color = colors[colorIndex % numberColors]
                keypoint1 = keypoints[index1, :-1].astype(np.int32)
                keypoint2 = keypoints[index2, :-1].astype(np.int32)
                cv2.line(img, tuple(keypoint1.tolist()), tuple(keypoint2.tolist()), tuple(color.tolist()), thicknessLineScaled, lineType, shift)
        for part in range(len(keypoints)):
            faceIndex = part
            if keypoints[faceIndex, -1] > threshold:
                radiusScaled = int(round(radius[faceIndex] * pose_scales[0]))
                thicknessCircleScaled = int(round(thicknessCircle[faceIndex] * pose_scales[0]))
                colorIndex = part
                color = colors[colorIndex % numberColors]
                center = keypoints[faceIndex, :-1].astype(np.int32)
                cv2.circle(img, tuple(center.tolist()), radiusScaled, tuple(color.tolist()), thicknessCircleScaled, lineType, shift)
    return img

def render_body_keypoints(img: np.array,
                          body_keypoints: np.array) -> np.array:
    """
    Render OpenPose body keypoints on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """

    thickness_circle_ratio = 1./75. * np.ones(body_keypoints.shape[0])
    thickness_line_ratio_wrt_circle = 0.75
    pairs = []
    pairs = [1,8,1,2,1,5,2,3,3,4,5,6,6,7,8,9,9,10,10,11,8,12,12,13,13,14,1,0,0,15,15,17,0,16,16,18,14,19,19,20,14,21,11,22,22,23,11,24]
    pairs = np.array(pairs).reshape(-1,2)
    colors = [255.,     0.,     85.,
              255.,     0.,     0.,
              255.,    85.,     0.,
              255.,   170.,     0.,
              255.,   255.,     0.,
              170.,   255.,     0.,
               85.,   255.,     0.,
                0.,   255.,     0.,
              255.,     0.,     0.,
                0.,   255.,    85.,
                0.,   255.,   170.,
                0.,   255.,   255.,
                0.,   170.,   255.,
                0.,    85.,   255.,
                0.,     0.,   255.,
              255.,     0.,   170.,
              170.,     0.,   255.,
              255.,     0.,   255.,
               85.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,     0.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.,
                0.,   255.,   255.]
    colors = np.array(colors).reshape(-1,3)
    pose_scales = [1]
    return render_keypoints(img, body_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)

def render_hand_keypoints(img, right_hand_keypoints, threshold=0.1, use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0):
    if use_confidence and map_fn is not None:
        #thicknessCircleRatioLeft = 1./50 * map_fn(left_hand_keypoints[:, -1])
        thicknessCircleRatioRight = 1./50 * map_fn(right_hand_keypoints[:, -1])
    else:
        #thicknessCircleRatioLeft = 1./50 * np.ones(left_hand_keypoints.shape[0])
        thicknessCircleRatioRight = 1./50 * np.ones(right_hand_keypoints.shape[0])
    thicknessLineRatioWRTCircle = 0.75
    pairs = [0,1,  1,2,  2,3,  3,4,  0,5,  5,6,  6,7,  7,8,  0,9,  9,10,  10,11,  11,12,  0,13,  13,14,  14,15,  15,16,  0,17,  17,18,  18,19,  19,20]
    pairs = np.array(pairs).reshape(-1,2)

    colors = [100.,  100.,  100.,
              100.,    0.,    0.,
              150.,    0.,    0.,
              200.,    0.,    0.,
              255.,    0.,    0.,
              100.,  100.,    0.,
              150.,  150.,    0.,
              200.,  200.,    0.,
              255.,  255.,    0.,
                0.,  100.,   50.,
                0.,  150.,   75.,
                0.,  200.,  100.,
                0.,  255.,  125.,
                0.,   50.,  100.,
                0.,   75.,  150.,
                0.,  100.,  200.,
                0.,  125.,  255.,
              100.,    0.,  100.,
              150.,    0.,  150.,
              200.,    0.,  200.,
              255.,    0.,  255.]
    colors = np.array(colors).reshape(-1,3)
    #colors = np.zeros_like(colors)
    poseScales = [1]
    #img = render_keypoints(img, left_hand_keypoints, pairs, colors, thicknessCircleRatioLeft, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    img = render_keypoints(img, right_hand_keypoints, pairs, colors, thicknessCircleRatioRight, thicknessLineRatioWRTCircle, poseScales, threshold, alpha=alpha)
    #img = render_keypoints(img, right_hand_keypoints, pairs, colors, thickness_circle_ratio, thickness_line_ratio_wrt_circle, pose_scales, 0.1)
    return img

def render_openpose(img: np.array,
                    body_keypoints: np.array) -> np.array:
    """
    Render keypoints in the OpenPose format on input image.
    Args:
        img (np.array): Input image of shape (H, W, 3) with pixel values in the [0,255] range.
        body_keypoints (np.array): Keypoint array of shape (N, 3); 3 <====> (x, y, confidence).
    Returns:
        (np.array): Image of shape (H, W, 3) with keypoints drawn on top of the original image. 
    """
    img = render_body_keypoints(img, body_keypoints)
    return img

def main(args):
    main_frame = args.main_frame
    root_path = args.root_path
    if not os.path.exists(root_path):
        root_path = os.path.join(BASE_OUTPUT_PATH, root_path)
    num_render = args.num_render
    threshold = args.threshold
    img_path = os.path.join(root_path, "images")
    single_view_vis = args.single_view_kps_vis
    if args.output_dir is None:
        parent_dir = "single_view_kps_vis" if single_view_vis else "triangulated_kps_vis"
        out_dir = os.path.join(root_path, 'final_vis', parent_dir)
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    video_name = os.path.basename(root_path)
    egoposes = load_ego_poses(video_name, num_render)
    cameras = np.load(os.path.join(root_path, 'slahmr', 'cameras', 'points_triangulated', 'shot-0', 'cameras.npz'))
    cams = {}
    num_views = cameras['intrins'].shape[0]-1  # excluding view 0 (used for triangulation)
    if num_render == -1:
        num_render = cameras['intrins'].shape[1] - main_frame - 1
    if not single_view_vis:
        num_views = 1
    for view in range(1,num_views+1):
        writer = imageio.get_writer(
                os.path.join(out_dir, f'view_{view}_{video_name}.mp4'),
                fps=30, codec='libx264', quality=6
            )

        device = 'cuda'
        dist_coeffs = torch.tensor(cameras['dist'][view,main_frame])
        intrins = cameras['intrins'][view,main_frame]
        w2c = cameras['w2c'][view,main_frame]

        cams[view] = {}
        cams[view]['mtx'] = cameras['intrins'][view,0]
        cams[view]['r'] = cv2.Rodrigues(cameras['w2c'][view,0,:3,:3])[0].T[0][:,None]
        cams[view]['t'] = cameras['w2c'][view,0][:3,3][:,None]
        cams[view]['dist'] = cameras['dist'][view,0]

        if view == 5:
            I = cv2.imread(os.path.join(img_path, 'aria', '%06d.jpg' % (main_frame+1)))
        else:
            I = cv2.imread(os.path.join(img_path, 'cam%02d/%06d.jpg' % (view, main_frame+1)))
        H, W, _ = I.shape
        
        for i in tqdm(range(main_frame, main_frame+num_render)):

            json_path_3d = os.path.join(root_path, "slahmr", "track_preds","points_triangulated", "001", '%06d_keypoints.json' % (i+1))
            pose_keypoints_3d = np.array(json.load(open(json_path_3d, 'rb'))['people'][0]['pose_keypoints_2d'])
            try:
                json_path_2d = os.path.join(root_path, "slahmr", "track_preds", "points_triangulated", '%03d/%06d_keypoints.json' % (view+1, i+1))
                single_view_pose_keypoints_2d = np.array(json.load(open(json_path_2d, 'rb'))['people'][0]['pose_keypoints_2d'])
            except:
                single_view_pose_keypoints_2d = np.zeros([67,3])

            #import ipdb; ipdb.set_trace()
            if view == 5:
                # zero out body keypoints for aria view
                pose_keypoints_3d[:25,:] = 0.0

            if single_view_vis:
                pose_keypoints_2d = single_view_pose_keypoints_2d
            else:
                orig_kps = pose_keypoints_3d.copy()
                # For aria (view 5), use per-frame camera matrices
                if view == 5:
                    intrins = cameras['intrins'][5, i]
                    w2c = cameras['w2c'][5, i]
                pose_keypoints_3d = np.einsum('dc, bc->bd', w2c[:3,:3], pose_keypoints_3d) + w2c[:3,3]
                pose_keypoints_3d = np.einsum('dc, bc->bd', intrins, pose_keypoints_3d)
                pose_keypoints_2d = pose_keypoints_3d/pose_keypoints_3d[:,[2]]
                conf = np.where(np.all(orig_kps == 0, axis=1), 0.0, 1.0)   # shape (N,)
                pose_keypoints_2d = np.hstack([pose_keypoints_2d[:, :2], conf[:, None]])
                # pose_keypoints_2d = np.hstack((pose_keypoints_2d[:, :2], np.expand_dims(single_view_pose_keypoints_2d[:, -1], axis=1)))
        
            if view == 5:
                I = cv2.imread(os.path.join(img_path, 'aria', '%06d.jpg'  % (i+1)))
                # if not single_view_vis:
                #     I = cv2.rotate(I, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                I = cv2.imread(os.path.join(img_path, 'cam%02d/%06d.jpg' %  (view, i+1)))
        
            input_img_overlay = I
            #import ipdb; ipdb.set_trace()

            input_img_overlay = cv2.cvtColor(input_img_overlay, cv2.COLOR_BGR2RGB)
            input_img_overlay = render_hand_keypoints(input_img_overlay, pose_keypoints_2d[-21:], threshold=threshold)
            input_img_overlay = render_hand_keypoints(input_img_overlay, pose_keypoints_2d[-42:-21], threshold=threshold)
            input_img_overlay = render_openpose(input_img_overlay, pose_keypoints_2d[:25])
            # render eyepos
            if i < len(egoposes) and view != 5:
                eyepos_in_view = cv2.projectPoints(egoposes[i]["translation"].reshape(1,3),cams[view]['r'], cams[view]['t'], cams[view]['mtx'], cams[view]['dist'])[0].reshape(2)
                # add rectangle for eyepos
                input_img_overlay = cv2.rectangle(input_img_overlay,
                            (int(eyepos_in_view[0]-5), int(eyepos_in_view[1]-5)),
                            (int(eyepos_in_view[0]+5), int(eyepos_in_view[1]+5)),
                            (255,255,255), -1)
            input_img_overlay = cv2.cvtColor(input_img_overlay, cv2.COLOR_RGB2BGR)
            if view != 5:
                input_img_overlay = cv2.resize(input_img_overlay, (W//2, H//2), interpolation=cv2.INTER_LINEAR)
            input_img_overlay = cv2.cvtColor(input_img_overlay, cv2.COLOR_BGR2RGB)
            # if view == 5 and not single_view_vis:
            #     input_img_overlay = cv2.rotate(input_img_overlay, cv2.ROTATE_90_CLOCKWISE)
            writer.append_data(input_img_overlay.astype(np.uint8))
        writer.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--main_frame', type=int, default=0)
    parser.add_argument("--root_path", type=str, required=True, help="Root path to the dataset.")
    parser.add_argument("--num_render", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for rendering keypoints.")
    parser.add_argument("--single_view_kps_vis", action='store_true')
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for the rendered videos.")
    args = parser.parse_args()

    main(args)
