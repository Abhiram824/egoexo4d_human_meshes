"""
Runs HaMeR hand-mesh recovery on hand crops derived from the merged ViTPose
keypoints and overwrites each hand's 2D keypoints with HaMeR's reprojected
predictions. For the aria view it instead stores per-frame MANO pose parameters
(<frame>_smpl.json in the 001 dir) for use by the optimizer.
"""

import detectron2.data.transforms as T
import torch
import numpy as np
import argparse
from pathlib import Path
import cv2
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from vitdet_batch_dataset import ViTDetBatchDataset
import shutil

import numpy as np
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
import json

from macros import CAM_NAME2ID

BATCH_SIZE=32

def build_hamer_dataset(hamer_cfg, images, vitposes_out):
    """
    Builds a HaMeR crop dataset with one item per confidently-detected hand
    (a frame can contribute 0, 1, or 2 items).
    """
    # maintain mapping of index in dataset to original frame index and whether or not it is a right or left hand
    ds_idx_to_frames = []
    hand_bboxes = []
    right_hands = []
    frames_sorted = sorted(list(images.keys()))
    assert frames_sorted == sorted(vitposes_out.keys()), "Frame keys in images and vitposes_out do not match"

    for i,frame in enumerate(frames_sorted):
        # hand slots of the 67-kp format: left = [-42:-21], right = [-21:]
        vi = vitposes_out[frame]
        right_hand_keyp = vi[-21:]
        left_hand_keyp = vi[-42:-21]
        # A hand is kept only if >3 of its 21 joints are confident (same rule
        # merge_pose_preds.py uses for the aria view); its crop bbox is the
        # tight bounds of the confident joints.
        keyp = left_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            hand_bboxes.append(bbox)
            right_hands.append(0)
            ds_idx_to_frames.append(frame)
        keyp = right_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            hand_bboxes.append(bbox)
            right_hands.append(1)
            ds_idx_to_frames.append(frame)
    # images are passed as Paths and read lazily per item inside the dataset
    # (loading all frames upfront OOMed on long takes); rescale_factor=2 pads
    # the tight keypoint bbox before cropping
    dataset = ViTDetBatchDataset(hamer_cfg, [images[f] for f in ds_idx_to_frames], np.stack(hand_bboxes), np.stack(right_hands), rescale_factor=2)
    return dataset, ds_idx_to_frames, right_hands


def init_hamer():
    """
    Loads the pretrained HaMeR model onto the GPU in eval mode.
    """
    HaMeR, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    HaMeR = HaMeR.to("cuda")
    HaMeR.eval()
    return HaMeR, model_cfg


def load_vitpose_kps(vitpose_kp_dir: Path):
    """
    Loads all merged 67-kp <frame>_keypoints.json files, keyed by frame string.
    """
    kp_files = list(vitpose_kp_dir.glob("*_keypoints.json"))
    sorted_kp_files = sorted(kp_files)
    kps = {}
    for f in sorted_kp_files:
        with open(f, "r") as fp:
            kp_dict = json.load(fp)
            key = f.stem.replace("_keypoints", "")
            kps[key] = kp_dict
    return kps

def main(args):
    """
    Runs HaMeR over one camera's hand crops and rewrites the per-frame
    keypoint JSONs (exo) or writes MANO param JSONs (aria).
    """
    root_dir = Path(args.root_dir)
    seqname = args.seqname
    camid = CAM_NAME2ID[args.cam]
    aria_view = (args.cam == "aria")
    img_dir = root_dir / "images" / args.cam
    # track_preds dir numbering is camid+1: cam01 -> 002, ..., aria -> 006
    vitpose_kp_dir = root_dir / "slahmr" / "track_preds" / seqname / f"{camid+1:03d}"
    mano_outdir = None
    if aria_view:
        # arbitrarily store mano pose params in 001 (the triangulated-points
        # dir), where the optimizer picks them up as <frame>_smpl.json
        mano_outdir = vitpose_kp_dir.parent / "001"
        mano_outdir.mkdir(parents=True, exist_ok=True)

    kp_dicts = load_vitpose_kps(vitpose_kp_dir)
    mano_dicts = {k: {} for k in kp_dicts}
    video_len = len(kp_dicts)

    hamer, hamer_cfg = init_hamer()

    frames = sorted(list(kp_dicts.keys()))
    dataset, ds_idx_to_frames, right_hands = build_hamer_dataset(hamer_cfg, {f: img_dir / f"{f}.jpg" for f in frames}, {k:np.array(v['people'][0]['pose_keypoints_2d']) for k,v in kp_dicts.items()})
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


    pred_keypoints_2d_all = []
    mano_preds_all = []
    for batch in tqdm(dataloader):
        batch = recursive_to(batch, 'cuda')
        with torch.no_grad():
            out = hamer(batch)
        box_center = batch['box_center']
        box_size = batch['box_size']
        right = batch['right']
        # HaMeR predicts in normalized crop coords for a right hand: un-mirror
        # x for left hands ((2*right-1) is -1 for left, +1 for right), then
        # scale/translate from crop space back to full-image pixels.
        pred_keypoints_2d = out['pred_keypoints_2d']
        pred_keypoints_2d[:,:,0] = (2*right[:,None]-1)*pred_keypoints_2d[:,:,0]
        pred_keypoints_2d = pred_keypoints_2d*box_size[:,None,None]+box_center[:,None]
        pred_keypoints_2d = pred_keypoints_2d.cpu().numpy()
        pred_keypoints_2d_all.append(pred_keypoints_2d)

        # (15, 3, 3) per-joint MANO rotation matrices
        hand_pose = out['pred_mano_params']['hand_pose'].cpu().numpy()
        mano_preds_all.append(hand_pose)

    pred_keypoints_2d_all = np.concatenate(pred_keypoints_2d_all, axis=0)
    mano_preds_all = np.concatenate(mano_preds_all, axis=0)

    # Write HaMeR's reprojected joints back over the hand slots (xy only —
    # the original ViTPose confidences are kept), and convert MANO rotation
    # matrices to axis-angle for the optimizer.
    for idx, frame in enumerate(ds_idx_to_frames):
        original_keypoints = np.array(kp_dicts[frame]['people'][0]['pose_keypoints_2d'])
        hamer_pred = pred_keypoints_2d_all[idx]
        hamer_pred_mano = mano_preds_all[idx]
        if right_hands[idx]:
            original_keypoints[-21:,:2] = hamer_pred
            right_hand_pose_aa = np.stack(
                [cv2.Rodrigues(x)[0].squeeze() for x in hamer_pred_mano], axis=0
            )
            mano_dicts[frame]['right_hand_pose'] = right_hand_pose_aa.tolist()
        else:
            original_keypoints[-42:-21,:2] = hamer_pred
            left_hand_pose_aa = np.stack(
                [cv2.Rodrigues(x)[0].squeeze() for x in hamer_pred_mano], axis=0
            )
            # HaMeR always outputs right-hand MANO poses; mirror to a left
            # hand by negating the y/z axis-angle components
            left_hand_pose_aa[:,1:] *= -1
            mano_dicts[frame]['left_hand_pose'] = left_hand_pose_aa.tolist()

        kp_dicts[frame]['people'][0]['pose_keypoints_2d'] = original_keypoints.tolist()

    output_dir = root_dir / "slahmr" / "track_preds" / seqname / f"{camid+1:03d}" if not aria_view else mano_outdir
    for frame in frames:
        outfile = output_dir / f"{frame}_keypoints.json" if not aria_view else output_dir / f"{frame}_smpl.json"
        if aria_view:
            # only frames where at least one hand passed the confidence gate
            # have MANO params worth writing
            if frame not in mano_dicts or not mano_dicts[frame]:
                continue
            with open(outfile, "w") as f:
                json.dump(mano_dicts[frame], f, indent=4)
        elif frame not in kp_dicts:
            zero_kps = np.zeros((67, 3))
            kp_dict = {
                "people": [{"pose_keypoints_2d": zero_kps.tolist()}]}
            with open(outfile, "w") as f:
                json.dump(kp_dict, f, indent=4)
        else:
            with open(outfile, "w") as f:
                json.dump(kp_dicts[frame], f, indent=4)

    if aria_view:
        # remove the aria detections dir (006): it was only needed to crop
        # hands for HaMeR, the MANO params are stored now, and triangulation
        # only reads the exo dirs (002-005)
        shutil.rmtree(vitpose_kp_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--cam", type=str, required=True)
    parser.add_argument("--seqname", type=str, required=True)
    args = parser.parse_args()

    main(args)