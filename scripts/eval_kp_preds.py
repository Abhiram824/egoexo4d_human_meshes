"""
Evaluates pipeline 3D keypoint predictions (triangulated or smooth_fit) against
EgoExo4D ego_pose ground truth, reporting world-frame MPJPE and PA-MPJPE in mm.
"""

from hmr2.utils.pose_utils import  reconstruction_error, eval_pose
import torch
import numpy as np
import cv2
import json
import os
import argparse
from macros import BASE_OUTPUT_PATH, DATASET_BASE_PATH, CAM_NAME2ID   # your project constant
from tqdm import tqdm
from pathlib import Path
from decord import VideoReader
import imageio.v2 as imageio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))

VIDEO_INFO_META = os.path.join(DATASET_BASE_PATH, "takes.json")
VIDEO_DIR = os.path.join(DATASET_BASE_PATH, "takes")
ANNOTATIONS_DIR = os.path.join(DATASET_BASE_PATH, "annotations/ego_pose/train/body/annotation")

# Fill value for GT keypoints absent from the annotation dict. Must never
# collide with a real coordinate: GT 3D is world-frame meters (routinely
# below -1, so -1 is NOT safe), GT 2D is pixels (>= 0).
INVALID_KP_VAL = -1e6

# Per-take smooth_fit results merged across optimizer chunks;
# frame axis is 0-based from video frame 0.
MERGED_NPZ_NAME = "points_triangulated_world_results_merged.npz"



BODY_SLICE = slice(0, 25)  # First 25 keypoints correspond to body in OpenPose format
# taken from https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html?utm_source=chatgpt.com#pose-output-format-body_25
OPENPOSE_KP_NAMES = [
    "nose", "neck", "right-shoulder", "right-elbow", "right-wrist",
    "left-shoulder", "left-elbow", "left-wrist", "mid-hip", "right-hip",
    "right-knee", "right-ankle", "left-hip", "left-knee", "left-ankle",
    "right-eye", "left-eye", "right-ear", "left-ear", "left-big-toe",
    "left-small-toe", "left-heel", "right-big-toe", "right-small-toe", "right-heel"
]


class KeypointEvaluator():

    def __init__(self, video_name, dataset_dir = DATASET_BASE_PATH, visualize=False, vis_output_dir="/data-local/abhicm81/eval_results/vis", preds_root=None):
        """
        Loads take metadata, GT annotations, and prediction frame count for one take.
        """
        self.dataset_dir = Path(dataset_dir)
        self.video_name = video_name
        self.video_dir = Path(preds_root or BASE_OUTPUT_PATH) / self.video_name
        video_meta_path = self.dataset_dir / "takes.json"
        with open(video_meta_path, 'r') as f:
            self.video_info = json.load(f)
        self.num_frames = self._get_num_frames()
        self.video_uid = self._get_video_uid()
        self._load_kp_mappings()
        self._load_annotations()


    def _load_kp_mappings(self):
        body_kps = [
            "nose", "neck", "right-shoulder", "right-elbow", "right-wrist",
            "left-shoulder", "left-elbow", "left-wrist", "mid-hip", "right-hip",
            "right-knee", "right-ankle", "left-hip", "left-knee", "left-ankle",
            "right-eye", "left-eye", "right-ear", "left-ear", "left-big-toe",
            "left-small-toe", "left-heel", "right-big-toe", "right-small-toe", "right-heel"
        ]

        hand_kps = ["wrist", "thumb_1", "thumb_2", "thumb_3", "thumb_4",
                    "index_1", "index_2", "index_3", "index_4",
                    "middle_1", "middle_2", "middle_3", "middle_4",
                    "ring_1", "ring_2", "ring_3", "ring_4",
                    "pinky_1", "pinky_2", "pinky_3", "pinky_4"]
        r_kps = [f"right_{kp}" for kp in hand_kps]
        l_kps = [f"left_{kp}" for kp in hand_kps]

        self.kp_mappings = body_kps + l_kps + r_kps
        self.slices = {
            "body": slice(0, 25),
            "hand": slice(25, 67),
        }
    
    def _get_video_uid(self):
        for v in self.video_info: 
            if v["take_name"] == self.video_name:
                return v["take_uid"]
        raise ValueError(f"Video {self.video_name} not found in metadata.")


    def _load_annotations(self):
        annotations = {
            "body": {},
            "hand": {}
        }
        base_annotations_dir = self.dataset_dir / "annotations" / "ego_pose"
        for ann_type in annotations.keys():
            # a take's manual annotations live in exactly one split (test ships none)
            for split in ["train", "val"]:
                annotations_path = base_annotations_dir / split / ann_type / "annotation" / f"{self.video_uid}.json"
                assert annotations_path.parent.exists(), f"Annotations directory {annotations_path.parent} does not exist."
                if annotations_path.exists():
                    with open(annotations_path, 'r') as f:
                        annotations[ann_type] = json.load(f)
                    break
        self.annotations = annotations
    
    def _get_num_frames(self):
        preds_dir = self.video_dir / "slahmr" / "track_preds" / "points_triangulated"
        if (preds_dir / "001").exists():
            return len([f for f in os.listdir(preds_dir / "001") if f.endswith('.json')])
        # trimmed release dirs (e.g. data_filtered) only carry the merged npz
        return self._load_merged_results()["joints3d"].shape[1]
      
    
    def _load_merged_results(self):
        if not hasattr(self, "_merged_data"):
            npz_path = self.video_dir / MERGED_NPZ_NAME
            if not npz_path.is_file():
                raise ValueError(f"Merged smooth fit results {npz_path} do not exist")
            data = np.load(npz_path)
            assert "joints3d" in data, "Smooth fit results do not contain 'joints3d' key"
            self._merged_data = {"joints3d": data["joints3d"], "valid": data["valid"]}
        return self._merged_data

    def _load_smooth_fit_preds(self, frame_i):
        data = self._load_merged_results()
        # npz arrays are 0-based from video frame 0, same as annotation keys.
        # The merge can be up to a few frames shorter than the take (tail
        # chunks < 10 frames are skipped by the pipeline).
        if frame_i >= data["joints3d"].shape[1]:
            return None
        if not data["valid"][0, frame_i]:
            return None  # frame belongs to a chunk flagged unreliable (NaN loss)
        return data["joints3d"][0][frame_i]

    
    def _get_preds_dict(self, frame_i, kp_type="triangulated"):
        # 3D-only: "triangulated" reads the 001 dir; "smooth_fit" reads the
        # merged optimizer npz. Both return world-frame 3D in the 67-kp order.
        preds_dir = self.video_dir / "slahmr" / "track_preds" / "points_triangulated"
        # annotation frame keys are 0-based video frame indices, while the
        # ffmpeg-extracted pred files are 1-based: pred file (frame_i + 1)
        # shows video frame frame_i
        pred_fname = f"{frame_i + 1:06d}_keypoints.json"
        if kp_type == "triangulated":
            preds_json = preds_dir / "001" / pred_fname
            if not preds_json.exists():
                return None
            return json.load(open(preds_json, 'r'))
        elif kp_type == "smooth_fit":
            preds_array = self._load_smooth_fit_preds(frame_i)
            if preds_array is None:
                return None
            return {"people": [{"pose_keypoints_2d": preds_array.tolist()}]}
        else:
            raise ValueError(f"Unknown kp_type: {kp_type}")
    
    def _get_gt_array(self, gts_dict,joint_type="body"):
        gt_list= []
        jnt_kps = self.kp_mappings[self.slices[joint_type]]
        for i, body_part in enumerate(jnt_kps):
            if body_part not in gts_dict:
                gt_list.append([INVALID_KP_VAL] * 3)
            else:
                gt_part = gts_dict[body_part]
                # official EgoExo4D protocol: manual 3D GT triangulated from
                # < 3 views is unreliable, treat as missing
                if gt_part.get("num_views_for_3d", 3) < 3:
                    gt_list.append([INVALID_KP_VAL] * 3)
                else:
                    gt_list.append([gt_part["x"], gt_part["y"], gt_part["z"]])
        return np.array(gt_list)
    
    def _get_mask_keypoints_3d(self, gt_array, pred_array):
        gt_valid = np.all(gt_array != INVALID_KP_VAL, axis=1)
        pred_valid = (pred_array[:, 0] != 0) & (pred_array[:, 1] != 0) & (pred_array[:, 2] != 0)
        return gt_valid & pred_valid
    
    def _eval_kp_preds_3d(self, preds, gts, thr=0.05, joint_type="body"):
        """
        Returns per-keypoint 3D Euclidean errors (mm) between preds and GT.
        """
        jnt_slice = self.slices[joint_type]
        gt_array = self._get_gt_array(gts, joint_type=joint_type)
        # Extract just the x,y,z coordinates from the prediction (removing confidence if present)
        pred_coords = np.array(preds["people"][0]["pose_keypoints_2d"])[jnt_slice, :]
        if pred_coords.shape[1] > 3:
            pred_coords = pred_coords[:, :3]  # Take only x,y,z

        # Convert from meters to millimeters for error calculation
        pred_coords_mm = pred_coords * 1000
        gt_array_mm = gt_array * 1000
        
        mask = self._get_mask_keypoints_3d(gt_array, pred_coords)
        # Calculate Euclidean distance error in mm
        errors = np.linalg.norm(pred_coords_mm[mask] - gt_array_mm[mask], axis=1)
        return errors

    def _procrustes_error(self, preds, gts, joint_type="body"):
        jnt_slice = self.slices[joint_type]
        gt_array = self._get_gt_array(gts, joint_type=joint_type)
        pred_coords = np.array(preds["people"][0]["pose_keypoints_2d"])[jnt_slice, :]
        mask = self._get_mask_keypoints_3d(gt_array, pred_coords)
        if np.sum(mask) == 0:
            return -1.0, -1.0
        absolute_error, procrustes_error = eval_pose(torch.tensor(pred_coords[mask], dtype=torch.float32).unsqueeze(0), torch.tensor(gt_array[mask], dtype=torch.float32).unsqueeze(0))
        return float(absolute_error), -1.0 if np.isnan(procrustes_error) else float(procrustes_error)


    def get_eval_results(self, joint_type="body", kp_type="triangulated"):
        """
        Evaluates all annotated frames of this take, returning mean MPJPE and
        PA-MPJPE (mm) across valid frames.
        """
        assert self.annotations[joint_type], f"No {joint_type} annotations found for video {self.video_name}"
        results = {"annotation3D": [], "procrustes": []}
        annotations_dict = self.annotations[joint_type]

        for frame_i in range(self.num_frames):
            if str(frame_i) not in annotations_dict:
                    continue
            frame_annotations = annotations_dict[str(frame_i)]
            if len(frame_annotations) != 1:
                print(f"WARNING: Expected one annotation set per frame. got {len(frame_annotations)} for frame {frame_i}, using first one.")

            frame_annotations = frame_annotations[0]
            gt_kps = frame_annotations["annotation3D"]

            preds_dict = self._get_preds_dict(frame_i, kp_type=kp_type)
            if preds_dict is None:
                continue  # no pred file for this frame

            absolute_error, procrustes_error = self._procrustes_error(preds_dict, gt_kps, joint_type=joint_type)
            if procrustes_error != -1.0:
                results["procrustes"].append(procrustes_error)
            if absolute_error != -1.0:
                results["annotation3D"].append(absolute_error)

        return self._postprocess_results(results)

    def _postprocess_results(self, results):
        for key in results:
            if len(results[key]) == 0:
                results[key] = -1.0
            else:
                results[key] = np.mean(results[key])
        return results

    @staticmethod
    def get_averaged_results(all_take_results):
        """Average the results across all takes"""
        averaged = {}
        for metric in ["annotation3D", "procrustes"]:
            vals = np.array([r[metric] for r in all_take_results.values()
                             if r[metric] != -1.0])
            averaged[metric] = {
                "num_takes": int(len(vals)),
                "mean": float(np.mean(vals)) if len(vals) else -1.0,
                "median": float(np.median(vals)) if len(vals) else -1.0,
            }
        return averaged


def get_takes_with_annotations(joint_type="body"):
    with open(VIDEO_INFO_META, 'r') as f:
        video_info = json.load(f)
    annotations_dirs = [os.path.join(DATASET_BASE_PATH, "annotations/ego_pose", split, joint_type, "annotation")
                        for split in ["train", "val"]]
    takes_with_annotations = []
    for v in video_info:
        take_uid = v["take_uid"]
        if any(os.path.exists(os.path.join(d, f"{take_uid}.json")) for d in annotations_dirs):
            takes_with_annotations.append(v["take_name"])
    return takes_with_annotations
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-name', type=str,
                        help='Name of the video file (without extension)')
    parser.add_argument('--output-dir', type=str, default=".",
                        help='Directory where predictions are stored')
    parser.add_argument('--eval-video-dir', type=str, default=BASE_OUTPUT_PATH,
                        help='Directory where evaluation videos are stored')
    parser.add_argument('--kp-type', type=str, default='smooth_fit',
                        help='Type of keypoints to evaluate (triangulated or smooth_fit)')
    parser.add_argument('--joint-type', type=str, default="body",
                        help='Type of joints to evaluate (e.g., body, hand)')
    args = parser.parse_args()
    # completed-take sentinel inside the preds root: smooth_fit reads the
    # merged npz; other kp_types need the full pipeline output dir
    sentinel = MERGED_NPZ_NAME if args.kp_type == "smooth_fit" else "time_log.json"
    if args.video_name is None:
        res = {}
        takes_with_annotations = get_takes_with_annotations(args.joint_type)
        # use all videos in video dir
        video_names = [d for d in os.listdir(args.eval_video_dir) if os.path.isdir(os.path.join(args.eval_video_dir, d))]
        for video_name in tqdm(video_names):
            args.video_name = video_name
            if not video_name in takes_with_annotations:
                print(f"Skipping video {video_name} as no annotations found.")
                continue
            if not os.path.exists(os.path.join(args.eval_video_dir, video_name, sentinel)):
                print(f"Skipping video {video_name} as no {sentinel} found.")
                continue
            print(f"Evaluating video: {video_name}")
            evalulator = KeypointEvaluator(video_name, preds_root=args.eval_video_dir)
            try:
                res[video_name] = evalulator.get_eval_results(joint_type=args.joint_type, kp_type=args.kp_type)
            except Exception as e:
                print(f"Error evaluating video {video_name}: {e}")
        output_dir = os.path.join(args.output_dir, args.kp_type)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{args.joint_type}_kp_eval_results.json"), 'w') as f:
            json.dump(res, f, indent=4)
        averaged = KeypointEvaluator.get_averaged_results(res)
        for metric, stats in averaged.items():
            print(f"{metric}: median {stats['median']:.2f} mm | mean {stats['mean']:.2f} mm "
                  f"over {stats['num_takes']} takes")
        with open(os.path.join(output_dir, f"{args.joint_type}_kp_eval_results_averaged.json"), 'w') as f:
            json.dump(averaged, f, indent=4)
    else:
        evalulator = KeypointEvaluator(args.video_name, preds_root=args.eval_video_dir)
        results = evalulator.get_eval_results(joint_type=args.joint_type, kp_type=args.kp_type)
        print(f"3D MPJPE (world frame): {results['annotation3D']:.2f} mm")
        print(f"3D Procrustes-aligned error: {results['procrustes']:.2f} mm")
