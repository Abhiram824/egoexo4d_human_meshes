"""
Stage-2 person detection for one camera: detects people, keeps the boxes that
contain the projected Aria ego position, and writes coco_bboxes.json +
dummy_coco.json (consumed by the ViTPose step) plus per-frame placeholder
keypoint JSONs. The aria view skips detection and gets a whole-image bbox.
"""

import detectron2.data.transforms as T
import torch
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, instantiate
from detectron2.data import MetadataCatalog
from omegaconf import OmegaConf
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
from util.egoexo4d_utils import load_ego_poses, get_camera_dict
import json
from macros import CAM_NAME2ID

DET_BATCH_SIZE = 32
THRESHOLD_BOX = 0.8
IMG_SIZE = (1088, 1920)


class YOLOWrapper:
    """
    Adapts an ultralytics YOLO model to detectron2's {"instances": Instances}
    output format so postprocess_detections() works with every detector.
    """
    def __init__(self, model, score_thresh=THRESHOLD_BOX, imgsz=IMG_SIZE, device="cuda:0"):
        self.model = model
        self.score_thresh = score_thresh
        self.imgsz = imgsz
        self.classes = [0]  # COCO class 0 = person
        self.device = device

    @torch.no_grad()
    def __call__(self, original_images):
        if isinstance(original_images, np.ndarray):
            images = [original_images]
        else:
            images = list(original_images)

        results = self.model.predict(
            source=images,
            imgsz=self.imgsz,           
            conf=self.score_thresh,
            device=self.device,
            verbose=False,
            classes=self.classes,       
            batch=DET_BATCH_SIZE,       
        )

        from detectron2.structures import Boxes, Instances

        predictions = []
        for img, r in zip(images, results):
            h, w = img.shape[:2]
            inst = Instances(image_size=(h, w))

            xyxy   = r.boxes.xyxy.detach().cpu()
            scores = r.boxes.conf.detach().cpu()
            clses  = r.boxes.cls.detach().cpu().to(torch.int64)

            inst.pred_boxes   = Boxes(xyxy)
            inst.scores       = scores 
            inst.pred_classes = clses

            predictions.append({"instances": inst})
        return predictions if isinstance(original_images, (list, tuple)) else predictions[0]

class DefaultPredictor_Lazy_Batched:
    """
    detectron2 DefaultPredictor variant that accepts LazyConfig configs and
    batched (list-of-images) inference.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: a yacs CfgNode or a omegaconf dict object.
        """
        if isinstance(cfg, CfgNode):
            self.cfg = cfg.clone()  # cfg can be modified by model
            self.model = build_model(self.cfg)  # noqa: F821
            if len(cfg.DATASETS.TEST):
                test_dataset = cfg.DATASETS.TEST[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )

            self.input_format = cfg.INPUT.FORMAT
        else:  # new LazyConfig
            self.cfg = cfg
            self.model = instantiate(cfg.model)
            test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
            if isinstance(test_dataset, (list, tuple)):
                test_dataset = test_dataset[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

            mapper = instantiate(cfg.dataloader.test.mapper)
            self.aug = mapper.augmentations
            self.input_format = mapper.image_format

        self.model.eval().cuda()
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_images):
        """
        Args:
            original_images (np.ndarray or list): 
                - Single image: an array of shape (H, W, C) (in BGR order)
                - Batch of images: a list of arrays, each of shape (H, W, C) (in BGR order)

        Returns:
            predictions (dict or list):
                - Single image: the output of the model for one image
                - Batch: list of predictions, one for each image
                See :doc:`/tutorials/models` for details about the format.
        """
        # Handle single image case (backward compatibility)
        if isinstance(original_images, np.ndarray):
            return self._predict_single(original_images)
        
        # Handle batch case
        if isinstance(original_images, (list, tuple)):
            return self._predict_batch(original_images)
        
        raise ValueError("Input must be a numpy array (single image) or list/tuple of arrays (batch)")
    
    def _predict_single(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
    
    def _predict_batch(self, original_images, max_batch_size=4096):
        """
        Process a batch of images with optional chunking for memory efficiency
        
        Args:
            original_images: list of images
            max_batch_size: maximum number of images to process at once (to manage memory)
        """
        with torch.no_grad():
            all_predictions = []
            
            # Process in chunks if batch is too large
            for i in range(0, len(original_images), max_batch_size):
                chunk = original_images[i:i + max_batch_size]
                batch_inputs = []
                
                for original_image in chunk:
                    if self.input_format == "RGB":
                        original_image = original_image[:, :, ::-1]
                    height, width = original_image.shape[:2]
                    image = self.aug(T.AugInput(original_image)).apply_image(original_image)
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    inputs = {"image": image, "height": height, "width": width}
                    batch_inputs.append(inputs)
                
                # Process the chunk
                chunk_predictions = self.model(batch_inputs)
                all_predictions.extend(chunk_predictions)
            
            return all_predictions

def load_detector(body_detector='vitdet'):
    """
    Loads one of the supported person detectors (vitdet / regnety / yolo11),
    all returning detectron2-style predictions.
    """
    if body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy_Batched(detectron2_cfg)
    elif body_detector == 'regnety':
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy_Batched(detectron2_cfg)
    elif body_detector == 'yolo11':
        from ultralytics import YOLO
        model = YOLO("yolo11n.pt") 
        detector = YOLOWrapper(model, score_thresh=THRESHOLD_BOX, device="cuda:0", imgsz=IMG_SIZE)

    else:
        raise ValueError(f"Unknown body_detector: {body_detector}")
    return detector

def get_bboxes_containing_point(bboxes, point_2d, cam):
    """
    Returns the boxes (and their indices) that contain point_2d — the bbox
    selection criterion: the projected ego position must fall inside the box.
    """
    x, y = point_2d
    bboxes_containing_point = []
    idcs_containing_point = []
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            bboxes_containing_point.append(bbox)
            idcs_containing_point.append(i)
    return bboxes_containing_point, idcs_containing_point

def postprocess_detections(outputs):
    """
    Filters detector output to confident person detections, returning
    (bboxes xyxy, scores, classes) as numpy arrays.
    """
    instances   = outputs['instances']
    instances   = instances[instances.pred_classes==0]
    instances   = instances[instances.scores>THRESHOLD_BOX]

    pred_bbox   = instances.pred_boxes.tensor.cpu().numpy()
    pred_scores = instances.scores.cpu().numpy()
    pred_classes= instances.pred_classes.cpu().numpy()

    return pred_bbox, pred_scores, pred_classes



def write_single_shot_idcs_json(root_dir, camid):
    """
    Writes the SLAHMR shot-index JSON mapping every frame to shot 0
    (EgoExo4D takes are single, uncut shots).
    """
    # assume for all videos its only one shot!
    root_dir = Path(root_dir)
    img_dir = root_dir / "images" / f"cam{camid:02d}"
    images = list(img_dir.glob("*.jpg"))
    video_len = len(images)
    sorted_images = sorted(images)
    shot_idcs = {}
    for img in sorted_images:
        shot_idcs[img.name] = 0  # only one shot, so all 0
    outdir = root_dir / "slahmr" / "shot_idcs" 
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"cam{camid:02d}.json", "w") as f:
        json.dump(shot_idcs, f, indent=4)

def write_dummy_bbox_json(root_dir, camid, video_len, seqname, img_size, img_files):
    """
    Write dummy bounding boxes (whole image) for aria camera view.
    This creates bbox files where each bbox covers the entire image.
    """
    root_dir = Path(root_dir)
    output_dir = root_dir / "slahmr" / "track_preds" / seqname / f"{camid+1:03d}"
    
    img_w, img_h = img_size
    # Dummy bbox covering the whole image: [x1, y1, x2, y2]
    dummy_bbox = [0.0, 0.0, float(img_w), float(img_h)]
    
    coco_bbox_list = []
    
    for frame_num in range(video_len):
        # Write per-frame keypoints JSON (with bbox as pose_keypoints_2d placeholder)
        tracklet_dict = {
            "people": [{"pose_keypoints_2d": dummy_bbox}],
        }
        outfile = output_dir / f"{frame_num+1:06d}_keypoints.json"
        with open(outfile, "w") as f:
            json.dump(tracklet_dict, f, indent=4)
        
        # COCO format bbox: [x, y, width, height]
        coco_bbox = {
            "image_id": frame_num,
            "category_id": 1,
            "bbox": [0.0, 0.0, float(img_w), float(img_h)],
            "score": 0.5,  # arbitrary score
        }
        coco_bbox_list.append(coco_bbox)
    
    # Save COCO bbox detections
    coco_file_json = output_dir / "coco_bboxes.json"
    with open(coco_file_json, "w") as f:
        json.dump(coco_bbox_list, f)
    
    # Create dummy COCO annotation file
    dummy_coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }
    
    for idx, img in enumerate(img_files):
        dummy_coco["images"].append({
            "id": idx,
            "file_name": img.name,
            "width": img_w,
            "height": img_h,
        })
    
    dummy_ann_path = output_dir / "dummy_coco.json"
    with open(dummy_ann_path, "w") as f:
        json.dump(dummy_coco, f)

def main(args):
    """
    Detects people in every frame of one camera and keeps the detections
    containing the projected ego position (the camera wearer).
    """
    root_dir = Path(args.root_dir)
    seqname = args.seqname
    camid = CAM_NAME2ID[args.cam]
    img_dir = root_dir / "images" / args.cam
    # track_preds dir numbering is camid+1: cam01 -> 002, ..., aria -> 006
    output_dir = root_dir / "slahmr" / "track_preds" / seqname / f"{camid+1:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)


    video_name = root_dir.stem
    images = list(img_dir.glob("*.jpg"))
    video_len = len(images)
    sorted_images = sorted(images)
    write_single_shot_idcs_json(root_dir, camid)

    if args.cam == "aria":
        # the wearer's hands can be anywhere in the ego view: skip detection
        # and use the whole image as the bbox
        write_dummy_bbox_json(root_dir, camid, video_len, seqname, img_size=(512, 512), img_files=sorted_images)
        return

    cam = get_camera_dict(args.root_dir, args.cam, video_len)
    ego_poses_3d = load_ego_poses(video_name, video_len)
    detector = load_detector(args.detector)

    # Read first image once to get shape (all frames share the same resolution)
    first_img = cv2.imread(str(sorted_images[0]))
    img_h, img_w = first_img.shape[:2]

    coco_bbox_list = []

    for i in tqdm(range(0, len(sorted_images), DET_BATCH_SIZE)):
        batch_imgs = [cv2.imread(str(f)) for f in sorted_images[i:i+DET_BATCH_SIZE]]
        batch_predictions = detector(batch_imgs)

        # Process each image in the batch individually
        for j, prediction in enumerate(batch_predictions):
            frame_num = i + j
            assert frame_num < len(sorted_images), f"frame_num: {frame_num}, len: {len(sorted_images)}"
            bboxes, scores, classes = postprocess_detections(prediction)
            if frame_num > len(ego_poses_3d)-1:
                if frame_num - 5 >= len(ego_poses_3d)-1:
                    raise ValueError(f"Frame {frame_num} exceeds available ego poses. Stopping further processing.")
                else:
                    ego_pose_3d = ego_poses_3d[-1]["translation"]
            else:
                ego_pose_3d = ego_poses_3d[frame_num]["translation"]

            # Keep every detection whose box contains the projected wearer
            # position; if several remain, the ambiguity is resolved after
            # ViTPose by merge_pose_preds.py (nearest-to-ego selection).
            point_2d = cv2.projectPoints(ego_pose_3d.reshape(1, 3), cam['r'], cam['t'], cam['mtx'], cam['dist'])[0].reshape(2)
            chosen_bboxes, chosen_idxs = get_bboxes_containing_point(bboxes, point_2d, cam)
            chosen_scores = scores[chosen_idxs]

            if len(chosen_bboxes) > 0:
                # placeholder JSON storing the bbox in the pose_keypoints_2d
                # field; overwritten with real 67-kp poses by merge_pose_preds
                tracklet_dict = {
                    "people": [{"pose_keypoints_2d": bbox.tolist()} for bbox in chosen_bboxes],
                }
                for bbox, score in zip(chosen_bboxes, chosen_scores):
                    coco_bbox = {
                        "image_id": frame_num,
                        "category_id": 1,
                        "bbox": [
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2] - bbox[0]),
                            float(bbox[3] - bbox[1]),
                        ],
                        "score": float(score),
                    }
                    coco_bbox_list.append(coco_bbox)
                outfile = output_dir / f"{frame_num+1:06d}_keypoints.json"
                with open(outfile, "w") as f:
                    json.dump(tracklet_dict, f, indent=4)
            else:
                print("No bbox found for frame:", frame_num)
        
    # Save coco bbox detections (the ViTPose step runs pose estimation on
    # exactly these boxes)
    coco_file_json = output_dir / f"coco_bboxes.json"
    with open(coco_file_json, "w") as f:
        json.dump(coco_bbox_list, f)

    # Minimal COCO annotation file so mmpose's test harness can index the
    # frames; image_id == 0-based frame index, matching coco_bbox_list.
    dummy_coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }

    # image_id == frame index (0..N-1), matching your coco_bbox_list entries.
    for idx, img_path in enumerate(sorted_images):
        dummy_coco["images"].append({
            "id": idx,
            "file_name": img_path.name,
            "width": int(img_w),
            "height": int(img_h),
        })

    dummy_ann_path = output_dir / "dummy_coco.json"
    with open(dummy_ann_path, "w") as f:
        json.dump(dummy_coco, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--cam", type=str, required=True)
    parser.add_argument("--detector", type=str, default="regnety", choices=["vitdet", "regnety", "yolo11"], help="Type of body detector to use")
    parser.add_argument("--seqname", type=str, required=True)
    parser.add_argument("--use_half_sized_img", action='store_true', help="Whether to use half-sized images for detection to speed up")
    args = parser.parse_args()

    main(args)