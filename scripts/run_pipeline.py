#!/usr/bin/env python3
"""
Main pipeline entrypoint: runs stages 0-4 (frame extraction/undistortion, cameras,
detection+pose, triangulation, SLAHMR smooth_fit) for one take or a JSONL jobs file.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import os
from filelock import FileLock, Timeout
import time
from preproc.vitpose_model import ViTPoseModel
import shutil
import json
import traceback
from macros import BASE_OUTPUT_PATH, DATASET_BASE_PATH   # your project constant
from util.egoexo4d_utils import get_take_cameras, get_ego_cam_name
from util.job_utils import _claim_one_job_with_lock, _finalize_job_with_lock

# env passed to every GPU subprocess; __main__ pins CUDA/EGL devices into it
ENV_GPU = os.environ.copy()
VIDEO_DIR = os.path.join(DATASET_BASE_PATH, "takes")


def run_batched_vitpose_inference(model, root_dir, cam, kp_dir, batch_size=32):
    """
    Runs a ViTPose model (body or wholebody, chosen by `model`) via mmpose's
    test.py over the bboxes in kp_dir, writing vitpose[_wholebody]_out.json.
    """
    script_path = Path(__file__).parent.parent / "third-party" / "ViTPose" / "tools" / "test.py"
    model_dict = ViTPoseModel.MODEL_DICT[model]
    config = model_dict["config"]
    checkpoint = model_dict["model"]
    fname = "vitpose_wholebody_out.json" if "wholebody" in checkpoint else "vitpose_out.json"
    outpath = kp_dir / fname
    img_prefix = root_dir / "images" / cam
    # both files are produced by run_bbox_selection.py; dummy_coco.json only
    # exists to satisfy the mmpose test harness
    bbox_file = kp_dir / "coco_bboxes.json"
    ann_file = kp_dir / "dummy_coco.json"

    subprocess.run([
        "python", str(script_path),
        str(config),
        str(checkpoint),
        "--out", str(outpath),
        "--cfg-options", "data.test.img_prefix=" + str(img_prefix),
        "data.test.ann_file=" + str(ann_file),
        "data.test.data_cfg.bbox_file=" + str(bbox_file),
        "data.test_dataloader.samples_per_gpu=" + str(batch_size),
    ], check=True, env=ENV_GPU)



def stitch_chunk_videos(chunk_dirs: list, output_dir: Path) -> None:
    """Concatenate per-chunk mp4s in output_dir using ffmpeg concat demuxer."""
    if not chunk_dirs:
        return

    # Collect all unique video filenames across chunks
    video_names: set = set()
    for chunk_dir in chunk_dirs:
        for f in chunk_dir.glob("*.mp4"):
            video_names.add(f.name)

    for video_name in sorted(video_names):
        available = [d / video_name for d in chunk_dirs if (d / video_name).exists()]
        if not available:
            continue
        out_path = output_dir / video_name
        if len(available) == 1:
            shutil.copy(str(available[0]), str(out_path))
            continue

        concat_file = output_dir / f"_concat_{video_name}.txt"
        with open(concat_file, "w") as f:
            for vp in available:
                f.write(f"file '{vp}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(out_path),
        ], check=True)
        concat_file.unlink()
    print(f"Stitched {len(video_names)} video(s) into {output_dir}")



def detect_resume_stage(root_dir: Path, video_name: str) -> int:
    """Auto-detect resume stage from sentinel files left by each completed stage."""
    # each sentinel is written by stage N itself, and finding it resumes AT
    # stage N — the last completed stage is deliberately redone in case a
    # previous run died partway through it
    seq_name_auto = "points_triangulated"
    stage3_sentinel = root_dir / "slahmr" / "shot_idcs" / f"{seq_name_auto}.json"
    stage2_sentinel = root_dir / "slahmr" / "track_preds" / seq_name_auto / "002" / "vitpose_out.json"
    stage1_sentinel = root_dir / "slahmr" / "cameras" / seq_name_auto / "shot-0" / "cameras.npz"
    cam01_dir = root_dir / "images" / "cam01"
    stage0_done = (
        cam01_dir.exists()
        and len(list(cam01_dir.glob("*.jpg"))) > 0
        and all((root_dir / "images" / f"cam{i:02d}").exists()
                and len(list((root_dir / "images" / f"cam{i:02d}").glob("*.jpg"))) > 0
                for i in range(1, 5))
    )
    if stage3_sentinel.exists():
        print(f"Auto-resuming {video_name} at stage 3 (stages 0-2 outputs found)")
        return 3
    elif stage2_sentinel.exists():
        print(f"Auto-resuming {video_name} at stage 2 (stages 0-1 outputs found)")
        return 2
    elif stage1_sentinel.exists():
        print(f"Auto-resuming {video_name} at stage 1 (stage 0 outputs found)")
        return 1
    elif stage0_done:
        print(f"Auto-resuming {video_name} at stage 0 (no prior outputs found)")
        return 0
    return 0


def extract_and_undistort_frames(video_dir: Path, root_dir: Path, cameras: list, use_aria: bool, video_name: str) -> None:
    """Stage 0: extract frames via ffmpeg and undistort them (GoPro fisheye + optional Aria pinhole)."""
    # cameras is the quality-filtered list from get_take_cameras(); the first 4
    # are renamed positionally to cam01..cam04, whatever their real names.
    # read_cameras.py applies the same quality filter so view i's extrinsics
    # stay paired with these images.
    for i, uid in enumerate(cameras[:4], start=1):
        img_dir = root_dir / "images" / f"cam{i:02d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "python", "scripts/undistort_egoexo.py",
            f"--video_path={video_dir / (uid + '.mp4')}",
            f"--video_path_to_save={img_dir}",
        ], check=True)

    if use_aria:
        aria_name = get_ego_cam_name(video_name)
        aria_img_dir = root_dir / "images" / "aria"
        # TODO hardcoded aria suffix
        aria_video_file = video_dir / f"{aria_name}_214-1.mp4"
        aria_img_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "python", "scripts/undistort_egocam.py",
            f"--video_path={aria_video_file}",
            f"--video_path_to_save={aria_img_dir}",
        ], check=True)


def compute_camera_parameters(video_len: int, camera_save_path: Path, video_name: str) -> None:
    """Stage 1: build cameras.npz from GoPro calibs + Aria MPS (scripts/read_cameras.py)."""
    cmd_args = [
        "python", "scripts/read_cameras.py",
        f"--video_len={video_len}",
        f"--save_path={camera_save_path}",
        f"--take-name={video_name}",
    ]
    subprocess.run(cmd_args, check=True)


def detect_and_estimate_pose(root_dir: Path, seq_name: str, num_views: int, video_name: str) -> None:
    """Stage 2: per-camera bbox detection, ViTPose body+wholebody inference, pose merge, HaMeR hand refinement."""
    for i in range(1, num_views+1):
        camera = f"cam{i:02d}" if i <=4 else "aria"
        subprocess.run([
            "python", "scripts/run_bbox_selection.py",
            f"--cam={camera}",
            f"--root-dir={root_dir}",
            f"--seqname={seq_name}",
        ], check=True, env=ENV_GPU)
        # track_preds dir numbering is camid+1: cam01 -> 002, ..., aria -> 006
        bbox_path = root_dir / "slahmr" / "track_preds" / seq_name / f"{i+1:03d}" / "coco_bboxes.json"
        with open(bbox_path, "r") as f:
            bboxes = json.load(f)
        # a camera that never saw the subject would poison triangulation, so
        # fail the whole take early
        if len(bboxes) == 0:
            raise ValueError(f"No bounding boxes found for camera {camera} in take {video_name}.")
        run_batched_vitpose_inference('ViTPose-G (multi-task train, COCO)', root_dir, camera, root_dir / "slahmr" / "track_preds" / seq_name / f"{i+1:03d}", batch_size=32)
        run_batched_vitpose_inference('ViTPose+-G (multi-task train, COCO)', root_dir, camera, root_dir / "slahmr" / "track_preds" / seq_name / f"{i+1:03d}", batch_size=32)
        subprocess.run([
            "python", "scripts/merge_pose_preds.py",
            f"--cam={camera}",
            f"--root-dir={root_dir}",
            f"--seqname={seq_name}",
        ], check=True, env=ENV_GPU)
        subprocess.run([
            "python", "scripts/run_batched_hamer.py",
            f"--cam={camera}",
            f"--root-dir={root_dir}",
            f"--seqname={seq_name}",
        ], check=True, env=ENV_GPU)


def triangulate_keypoints(root_dir: Path, seq_name: str, video_len: int) -> None:
    """Stage 3: multi-view RANSAC triangulation of the 67 keypoints (scripts/triangulate_points.py)."""
    subprocess.run([
        "python", "scripts/triangulate_points.py",
        f"--seqname={seq_name}",
        f"--video_len={video_len}",
        f"--root_path={root_dir}",
    ], check=True)


def render_triangulation_debug_vis(root_dir: Path, start_idx: int, num_render: int) -> None:
    """--debug only: render triangulated + single-view 2D keypoint overlay videos. Not part of stage-3 timing."""
    subprocess.run([
        "python", "slahmr/run_keyp_vis_hands_egoexo.py",
        f"--main_frame={start_idx}",
        f"--num_render={num_render}",
        f"--root_path={root_dir}",
    ], check=True, env=ENV_GPU)
    subprocess.run([
        "python", "slahmr/run_keyp_vis_hands_egoexo.py",
        f"--main_frame={start_idx}",
        f"--num_render={num_render}",
        f"--root_path={root_dir}",
        "--single_view_kps_vis"
    ], check=True, env=ENV_GPU)


def run_smooth_fit_optimization(root_dir: Path, seq_name: str, start_idx: int, end_idx: int, chunk_size: int, final_vis: Path) -> None:
    """Stage 4: SLAHMR smooth_fit optimization in chunks of chunk_size frames, then stitch per-chunk videos."""
    chunk_dirs = []
    chunk_start = start_idx
    while chunk_start < end_idx:
        chunk_end = min(chunk_start + chunk_size, end_idx)
        chunk_dir = final_vis / f"chunk_{chunk_start:06d}_{chunk_end:06d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_dirs.append(chunk_dir)
        print(f"Running smooth_fit chunk [{chunk_start}, {chunk_end})")
        multiview_args = [
            "python", "slahmr/run_opt.py",
            "data=video",
            f"data.seq={seq_name}",
            f"data.root={root_dir}",
            "run_opt=True",
            "run_vis=True",
            f"+vis.save_dir={chunk_dir}",
            f"log_root={root_dir / 'logs'}",
            f"data.start_idx={chunk_start}",
            f"data.end_idx={chunk_end}",
            "~optim.motion_chunks",
            "vis.phases=[smooth_fit]",
        ]
        subprocess.run(multiview_args, check=True, env=ENV_GPU)
        chunk_start = chunk_end

    # Stitch per-chunk videos into final_vis
    stitch_chunk_videos(chunk_dirs, final_vis)


def run_pipeline(video_dir: Path, args) -> None:
    """
    Runs stages 0-4 for one take, timing each stage and skipping any before
    args.resume_stage (auto-detected from sentinels when left at 0).
    """
    start_idx = args.start_idx
    end_idx = args.end_idx
    # video_dir is <take>/frame_aligned_videos, so the take name is its parent
    video_name = video_dir.parent.stem
    root_dir = Path(BASE_OUTPUT_PATH) / f"{video_name}"
    resume_stage = args.resume_stage
    os.makedirs(root_dir, exist_ok=True)
    time_log_path = root_dir / "time_log.json"
    time_log = {}

    cameras = get_take_cameras(video_name)
    if len(cameras) < 4:
        raise ValueError(f"Take {video_name} does not have all 4 required cameras: {cameras}")

    # the Aria ego camera is only used for cooking and bike takes
    is_cooking = "cooking" in video_name
    is_bike = "bike" in video_name
    use_aria = is_cooking or is_bike

    if resume_stage == 0:
        resume_stage = detect_resume_stage(root_dir, video_name)

    video = args.video

    seq_name = "points_triangulated"

    # Track overall pipeline start time
    pipeline_start = time.time()

    if resume_stage <= 0:
        stage_0_start = time.time()
        extract_and_undistort_frames(video_dir, root_dir, cameras, use_aria, video_name)
        stage_0_end = time.time()
        time_log["stage_0"] = stage_0_end - stage_0_start
    else:
        time_log["stage_0"] = "skipped"

    video_len = len(list((root_dir / "images").glob("cam01/*.jpg")))

    assert end_idx <= video_len, "End index exceeds video length."
    assert start_idx <= end_idx, "Start index must be less than or equal to end index."

    print(f"Video length: {video_len} frames")
    final_vis = root_dir / "final_vis"
    final_vis.mkdir(parents=True, exist_ok=True)

    camera_save_path = root_dir / "slahmr" / "cameras"/ seq_name / "shot-0" / "cameras.npz"
    camera_save_path.parent.mkdir(parents=True, exist_ok=True)
    if start_idx == -1:
        start_idx = 0
    if end_idx == -1:
        # end_idx = math.floor(video_len / 100) * 100
        end_idx = video_len - 1

    assert start_idx < end_idx, "Start index must be less than end index."

    num_render = end_idx - start_idx
    # Used chunk size of at most 2000 to avoid OOM during smooth_fit. 
    # Used chunk size of 100 for cooking takes for better results
    chunk_size = 100 if is_cooking else 2000

    if resume_stage <= 1:
        stage_1_start = time.time()
        compute_camera_parameters(video_len, camera_save_path, video_name)
        stage_1_end = time.time()
        time_log["stage_1"] = stage_1_end - stage_1_start
    else:
        time_log["stage_1"] = "skipped"

    num_views = 5 if use_aria else 4
    if resume_stage <= 2:
        stage_2_start = time.time()
        detect_and_estimate_pose(root_dir, seq_name, num_views, video_name)
        stage_2_end = time.time()
        time_log["stage_2"] = stage_2_end - stage_2_start
    else:
        time_log["stage_2"] = "skipped"

    if resume_stage <= 3:
        stage_3_start = time.time()
        triangulate_keypoints(root_dir, seq_name, video_len)
        stage_3_end = time.time()
        time_log["stage_3"] = stage_3_end - stage_3_start
        if args.debug:
            render_triangulation_debug_vis(root_dir, start_idx, num_render)
    else:
        time_log["stage_3"] = "skipped"

    if resume_stage <= 4:
        stage_4_start = time.time()
        run_smooth_fit_optimization(root_dir, seq_name, start_idx, end_idx, chunk_size, final_vis)
        stage_4_end = time.time()
        time_log["stage_4"] = stage_4_end - stage_4_start
    else:
        time_log["stage_4"] = "skipped"

    if not args.debug:
        img_parent_dir = root_dir / "images"
        if img_parent_dir.exists():
            # remove undistorted images to save space
            shutil.rmtree(img_parent_dir)

    # Calculate total time only for executed stages
    pipeline_end = time.time()
    time_log["total_pipeline_time"] = pipeline_end - pipeline_start
    time_log["resume_stage"] = resume_stage
    time_log["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    print("Pipeline completed successfully.")

    # time_log.json is written as the very last step, so its presence is the
    # pipeline-success sentinel used by all downstream tooling
    if args.log_time:
        try:
            with open(time_log_path, 'w') as f:
                json.dump(time_log, f, indent=2)
            print(f"Timing log saved to: {time_log_path}")
        except Exception as e:
            print(f"Warning: Failed to save timing log: {e}")

def _run_one_take(args, take_name=None):
    """
    Resolves the take's frame_aligned_videos dir and runs the full pipeline on it.
    """
    if take_name is not None:
        args.video = take_name

    video_path = os.path.join(VIDEO_DIR, args.video, "frame_aligned_videos")

    vp = Path(video_path).expanduser().resolve()
    run_pipeline(vp, args)

if __name__ == "__main__":
    # stdout is block-buffered when redirected to a file (e.g. SLURM logs), which
    # holds progress messages back until the process exits; line-buffer so stage
    # progress interleaves correctly with the subprocess output streaming through
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Run the SLAMHR Multiview pipeline")
    parser.add_argument("--video", required=False)
    parser.add_argument("--jobs_file", required=False, help="Path to JSONL jobs file with {'take','status'} per line")

    parser.add_argument("--num_render", default=0, type=int, help="Number of frames to render (default: 0)")
    parser.add_argument("--device_num", type=int, help="Device number to use", default=1)
    parser.add_argument("--start_idx", type=int, default=-1, help="Start index for processing frames")
    parser.add_argument("--end_idx", type=int, default=-1, help="End index for processing frames")
    parser.add_argument("--resume_stage", type=int, default=0, help="Resume stage of the pipeline (default: 0)")
    parser.add_argument("--debug", action="store_true", help="If set will run visualization scripts too")
    parser.add_argument("--log-time", action="store_true", help="If set will save timing info to time_log.json")
    args = parser.parse_args()
    ENV_GPU["CUDA_VISIBLE_DEVICES"] = str(args.device_num) # Set the GPU device to use
    ENV_GPU["EGL_DEVICE_ID"] = str(args.device_num)  # Set the EGL device ID
    # force PyOpenGL to use headless EGL (not GLX)
    ENV_GPU["PYOPENGL_PLATFORM"] = "egl"

    # headless: ensure no X/Wayland contexts get picked up
    ENV_GPU.pop("DISPLAY", None)
    ENV_GPU.pop("WAYLAND_DISPLAY", None)
    ENV_GPU["EGL_LOG_LEVEL"] = "debug"

    if not args.video and not args.jobs_file:
        raise SystemExit("Provide either --video or --jobs_file")

    if args.video and args.jobs_file:
        raise SystemExit("Use only one of --video or --jobs_file")

    if args.video:
        try:
            _run_one_take(args)
        except subprocess.CalledProcessError as e:
            error_details = f"Command failed with exit code {e.returncode}\nCommand: {' '.join(e.cmd)}"
            if e.stderr:
                error_details += f"Stderr: {e.stderr}"
            print(f"Error running video {args.video}: {error_details}")
        except Exception as e:
            print(f"Error running video {args.video}: {e}")
    else:
        jobs_path = Path(args.jobs_file)
        if not jobs_path.exists():
            raise SystemExit(f"--jobs_file '{jobs_path}' does not exist")

        print(f"[GPU {args.device_num}] using jobs file: {jobs_path}")
        # multiple GPU workers can share one jobs file: every claim/finalize
        # rewrite of the JSONL is serialized through this lock file
        lock = FileLock(str(jobs_path) + ".lock", timeout=60.0)


        # keep claiming not_done takes until the jobs file has none left
        while True:
            try:
                take = _claim_one_job_with_lock(jobs_path, lock)
            except Timeout:
                print("ERROR: Timeout acquiring lock to claim job")
                take = None
            
            if take is None:
                print(f"[GPU {args.device_num}] no more jobs in {jobs_path}, exiting")
                break

            print(f"[GPU {args.device_num}] running: {take}")
            ok = True
            msg = None
            log_msg = None
            log_file_pth = f"{BASE_OUTPUT_PATH}/{take}/pipeline.log"

            try:
                _run_one_take(args, take)
            except subprocess.CalledProcessError as e:
                ok = False
                error_details = f"Command failed with exit code {e.returncode}\nCommand: {' '.join(str(a) for a in e.cmd)}"
                if e.stderr:
                    error_details += f"\nStderr:\n{e.stderr}"
                if e.stdout:
                    error_details += f"\nStdout:\n{e.stdout}"
                msg = error_details
                log_msg = error_details
                print(f"[GPU {args.device_num}] FAILED {take}: {error_details}")
            except Exception as e:
                ok = False
                tb = traceback.format_exc()
                msg = tb
                log_msg = tb
                print(f"[GPU {args.device_num}] FAILED {take}: {tb}")
            if log_msg:
                os.makedirs(os.path.dirname(log_file_pth), exist_ok=True)
                with open(log_file_pth, "a") as log_file:
                    log_file.write(f"\n\n===== ERROR LOG ({time.strftime('%Y-%m-%d %H:%M:%S')}) =====\n")
                    log_file.write(log_msg)

            # finalize status
            try:
                _finalize_job_with_lock(jobs_path, lock, take, ok, msg)
            except Timeout:
                print("ERROR: Timeout acquiring lock to claim job")
                time.sleep(1.0)
                _finalize_job_with_lock(jobs_path, lock, take, ok, msg)  # Pass msg here too

            print(f"[GPU {args.device_num}] {'done' if ok else 'failed'}: {take}")
