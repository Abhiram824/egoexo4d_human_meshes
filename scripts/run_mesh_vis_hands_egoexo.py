"""
Debug/QA visualization: renders the optimizer's per-frame SMPL mesh into each
exo camera's view (one output video per camera) by calling the SAME production
code path that produces the pipeline's final_vis src_cam videos:
prep_result_vis -> animate_scene -> vis.viewer.OffscreenAnimation. The only
thing swapped per view is scene_dict["cameras"]["src_cam"] (that view's c2w)
and the viewer/background images, so the rendering itself (mesh building,
lighting, colors, compositing, encoding) is byte-for-byte the production one.
"""
import torch
import numpy as np
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from macros import DATASET_BASE_PATH
from util.egoexo4d_utils import get_take_cameras
import imageio
from pathlib import Path

from util.loaders import load_smpl_body_model

import subprocess
import shutil
import tempfile

from vis.output import prep_result_vis, animate_scene
from vis.viewer import init_viewer


SMPL_BODY_MODEL = Path(__file__).resolve().parent.parent / "_DATA" / "body_models" / "smplh" / "male" / "model.npz"

# body params are (1, T, ...); camera params are (num_views, T, ...) with
# view 0 = reference (cam01 copy), views 1-4 = exo cams
_BODY_KEYS = ('trans', 'root_orient', 'pose_body', 'hand_pose')
_CAM_KEYS = ('cam_R', 'cam_t', 'intrins', 'cam_dist')

# the production pipeline's track id for the triangulated sequence: run_vis
# passes obs_data["track_id"], which comes from the track_preds dir name --
# "001" for points_triangulated -> id 1 (vis/colors.txt row 1, teal)
TRACK_ID = 1

FPS = 30


def get_image_dir(npz_path: Path) -> Path:
    """
    Extracts + undistorts cam01-04 frames for the npz's take into a temp dir,
    so this script can run off just the npz (no dependency on a full pipeline
    output dir with images/ already populated). Skips cameras that were
    already extracted by a previous call.
    """
    take_name = npz_path.parent.name
    # make a temp dir to extract images to
    temp_dir = Path("/tmp/slahmr_multiview_images/") / take_name
    video_dir = Path(DATASET_BASE_PATH) / "takes" / take_name / "frame_aligned_videos"

    # quality-filtered cam uids in gopro_calibs.csv order; first 4 map
    # positionally to cam01..cam04, matching extract_and_undistort_frames()
    # in scripts/run_pipeline.py
    cameras = get_take_cameras(take_name)
    repo_root = Path(__file__).resolve().parent.parent

    for i, uid in enumerate(cameras[:4], start=1):
        img_dir = temp_dir / f"cam{i:02d}"
        if img_dir.exists() and len(list(img_dir.glob("*.jpg"))) > 0:
            print(f"[{take_name}] cam{i:02d}: reusing frames already extracted "
                  f"in {img_dir} (delete it to re-extract)", flush=True)
            continue
        img_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{take_name}] cam{i:02d} ({i}/4): extracting + undistorting "
              f"frames -> {img_dir}", flush=True)
        subprocess.run(
            [
                "python", str(repo_root / "scripts" / "undistort_egoexo.py"),
                f"--video_path={video_dir / (uid + '.mp4')}",
                f"--video_path_to_save={img_dir}",
            ],
            check=True,
            cwd=repo_root,
        )

    return temp_dir


def concat_videos(chunk_paths, out_path):
    """Concatenate chunk videos with the ffmpeg concat demuxer (stream copy)."""
    if len(chunk_paths) == 1:
        shutil.copy(str(chunk_paths[0]), str(out_path))
        return
    import imageio_ffmpeg
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in chunk_paths:
            f.write(f"file '{p}'\n")
        list_path = f.name
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "concat", "-safe", "0",
         "-i", list_path, "-c", "copy", str(out_path)],
        check=True, capture_output=True, text=True,
    )
    os.remove(list_path)


def main(args):
    npz_path = Path(args.npz_path)
    output_dir = Path(args.output_dir) if args.output_dir else npz_path.parent / "final_vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    img_dir = get_image_dir(npz_path)
    raw = np.load(npz_path)
    T = raw['trans'].shape[1]
    if args.num_render > 0:
        T = min(T, args.num_render)

    # merged npz's from merge_chunk_npz.py flag NaN-loss chunks as invalid
    # (valid[0,i]==0); raw per-chunk optimizer npz's don't have this key.
    valid = raw['valid'][0] if 'valid' in raw.files else np.ones(T, dtype=bool)

    # SMPL betas are fit per smooth_fit chunk, so shape legitimately changes
    # at chunk boundaries; render each chunk separately with its own betas
    # (mirrors the production per-chunk run_vis + stitch_chunk_videos flow).
    chunk_ranges = raw['chunk_ranges'] if 'chunk_ranges' in raw.files else np.array([[0, T]])
    chunk_ranges = [(int(s), min(int(e), T)) for s, e in chunk_ranges if int(s) < T]

    H, W = imageio.imread(img_dir / "cam01" / "000001.jpg").shape[:2]
    img_size = (W, H)

    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    body_models = {}  # chunk_T -> body model
    chunks_dir = Path(tempfile.mkdtemp(prefix="mesh_vis_chunks_"))
    chunk_videos = {view: [] for view in range(1, 5)}

    n_chunks = len(chunk_ranges)
    print(f"Rendering {T} frames into 4 camera views "
          f"({n_chunks} chunk(s) x 4 views)", flush=True)

    for chunk_i, (start, end) in enumerate(chunk_ranges, start=1):
        chunk_T = end - start
        chunk_valid = valid[start:end].astype(bool)
        bg_paths = {
            view: [img_dir / f"cam{view:02d}" / f"{i+1:06d}.jpg" for i in range(start, end)]
            for view in range(1, 5)
        }

        if not chunk_valid.any():
            # whole chunk invalid: nothing to fit a scene to -> background-only
            # frames, written with the same encoder settings vis.animate uses
            print(f"chunk {chunk_i}/{n_chunks} (frames {start}-{end}): "
                  f"no valid optimizer output, writing background only", flush=True)
            for view in range(1, 5):
                out_path = chunks_dir / f"view{view}_chunk{start:06d}_src_cam.mp4"
                frames = [imageio.imread(p) for p in bg_paths[view]]
                imageio.mimwrite(out_path, frames, fps=FPS)
                chunk_videos[view].append(out_path)
            continue

        if chunk_T not in body_models:
            body_models[chunk_T] = load_smpl_body_model(str(SMPL_BODY_MODEL), chunk_T)[0]

        res = {k: torch.from_numpy(raw[k][:, start:end]) for k in _BODY_KEYS if k in raw.files}
        # camera slot: prep_result_vis takes view [[0]]; the actual per-view
        # camera is swapped into scene_dict["cameras"] below
        for k in _CAM_KEYS:
            res[k] = torch.from_numpy(raw[k][[0], start:end])
        if 'betas_per_frame' in raw.files:
            # constant within a chunk; take the chunk's first frame
            res['betas'] = torch.from_numpy(raw['betas_per_frame'][:, start])
        elif 'betas' in raw.files:
            res['betas'] = torch.from_numpy(raw['betas'])

        # -1 marks a track as not-in-frame, which makes the production path
        # render background-only for that frame
        vis_mask = torch.ones(1, chunk_T)
        vis_mask[0, ~chunk_valid] = -1.0
        track_ids = torch.full((1,), TRACK_ID, dtype=torch.long)

        with tempfile.TemporaryDirectory() as obj_tmp:
            scene_dict = prep_result_vis(
                res, vis_mask, track_ids, body_models[chunk_T], save_dir=obj_tmp
            )
            # keep the per-frame .obj export, renumbered to absolute frames
            for local_i in range(chunk_T):
                obj_path = Path(obj_tmp) / "meshes" / f"{local_i:06d}.obj"
                if obj_path.exists():
                    shutil.move(str(obj_path), str(mesh_dir / f"{start + local_i:06d}.obj"))

        for view in range(1, 5):
            # w2c -> c2w for this view over the chunk
            w2c = np.zeros((chunk_T, 4, 4), dtype=np.float64)
            w2c[:, :3, :3] = raw['cam_R'][view, start:end]
            w2c[:, :3, 3] = raw['cam_t'][view, start:end]
            w2c[:, 3, 3] = 1.0
            c2w = torch.from_numpy(np.linalg.inv(w2c)).float()
            scene_dict["cameras"] = {"src_cam": c2w}

            vis = init_viewer(
                img_size,
                raw['intrins'][view, start:end],
                vis_scale=1.0,
                bg_paths=[str(p) for p in bg_paths[view]],
                fps=FPS,
            )
            out_name = chunks_dir / f"view{view}_chunk{start:06d}"
            # labels the per-frame render bar inside vis.viewer.render_frames
            vis.render_desc = (f"chunk {chunk_i}/{n_chunks} cam{view:02d} "
                               f"(view {view}/4)")
            # seq_name matches production run_vis (dataset.seq_name)
            animate_scene(
                vis, scene_dict, str(out_name), seq_name="points_triangulated",
                render_views=["src_cam"],
            )
            vis.close()
            chunk_videos[view].append(Path(f"{out_name}_src_cam.mp4"))

    for view in range(1, 5):
        concat_videos(chunk_videos[view], output_dir / f"view_{view}.mp4")

    shutil.rmtree(chunks_dir, ignore_errors=True)
    print(f"Done. Wrote view_1.mp4 - view_4.mp4 to {output_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_render", type=int, default=-1, help="Number of frames to render; -1 means all frames.")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to the npz file containing the camera intrinsics and extrinsics.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the output videos. If not provided, the output videos will be saved in the same directory as the mesh_dir.")
    args = parser.parse_args()

    main(args)
