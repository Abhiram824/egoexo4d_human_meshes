# Ego-Exo4D Human Meshes Dataset
### 4D Human Motion Reconstruction for Ego-Exo Captures

Abhiram Maddukuri, Georgios Pavlakos

[**Paper**](https://arxiv.org/abs/2609.30187) &nbsp;·&nbsp; [**Project page**](https://abhiram824.github.io/egoexo4d_human_meshes/) &nbsp;·&nbsp; [**Dataset**](#using-the-pre-computed-dataset-huggingface)

## Getting started
This code was tested on Ubuntu 22.04 LTS and requires a CUDA-capable GPU (with `nvcc`/a devel CUDA install, since several dependencies compile CUDA extensions from source). The steps below were verified against a genuine from-scratch install (fresh clone, no pre-existing caches) on real hardware — see "Why the script looks the way it does" below for the reasoning behind each fix.

0. Install the Ego-Exo4D dataset and point the codebase at it.

    Download the [Ego-Exo4D dataset](https://ego-exo4d-data.org/) (see their [downloader docs](https://docs.ego-exo4d-data.org/) — you'll need `takes.json`, `captures.json`, the `takes/` directory with `trajectory/gopro_calibs.csv` per take, and `annotations/ego_pose/` if you want to run evaluation). Then set the dataset root as an environment variable:
    ```bash
    export EGOEXO4D_DATASET=/path/to/EgoExo4d_dataset
    ```
    `slahmr/macros.py` reads this at import time and will raise immediately (`EGOEXO4D_DATASET is not set...`) if it's missing, so every shell you run any pipeline script from needs this exported first.

1. Clone repository and submodules
    ```
    git clone --recursive https://github.com/Abhiram824/egoexo4d_human_meshes.git
    cd egoexo4d_human_meshes
    git submodule sync --recursive && git submodule update --init --recursive
    ```
    `submodule sync` ensures submodule URLs/commits (e.g. `third-party/ViTPose`, pinned to a fork with an EGL rendering fix) match what's recorded in this repo.

2. Install system prerequisites: `unzip` (needed by `download_models.sh`, which doesn't check whether extraction succeeded) and a C/C++ toolchain + `nvcc` (needed to build detectron2, neural-renderer-pytorch, and lietorch from source).

3. Set up conda environment. Run
    ```
    bash install_conda.sh
    ```

   Alternatively, you can also create a virtualenv environment:
    ```
    bash install_pip.sh
    ```
    If your conda is new enough to require accepting Terms of Service before `conda create` will run:
    ```bash
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    ```

    <details>
        <summary>Why the script looks the way it does (if you're diffing against an older version or troubleshooting)</summary>

    | Symptom | Cause | Fix |
    |---|---|---|
    | `ModuleNotFoundError: No module named 'torch'` building detectron2 (via PHALP or HaMeR) | pip's build isolation hides the already-installed torch from detectron2's `setup.py` | `--no-build-isolation` on the detectron2, PHALP, and HaMeR install lines |
    | Same error for **detectron2** specifically, even with `--no-build-isolation` on the PHALP line | Newer pip does not propagate a command-line `--no-build-isolation` flag down to *transitively* resolved git dependencies (detectron2 is only discovered via PHALP's own `setup.py`) | Install detectron2 standalone, explicitly, immediately before the PHALP line |
    | `ModuleNotFoundError: No module named 'pip'` building chumpy | chumpy's `setup.py` does `import pip` directly, which fails in an isolated build env without pip | `--no-build-isolation` on `pip install -r requirements.txt` |
    | `ModuleNotFoundError: No module named 'pkg_resources'` building mmcv, or `missing the 'build_editable' hook` installing HaMeR | very recent setuptools dropped `pkg_resources`, but HaMeR's/our own editable installs need setuptools ≥64 (PEP 660) | pin `setuptools==68.2.2` — the last version with both |
    | `numpy.dtype size changed, may indicate binary incompatibility` importing mmpose/xtcocotools | unpinned numpy resolves to 2.x, breaking xtcocotools's compiled extension (built for numpy 1.x ABI) | `pip install numpy==1.26.4` as the last step, *after* `projectaria-tools` (which itself pulls numpy>=2 if installed after) |
    | `json.decoder.JSONDecodeError: Unterminated string` parsing Aria calibration (cooking/bike takes only) | newer `vrs` builds truncate long tag values in their plain-text summary dump, corrupting the calib JSON the code parses out of it | pin `vrs=1.3.0` |
    | `pip's dependency resolver ... opencv-python requires numpy>=2 ... incompatible` | expected side effect of the numpy 1.26.4 re-pin above | harmless warning, not a failure — this pipeline doesn't break at runtime with numpy 1.x |

    </details>

    Finally, do a small hack to make hamer consistent with slahmr:
    ```
    sed -i '5s/".\/_DATA"/os.path.abspath(f"{__file__}\/..\/..\/..\/..\/_DATA")/' hamer/hamer/configs/__init__.py
    ```

    Then activate the environment in your shell — the install script activates it only inside its own subshell, and every step below (including `gdown` in steps 4–5) needs it:
    ```bash
    conda activate slahmr_hands
    # or, if you used install_pip.sh:
    source .slahmr_hands/bin/activate
    ```

4. Download models from [here](https://drive.google.com/file/d/1GXAd-45GzGYNENKgQxFQ4PHrBp8wDRlW/view?usp=sharing). Run
    ```
    ./download_models.sh
    ```
    or
    ```
    gdown https://drive.google.com/uc?id=1GXAd-45GzGYNENKgQxFQ4PHrBp8wDRlW
    unzip -q slahmr_dependencies.zip
    rm slahmr_dependencies.zip
    ```

    All models and checkpoints should have been unpacked in `_DATA` (`body_models/{smpl,smplh}`, `humor_ckpts`, `vitpose_ckpts`, `droid.pth`).

5. Fetch HaMeR's own checkpoints (a separate ~6GB download, no auth wall), and merge them into the repo-root `_DATA/` (`fetch_demo_data.sh` extracts relative to wherever you run it — `hamer/_DATA/` — but the `sed` patch in step 3 points HaMeR's own code at the repo root's `_DATA/`):
    ```bash
    cd hamer
    bash fetch_demo_data.sh
    cd ..
    cp -rn hamer/_DATA/hamer_ckpts _DATA/
    cp -rn hamer/_DATA/data _DATA/
    cp -rn hamer/_DATA/vitpose_ckpts/* _DATA/vitpose_ckpts/
    rm -rf hamer/_DATA hamer/hamer_demo_data.tar.gz
    ```

6. Two license-gated model files aren't included in any of the automated downloads above and must be obtained manually — both are one-time account registrations, not automatable:

    * **MANO** — register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de/), download the MANO model, and place **only** `MANO_RIGHT.pkl` at `_DATA/data/mano/MANO_RIGHT.pkl` (the directory exists but is empty after step 5). `MANO_LEFT.pkl` is not needed — HaMeR only ever instantiates a right-hand model (`_DATA/hamer_ckpts/model_config.yaml` never sets `IS_RHAND`, so `smplx.MANOLayer` defaults to `is_rhand=True`); left-hand crops are flipped into right-hand space before inference and flipped back after.
    * **SMPL-X** — register at [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/) (a separate account from MANO's), download the SMPL-X neutral model, and place it at `_DATA/body_models/smplx/SMPLX_NEUTRAL.npz` (this directory doesn't exist yet — create it first with `mkdir -p _DATA/body_models/smplx`). This is needed even though the pipeline's body model is SMPL+H, not SMPL-X: `slahmr/body_model/body_model.py`'s `BodyModel.__init__` borrows SMPL-X's hand PCA basis (`hands_componentsl/r`, `hands_meanl/r`) to fill in what SMPL+H's own model doesn't include. Without this file, everything through triangulation (stage 3) still works — it only fails once you reach the SLAHMR optimizer (stage 4), so it's easy to miss until then.

### Verify the install with a test run

Once all of the above is in place, confirm everything actually works end-to-end
by running one short take through the full pipeline:

```bash
export CUDA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0
export PYOPENGL_PLATFORM=egl
unset DISPLAY WAYLAND_DISPLAY

python scripts/run_pipeline.py --video cmu_bike02_4 --device_num 0 --log-time
```

`cmu_bike02_4` is a good choice for this: it's short (110 frames, ~13-14
minutes end-to-end on a single GPU) but still exercises the Aria/egocentric
path (bike takes use Aria, same as cooking takes), so a clean run through it
touches every stage — undistortion, camera params, per-camera detection/pose,
HaMeR+MANO hand mesh, multi-view triangulation, and SLAHMR `smooth_fit`
optimization with mesh rendering.

A successful run ends with:
```
Stitched 5 video(s) into outputs/cmu_bike02_4/final_vis
Pipeline completed successfully.
Timing log saved to: outputs/cmu_bike02_4/time_log.json
```
The presence of `outputs/cmu_bike02_4/time_log.json` is the project's success
sentinel — its absence means the run didn't complete. You should also have 5
rendered videos in `outputs/cmu_bike02_4/final_vis/*.mp4`
(`points_triangulated_input.mp4`, the `..._smooth_fit_final_000060_{above,side,src_cam}.mp4`
views, and `..._smooth_fit_grid.mp4`) with real, non-trivial file sizes — if
any of these fail to render (in particular, `pyrender`/EGL errors), see the
GPU environment variables above; a container without proper GPU/EGL passthrough
(e.g. rootless Docker substitutes like `udocker`) can complete every other
stage but fail specifically at mesh rendering.

## Using the pre-computed dataset (HuggingFace)

Pre-computed SLAHMR optimization results for the full Ego-Exo4D take set are
published at
[Ego-Exo4D-HM/npz-datasets](https://huggingface.co/datasets/Ego-Exo4D-HM/npz-datasets/tree/main)
— one merged `.npz` per take (`<take_name>/points_triangulated_world_results_merged.npz`,
the optimizer's per-chunk results concatenated over the full take),
2649 takes, ~48GB total. This lets you render or evaluate results without
running the pipeline yourself — you still need `EGOEXO4D_DATASET` set (step 0
above), since rendering re-extracts the raw take's video frames.

Install the `huggingface_hub` CLI if you don't already have it (it's not part
of `install_conda.sh`/`install_pip.sh`, since it's only needed for this
workflow, not the pipeline itself):
```bash
pip install -U "huggingface_hub[cli]"
hf auth login   # only needed if the dataset repo is private
```

### Download one take and render it

```bash
hf download Ego-Exo4D-HM/npz-datasets --repo-type dataset \
    --include "cmu_bike02_4/*" --local-dir data_filtered

python scripts/run_mesh_vis_hands_egoexo.py \
    --npz_path data_filtered/cmu_bike02_4/points_triangulated_world_results_merged.npz
```
`run_mesh_vis_hands_egoexo.py` only needs the merged npz — it generates the
SMPL meshes from it directly (`generate_meshes()`) and extracts+undistorts the
take's raw frames itself (`get_image_dir()`), so there's no dependency on a
full pipeline output directory. It writes one rendered video per exo camera
(`view_1.mp4`–`view_4.mp4`) next to the npz, in a `final_vis/` subdirectory by
default (override with `--output_dir`); `--num_render N` caps the frame count
for a quick smoke test on a long take.

**Pick a short take for a first try** — take length varies a lot (some are
3000+ frames / multiple GPU-minutes to render, since the script reconstructs
its neural mesh renderer per frame per camera view). `cmu_bike02_4` (110
frames, ~2.5 minutes to render all 4 views) is a good default; if you want a
different one, check its length first:
```bash
python3 -c "
import numpy as np
d = np.load('data_filtered/<take_name>/points_triangulated_world_results_merged.npz')
print(d['trans'].shape[1], 'frames')
"
```

### Download the entire dataset

```bash
hf download Ego-Exo4D-HM/npz-datasets --repo-type dataset --local-dir data_filtered
```
This pulls all 2649 takes (~48GB). If it stalls partway with no visible
progress (check with `du -sh data_filtered`, since `hf download`'s progress
bar doesn't always flush cleanly to a redirected/logged output) — this is a
real, reproducible failure mode we hit during testing, not a hypothetical one:
- `hf download` (default `--max-workers 8`) can wedge with all its HTTP
  connections stuck in `CLOSE-WAIT` (server closed the connection, client
  never noticed) — retry with `--max-workers 4` or lower.
- Repeated retries in a tight loop can trip HuggingFace's API rate limit
  (1000 requests/5min — `429 Too Many Requests`) rather than a real network
  problem, since every invocation re-lists all 2649 files' metadata even to
  just resume. If you hit this, wait ~5 minutes before retrying rather than
  looping immediately.
- The command is safe to just re-run — already-downloaded files are skipped,
  so an interrupted or stalled download picks up where it left off.

Verify a complete download by comparing file counts:
```bash
python3 -c "
from huggingface_hub import HfApi
import os
remote = {f for f in HfApi().list_repo_files('Ego-Exo4D-HM/npz-datasets', repo_type='dataset') if f.endswith('.npz')}
local = {os.path.relpath(os.path.join(r, f), 'data_filtered')
         for r, _, fs in os.walk('data_filtered') for f in fs
         if f.endswith('.npz') and '.cache' not in r}
print(f'{len(local)}/{len(remote)} present, {len(remote - local)} missing')
"
```

## BibTeX

If you use our code or dataset in your research, please cite:
```
@misc{maddukuri2026egoexo4dhumanmeshesdataset,
      title={Ego-Exo4D Human Meshes Dataset: 4D Human Motion Reconstruction for Ego-Exo Captures},
      author={Abhiram Maddukuri and Georgios Pavlakos},
      year={2026},
      eprint={2609.30187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.30187},
}
```
