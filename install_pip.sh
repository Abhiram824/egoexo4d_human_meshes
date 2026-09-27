#!/usr/bin/env bash
set -e

# ffmpeg (frame extraction) and vrs (Aria calibration extraction) are not
# pip-installable -- this venv approach assumes both are already on PATH
# (e.g. `apt install ffmpeg` and building/installing vrs from
# https://github.com/facebookresearch/vrs). If using conda instead, see
# install_conda.sh, which installs both via conda-forge.

echo "Creating virtual environment"
python3.10 -m venv .slahmr_hands
echo "Activating virtual environment"

source $PWD/.slahmr_hands/bin/activate

# install pytorch
$PWD/.slahmr_hands/bin/pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# torch-scatter
$PWD/.slahmr_hands/bin/pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# mmcv==1.3.9 (pulled in below by both HaMeR and requirements.txt) needs pkg_resources,
# which very recent setuptools no longer ships; but our own package and HaMeR's both need
# editable installs (PEP 660's build_editable hook, added in setuptools 64.0.0). 68.2.2 is
# the last easy-to-pin version with both pkg_resources and PEP 660 support -- installing it
# now, ahead of everything below, so it's active for every subsequent build in this script
$PWD/.slahmr_hands/bin/pip install setuptools==68.2.2

# install PHALP
# PHALP depends on detectron2, whose setup.py imports torch to determine which CUDA
# extensions to build; pip's default build isolation hides the torch we just installed
# above, so the build fails with "ModuleNotFoundError: No module named 'torch'" without
# --no-build-isolation
#
# --no-build-isolation only applies to phalp itself, not to detectron2 (a
# transitive git dependency pulled in via phalp's own setup.py) -- pip does not
# propagate the flag down the dependency tree, so detectron2 still gets built
# in an isolated env and hits the same "No module named 'torch'" error unless
# it's installed explicitly first, here, so it's already satisfied by the time
# phalp's own install runs
$PWD/.slahmr_hands/bin/pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
$PWD/.slahmr_hands/bin/pip install --no-build-isolation phalp[all]@git+https://github.com/brjathu/PHALP.git

# install HaMeR
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
# --no-build-isolation for the same reason as PHALP above: hamer's own setup.py pulls in
# detectron2 (and mmcv) as direct dependencies, which need to see the already-installed
# torch/setuptools rather than an isolated build env
$PWD/.slahmr_hands/bin/pip install --no-build-isolation -e .[all]
cd ..

# install source
$PWD/.slahmr_hands/bin/pip install -e .

# install remaining requirements
# --no-build-isolation: chumpy's setup.py does `import pip` directly (a legacy pattern),
# which fails in an isolated build env that doesn't include pip itself
$PWD/.slahmr_hands/bin/pip install --no-build-isolation -r requirements.txt

# pyrender (in requirements.txt) pulls in PyOpenGL unpinned; pin it to the
# exact version pyrender actually requires (headless rendering uses EGL, so
# this version's older osmesa support doesn't matter here)
$PWD/.slahmr_hands/bin/pip install pyopengl==3.1.0

# install ViTPose
$PWD/.slahmr_hands/bin/pip install -v -e third-party/ViTPose

$PWD/.slahmr_hands/bin/pip install projectaria-tools==1.6.0
$PWD/.slahmr_hands/bin/pip install numpy==1.26.4
