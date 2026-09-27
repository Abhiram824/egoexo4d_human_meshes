#!/usr/bin/env bash
set -e

export CONDA_ENV_NAME=slahmr_hands

eval "$(conda shell.bash hook)"

conda create -n $CONDA_ENV_NAME python=3.10 -y

conda activate $CONDA_ENV_NAME

# install pytorch using pip, update with appropriate cuda drivers if necessary
pip install torch==2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121
# uncomment if pip installation isn't working
# conda install pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# install pytorch scatter using pip, update with appropriate cuda drivers if necessary
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
# uncomment if pip installation isn't working
# conda install pytorch-scatter -c pyg -y

# ffmpeg (frame extraction) and vrs (Aria calibration extraction) are not
# pip-installable; both are required on PATH by undistort_egoexo.py /
# undistort_egocam.py / egoexo4d_utils.py
#
# vrs is pinned to 1.3.0: slahmr/util/egoexo4d_utils.py's extract_aria_calib_to_dict()
# greps the `calib_json` tag out of `vrs <file.vrs>`'s plain-text summary dump (not a
# structured export command) and parses it as JSON. Newer vrs builds (seen with whatever
# conda-forge currently resolves as latest, 1.4.0) truncate long tag values in that summary
# with "..." for terminal-readability, which silently corrupts the calib JSON --
# "json.decoder.JSONDecodeError: Unterminated string" when parsing an Aria take's
# calibration. 1.3.0 does not truncate.
conda install -c conda-forge ffmpeg vrs=1.3.0 -y

# mmcv==1.3.9 (pulled in below by both HaMeR and requirements.txt) needs pkg_resources,
# which very recent setuptools no longer ships; but our own package and HaMeR's both need
# editable installs (PEP 660's build_editable hook, added in setuptools 64.0.0). 68.2.2 is
# the last easy-to-pin version with both pkg_resources and PEP 660 support -- installing it
# now, ahead of everything below, so it's active for every subsequent build in this script
pip install setuptools==68.2.2

# # install PHALP
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
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
pip install --no-build-isolation phalp[all]@git+https://github.com/brjathu/PHALP.git

# install HaMeR
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
# --no-build-isolation for the same reason as PHALP above: hamer's own setup.py pulls in
# detectron2 (and mmcv) as direct dependencies, which need to see the already-installed
# torch/setuptools rather than an isolated build env
pip install --no-build-isolation -e .[all]
cd ..

# install remaining requirements
# --no-build-isolation: chumpy's setup.py does `import pip` directly (a legacy pattern),
# which fails in an isolated build env that doesn't include pip itself
pip install --no-build-isolation -r requirements.txt

# pyrender (in requirements.txt) pulls in PyOpenGL unpinned; pin it to the
# exact version pyrender actually requires (headless rendering uses EGL, so
# this version's older osmesa support doesn't matter here)
pip install pyopengl==3.1.0

# install source
pip install -e .

# install ViTPose
pip install -v -e third-party/ViTPose

# install DROID-SLAM
cd third-party/DROID-SLAM
python setup.py install
# --no-build-isolation: lietorch's setup.py imports torch directly to build its CUDA
# extensions, same issue as PHALP/HaMeR above
pip install --no-build-isolation thirdparty/lietorch
cd ../..

# projectaria-tools (Aria MPS/timesync parsing, used for cooking/bike takes) and opencv
# (pulled in transitively above) both resolve to numpy>=2 if left unpinned, which breaks
# ABI compatibility with mmcv/xtcocotools's compiled extensions (built against numpy 1.x):
# "ValueError: numpy.dtype size changed, may indicate binary incompatibility". Re-pinning
# numpy last, after everything above has had a chance to pull in whatever it wants.
pip install projectaria-tools==1.6.0
pip install numpy==1.26.4
