# SOFA Setup for stEVE

stEVE requires SOFA with the `SofaPython3` and `BeamAdapter` plugins. I didn't know you had to build this but its something you have to build and it's 
separate from `requirements.txt`; `Sofa` is not a normal pip package.


Anyways.. If you didn't read their readme you might get this error during training:

```text
ModuleNotFoundError: No module named 'Sofa'
```

means the active Python environment cannot find the SOFA Python bindings.

## Check Current Environment

Run this inside the training environment:

```bash
conda activate steve_training
python -c "import Sofa; import SofaRuntime; print('SOFA OK')"
```

If that fails, SOFA is either not installed or `SOFA_ROOT` / `PYTHONPATH` is not
configured.

## Important Python Version Rule

The SOFA Python bindings must match the Python version used by the environment.
For the current setup:

```bash
conda activate steve_training
python --version
```

This environment uses Python 3.10, so SOFA must be built or installed with
Python 3.10 bindings. A SOFA build made for Python 3.8 will not import in this
environment.

## If SOFA Is Already Installed

Set the environment variables before running stEVE:

```bash
export SOFA_ROOT=/path/to/sofa/install
export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH

python -c "import Sofa; import SofaRuntime; print('SOFA OK')"
```

If the import succeeds, then run the stEVE function check:

```bash
python3 ./eve/examples/function_check.py
```

## If SOFA Is Not Installed

Follow the upstream stEVE SOFA instructions:

```text
https://github.com/lkarstensen/stEVE?tab=readme-ov-file#install-sofa-with-sofapython3-and-beamadapter
```
So basically here is what I did.

Use this path for **your setup**: `steve_training` conda env, Python 3.10, stEVE needing BeamAdapter-compatible SOFA. Do **not** use latest SOFA `v25.12` for this repo.

```bash
# 1. System build dependencies
sudo apt update
sudo apt install -y \
  build-essential software-properties-common git cmake ninja-build ccache \
  libtinyxml2-dev libopengl0 libboost-all-dev \
  libpng-dev libjpeg-dev libtiff-dev libglew-dev zlib1g-dev libeigen3-dev \
  xorg-dev libgtk-3-dev qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools \
  libqt5opengl5-dev qtwayland5
```

```bash
# 2. Activate your env and install Python build deps
conda activate steve_training
cd /home/eba/Desktop/RL_health/stEVE_training
python -m pip install -r requirements.txt
python -m pip install "pybind11==2.12.0" numpy scipy
```

```bash
# 3. Build SOFA for this conda env
mkdir -p ~/sofa_stEVE
cd ~/sofa_stEVE

git clone -b v22.12 --depth 1 https://github.com/sofa-framework/sofa.git src
mkdir -p build install

export PYTHON_EXEC="$CONDA_PREFIX/bin/python"
export PYTHON_LIB="$CONDA_PREFIX/lib/libpython3.10.so"
export PYTHON_INCLUDE="$CONDA_PREFIX/include/python3.10"
export PYTHON_SITE_PACKAGES="$CONDA_PREFIX/lib/python3.10/site-packages"
export PYBIND11_DIR="$($PYTHON_EXEC -c 'import pybind11; print(pybind11.get_cmake_dir())')"
export QT_DIR="/usr/lib/x86_64-linux-gnu/cmake/Qt5"

cmake \
  -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="$HOME/sofa_stEVE/install" \
  -D CMAKE_PREFIX_PATH="$QT_DIR;$PYTHON_SITE_PACKAGES" \
  -D SOFA_FETCH_SOFAPYTHON3=ON \
  -D SOFA_FETCH_BEAMADAPTER=ON \
  -D PYTHON_VERSION=3.10 \
  -D PYTHON_EXECUTABLE="$PYTHON_EXEC" \
  -D Python_EXECUTABLE="$PYTHON_EXEC" \
  -D Python_LIBRARY="$PYTHON_LIB" \
  -D Python_INCLUDE_DIR="$PYTHON_INCLUDE" \
  -D pybind11_DIR="$PYBIND11_DIR" \
  -D SP3_LINK_TO_USER_SITE=ON \
  -D SP3_PYTHON_PACKAGES_LINK_DIRECTORY="$PYTHON_SITE_PACKAGES" \
  -S src \
  -B build

cmake \
  -D PLUGIN_SOFAPYTHON3=ON \
  -D PLUGIN_BEAMADAPTER=ON \
  -S src \
  -B build

ninja -C build -j 12
ninja -C build install
```

```bash
# 4. Use SOFA from this env
export SOFA_ROOT="$HOME/sofa_stEVE/install"
export PYTHONPATH="$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="$SOFA_ROOT/lib:$LD_LIBRARY_PATH"

python -c "import Sofa; import SofaRuntime; SofaRuntime.importPlugin('BeamAdapter'); print('SOFA OK')"
```

Then test stEVE:

```bash
cd /home/eba/Desktop/RL_health/stEVE_training
python3 ./eve/examples/function_check.py
```

I chose `v22.12` because this repo’s own Dockerfile uses SOFA `v22.12`, and stEVE notes that newer BeamAdapter versions changed compatibility. SOFA’s Linux docs confirm the Linux build dependencies and CMake/Ninja flow, while stEVE’s README specifically requires SOFA with `SofaPython3` and `BeamAdapter`. Sources: https://sofa-framework.github.io/doc/getting-started/build/linux/ and https://github.com/lkarstensen/stEVE?tab=readme-ov-file#install-sofa-with-sofapython3-and-beamadapter



For this repo, SOFA must be built with:

- `SofaPython3`
- `BeamAdapter`
- the same Python executable as the active conda environment

The local stEVE instructions are also available here:

```text
eve/README.md
```

## After SOFA Works

Install and test the local packages:

```bash
python3 -m pip install -e ./eve
python3 ./eve/examples/function_check.py

python3 -m pip install -e ./eve_bench
python3 ./eve_bench/examples/function_check.py

python3 -m pip install -e ./eve_rl
python3 ./eve_rl/examples/function_check.py
```

Then run a small training smoke test before launching the full 29-worker run:

```bash
python3 ./training_scripts/BasicWireNav_train.py -d cuda -nw 2 -n BasicWireNav_smoke
```
