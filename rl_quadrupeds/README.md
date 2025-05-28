# RL Quadrupeds

This folder contains several Python packages with custom functionalities for training Reinforcement Learning policies for quadruped robots.

## Prerequisites

- IsaacSim 4.5.0
- IsaacLab

## Setting Up

Install [IsaacSim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html) and [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html). Summarizing the content of the tutorials, you can install IsaacSim by downloading the release from [this link](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release) and running the following commands:

```bash
mkdir ~/isaacsim
cd ~/Downloads
unzip <isaacsim-zip-file-you-downloaded> -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
```

You can install IsaacLab by running the following commands:

```bash
export ISAACSIM_PATH="${HOME}/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
cd ~
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
ln -s ${ISAACSIM_PATH} _isaac_sim
sudo apt install cmake build-essential
./isaaclab.sh --install
```

We use the Python environment inside IsaacLab to make development easier. Because of that, defining some aliases is helpful:

```bash
echo "alias isaacsim='~/isaacsim/isaac-sim.sh'" >> ~/.bashrc
echo "alias isaaclab='~/IsaacLab/isaaclab.sh'" >> ~/.bashrc
echo "export ISAACLAB_FOLDER='~/IsaacLab/'" >> ~/.bashrc
source ~/.bashrc
```

To install the custom Python packages containing quadruped robot functionality:

```bash
cd rl_quadrupeds # the root of the repository
source ./rl_quadrupeds/install.sh install --isaaclab
```

## Usage

**(This section assumes you are inside the repository root folder)**

To use it, run the skrl scripts inside the `rl_quadrupeds/skrl` folder. To train a Go1 locomotion policy with 1 GPU, run:

```bash
isaaclab -p rl_quadrupeds/skrl/train.py --task=Go1-Locomotion
```

If you do not want to render the simulator, run:

```bash
isaaclab -p rl_quadrupeds/skrl/train.py --task=Go1-Locomotion --headless
```

For multiple GPUs, run:

```bash
isaaclab -p -m torch.distributed.run --nnodes=<NUM_NODES> --nproc_per_node=<NUM_GPUS> rl_quadrupeds/skrl/train.py --task=Go1-Locomotion --distributed
```

To play the policy, run:

```bash
isaaclab -p rl_quadrupeds/skrl/play.py --task=Go1-Locomotion --num_envs 32 --checkpoint <CHECKPOINT_PATH>
```
