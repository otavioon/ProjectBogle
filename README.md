# ProjectBogle

The goal of this project is to build a predictive model that maps programs to optimization sequences for code-size reduction. In this effort, we shall investigate different ways to produce candidate sequences, and different program representations that can be used to guide prediction.

This repository contains a reproducible framework to generate and analyse code size reduction, powered by [YaCoS](https://github.com/ComputerSystemsLab/YaCoS).

# Instalation and Execution

1. This framework uses Docker to enable a reproducible approach. You may wish to install Docker packages before continue.

2. Download YaCos and extract it inside YaCoS directory (that is inside this repository). 

> **note**: We tested with YaCos development [branch at this specific commit](https://github.com/ComputerSystemsLab/YaCoS/commit/290b1e0b8fadbe39602a44cb5cf7791b60a67759).

> **warning**: When extracting YaCoS, the root must placed inside YaCos directory, without any sub-directory.

3. Run docker build to build an YaCoS image, using the command below. The image will be called `yacos-gpu` and the tag will be `2.0`. 

`docker build -t yacos-gpu:2.0`


> **warning**: The dockerfile uses a GPU version for tensorflow and other packages. If you do not want, change line \#3 from Dockerfile file from `FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04` to `FROM ubuntu:20.04`.

4. You can step inside YaCos docker (which will open a shell) using the command below.

`docker run --gpus all -it -v ~/ProjectBogle/:/home/nonroot/experiment --rm yacos-gpu:2.0 bash`

5. Or you can start a jupyterlab server using the command below.
    - The optional `-v` parameter will map a local user directory (`/~/ProjectBogle/t`, in the example) inside Docker container (`/home/nonroot/experiment`)
    - You can connect to jupyter's server at `localhost:8888` address. 


`docker run --gpus all -it -v ~/ProjectBogle/:/home/nonroot/experiment -p 8888:8888 --ipc=host --rm yacos-gpu:2.1 python -m jupyterlab --port 8888 --no-browser --ip='*' --NotebookApp.token='' --NotebookApp.password=''`

> **note**: the Dockerfile will create a user called `nonroot` with home directory at `/home/nonroot/`.

> **note**: if you are not using gpus, remove the `--gpus all` argument for the command-line.

## Tips

### Remote SSH connection to jupyter-lab

You may want to start jupyter's Docker in one machine (server) and connect to jupyterlab using another one (client). A simple solution is performing a [SSH Port Forwarding](https://www.ssh.com/academy/ssh/tunneling/example).

The following example will redirect port 8888 on the server (`gw.example.com`) to the local machine, at port 8888. Now you can access jupyter-lab (on port 8888) in the client machine.

`ssh -NT -L 8888:localhost:8888 gw.example.com`

### Keep jupyter-lab alive

You may want to keep jupyterlab running indefinetly on the server machine. You can use tmux and [dettach a session](https://danielmiessler.com/study/tmux/).

### Using Bourne-again Shell in jupyter-lab

You may wish to use bourne again shell instead of `sh` in your jupyter lab. To do this, export `SHELL` variable before executing jupyter, using the following command.

`export SHELL=/bin/bash`

### Permissions

You may wish to turn data directory readable and writable.


# Usage

In general, datasets and representations must be stored (or linked) to `datasets/` directory. Python notebooks must be stored at `notebooks/` directory. Additional scripts must be placed in `scripts/` directory and experiment results must be placed at `results/` directory. Finally, the `YaCoS/` directory contents may be removed after generating the Docker image. 

You may want to check the notebooks at `/home/nonroot/experiment/notebook/` directory to check several experiments performed. 
