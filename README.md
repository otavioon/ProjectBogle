# ProjectBogle

The goal of this project is to build a predictive model that maps programs to optimization sequences for code-size reduction. In this effort, we shall investigate different ways to produce candidate sequences, and different program representations that can be used to guide prediction.

This repository contains a reproducible framework to generate and analyse code size reduction, powered by [YaCoS](https://github.com/ComputerSystemsLab/YaCoS).

## Instalation and Execution

1. This framework uses Docker to enable a reproducible approach. You may wish to install Docker packages before continue.

2. Download YaCos and extract it inside YaCoS directory (that is inside this repository). We tested with YaCos development [branch at this commit](https://github.com/ComputerSystemsLab/YaCoS/commit/290b1e0b8fadbe39602a44cb5cf7791b60a67759).

> **warning**: When extracting YaCoS, the root must placed inside YaCos directory, without any sub-directory.

3. Run docker build to build an YaCoS image, using the command below. The image will be called `yacos-gpu` and the tag will be `2.0`.

`docker build -t yacos-gpu:2.0`


> **warning**: The dockerfile uses a GPU version for tensorflow and other packages. If you do not want, change line \#3 from Dockerfile file from `FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04` to `FROM ubuntu:20.04`.

