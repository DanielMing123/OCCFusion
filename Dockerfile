ARG PYTORCH="2.1.1"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# RUN mkdir /project
# RUN mkdir /scratch
# WORKDIR /project/RDS-FEI-OccFusion-RW/OCCFusion
WORKDIR /workdir

ENV FORCE_CUDA="1"
ENV MMCV_WITH_OPS=1
ENV TZ=Australia/Sydney \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y ffmpeg git htop vim wget unzip sudo\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
RUN pip install pykitti
RUN pip install timm
RUN pip install debugpy
RUN pip install Cython
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install 'mmcv>=2.0.0rc4'
RUN mim install 'mmdet>=3.0.0'
RUN mim install 'mmdet3d>=1.1.0'
RUN pip install "mmsegmentation>=1.0.0"
RUN pip install focal_loss_torch



