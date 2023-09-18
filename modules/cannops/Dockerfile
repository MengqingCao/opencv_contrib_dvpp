# User guides
#
# 0. Install Ascend driver on host.
#    (https://www.hiascend.com/en/hardware/firmware-drivers)
#
# 1. Run docker container.
# docker run -it \
#    --name opencv \
#    --device /dev/davinci0 \
#    --device /dev/davinci_manager \
#    --device /dev/devmm_svm \
#    --device /dev/hisi_hdc \
#    -v /usr/local/dcmi:/usr/local/dcmi \
#    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
#    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
#    opencv bash
#
# 2. Check environment.
# npu-smi info
#
# 3. Compile opencv with Ascend NPU backend.
# cmake -DWITH_CANN=1
#
# 4. Run opencv_test_cannops.
# ./bin/opencv_test_cannops

FROM ubuntu:20.04

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    git \
    wget \
    gcc \
    g++ \
    cmake \
    make \
    zlib1g \
    zlib1g-dev \
    openssl \
    libsqlite3-dev \
    libssl-dev \
    libffi-dev \
    unzip \
    pciutils \
    net-tools \
    libblas-dev \
    gfortran \
    patchelf \
    libblas3 && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.9
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /root/miniconda && \
    rm -f ~/miniconda.sh
RUN /root/miniconda/bin/conda create --name opencv -y python=3.9
ENV PATH=/root/miniconda/envs/opencv/bin/:$PATH
RUN pip install numpy sympy decorator scipy attrs

# Install CANN
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH \
    CANN_TOOLKIT_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C13SPC702/Ascend-cann-toolkit_7.0.RC1.alpha002_linux-x86_64.run

RUN wget ${CANN_TOOLKIT_URL} && \
    chmod +x Ascend-cann-toolkit_7.0.RC1.alpha002_linux-x86_64.run && \
    ./Ascend-cann-toolkit_7.0.RC1.alpha002_linux-x86_64.run --install && \
    rm -f ./Ascend-cann-toolkit_7.0.RC1.alpha002_linux-x86_64.run

ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH:/usr/lib64
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:$PATH
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
