FROM openvino/ubuntu18_dev:2019_R3.1

ADD . /app
WORKDIR /app

ARG INSTALL_DIR=/opt/intel/computer_vision_sdk

RUN apt-get update && apt-get -y upgrade && apt-get autoremove

#Pick up some TF dependencies
RUN apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        cpio \
        curl \
        vim \
        git \
        lsb-release \
        pciutils \
        python3.5 \
        python3-pip \
        cmake \
        sudo 

RUN pip3 install --upgrade pip setuptools wheel


CMD ["/bin/bash"]
