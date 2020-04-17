FROM openvino/ubuntu18_dev:2019_R3.1

ADD . /app
WORKDIR /app

USER root
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
        libgflags-dev \
        libboost-dev \
        libboost-log-dev \
        libsndfile1-dev \
        libao-dev \
        cmake \
        libx11-dev \
        sudo 

RUN pip3 install --upgrade pip setuptools wheel Flask==1.0.2

COPY DriverBehavior/* app/
WORKDIR /app/DriverBehavior
RUN mkdir build
WORKDIR /app/DriverBehavior/build
RUN chmod +x /app/DriverBehavior/scripts/setupenv.sh
RUN /bin/bash -c 'source /opt/intel/openvino/bin/setupvars.sh && source /app/DriverBehavior/scripts/setupenv.sh && cmake -DCMAKE_BUILD_TYPE=Release ../ && make'
RUN /bin/bash -c 'source /opt/intel/openvino/bin/setupvars.sh && source /app/DriverBehavior/scripts/download_models.sh'

COPY ActionRecognition/* app/
WORKDIR /app/ActionRecognition
RUN /bin/bash -c 'source /opt/intel/openvino/bin/setupvars.sh && source /app/ActionRecognition/scripts/download_models.sh'

CMD ["/bin/bash"]
