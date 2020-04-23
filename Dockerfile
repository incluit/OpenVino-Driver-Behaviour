FROM openvino/ubuntu18_dev:2019_R3.1

ADD . /app
WORKDIR /app

USER root
RUN apt-get update && apt-get -y upgrade && apt-get autoremove

#Pick up some TF dependencies
RUN apt-get install -y --no-install-recommends \
        build-essential \
        xdg-utils \
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
        libssl-dev \
        locales \
        libjpeg8-dev \
        libopenblas-dev \
        sudo 

RUN pip3 install --upgrade pip setuptools wheel Flask==1.0.2 AWSIoTPythonSDK

COPY DriverBehavior/* app/
WORKDIR /app/DriverBehavior
RUN git clone --recursive https://github.com/awslabs/aws-crt-cpp.git
RUN mkdir build

RUN chmod +x /app/DriverBehavior/scripts/setupenv.sh

# Ros2
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN apt install curl gnupg2 lsb-release
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list'
ARG DEBIAN_FRONTEND=noninteractive
RUN sudo apt install 
RUN apt update && apt install -y ros-crystal-ros-base

RUN apt update && apt install -y \
        python3-colcon-common-extensions \
        ros-crystal-rosbag2-test-common \
        ros-crystal-rosbag2-storage-default-plugins \
        ros-crystal-rosbag2-storage
        
RUN apt-get install -y --no-install-recommends \
        ros-crystal-sqlite3-vendor \
        ros-crystal-ros2bag*

# ETS-ROS2
RUN git clone https://github.com/HernanG234/ets_ros2/
WORKDIR /app/DriverBehavior/ets_ros2
RUN /bin/bash -c 'source /opt/intel/openvino/bin/setupvars.sh && source /opt/ros/crystal/setup.bash && colcon build --symlink-instal --parallel-workers 1 --cmake-args -DSIMULATOR=ON -DBUILD_DEPS=ON'

WORKDIR /app/DriverBehavior/build
RUN /bin/bash -c 'source /opt/ros/crystal/setup.bash && source /app/DriverBehavior/ets_ros2/install/setup.bash && source /opt/intel/openvino/bin/setupvars.sh && source /app/DriverBehavior/scripts/setupenv.sh && cmake -DCMAKE_BUILD_TYPE=Release -DSIMULATOR=ON -DBUILD_DEPS=ON ../ && make'
RUN /bin/bash -c 'source /opt/ros/crystal/setup.bash && source /app/DriverBehavior/ets_ros2/install/setup.bash && source /opt/intel/openvino/bin/setupvars.sh && source /app/DriverBehavior/scripts/download_models.sh'


COPY ActionRecognition/* app/
WORKDIR /app/ActionRecognition
RUN /bin/bash -c 'source /opt/intel/openvino/bin/setupvars.sh && source /app/ActionRecognition/scripts/download_models.sh'

COPY UI/* app/
COPY AWS/* app/
WORKDIR /app/UI
COPY entrypoint.sh /
EXPOSE 5000
RUN chmod +x /entrypoint.sh
# ENTRYPOINT ["/entrypoint.sh"]

CMD ["/bin/bash"]
