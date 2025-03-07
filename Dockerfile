FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
LABEL branch="khansamad47/openpose/master"
LABEL description="CMU version"

RUN echo "Installing dependencies..." && \
	apt-get -y --no-install-recommends update && \
	apt-get -y --no-install-recommends upgrade && \
	apt-get install -y --no-install-recommends \
	build-essential \
	cmake \
	git \
	libatlas-base-dev \
	libprotobuf-dev \
	libleveldb-dev \
	libsnappy-dev \
	libhdf5-serial-dev \
	protobuf-compiler \
	libboost-all-dev \
	libgflags-dev \
	libgoogle-glog-dev \
	liblmdb-dev \
	pciutils \
	python3-setuptools \
	python3-dev \
	python3-pip \
	opencl-headers \
	ocl-icd-opencl-dev \
	libviennacl-dev \
	libcanberra-gtk-module \
	libopencv-dev && \
	python3 -m pip install \
	numpy \
	protobuf \
	opencv-python

RUN echo "Downloading and building OpenPose..." && \
	git clone -b master https://github.com/khansamad47/openpose.git && \
	mkdir -p /openpose/build && \
	cd /openpose/build && \
	cmake -DBUILD_PYTHON=ON .. && \
	make -j`nproc` && \
	make install

WORKDIR /openpose
