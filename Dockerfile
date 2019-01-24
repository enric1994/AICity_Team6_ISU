FROM mxnet/python:gpu
WORKDIR /mxnet
RUN apt-get -y update && \
    apt-get -y install \
        git \
		libgtk2.0-dev \
		python-tk

RUN pip install opencv-python setuptools PyYAML easydict pillow cython matplotlib

RUN git clone https://github.com/msracver/Deformable-ConvNets.git /mxnet
RUN sh /mxnet/init.sh

# place https://1drv.ms/u/s!Am-5JzdW2XHzhqMSjehIcCgAhvEAHw into the model folder
