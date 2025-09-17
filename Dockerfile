FROM nvcr.io/nvidia/pytorch:24.11-py3

RUN mkdir -p /DLC2action
COPY . /DLC2action
WORKDIR /DLC2action

# Install required packages
RUN apt update
RUN apt install python3-pip -y
RUN pip install -e .

RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y git
RUN apt-get install vim -y
