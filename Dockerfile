FROM nvidia/cuda:12.2.2-devel-ubuntu22.04



# --------------- Environment Variables --------------- #

ENV PATH="/code:$PATH"
ENV PATH="/code/TRELLIS:$PATH"
ENV PATH="/usr/local/lib/python3.10/site-packages:$PATH"
ENV PATH="/usr/local/lib/python3.10/dist-packages:$PATH"

ENV CUDA_HOME=/usr/local/cuda-12
ENV LIBRARY_PATH="${CUDA_HOME}/lib64"
ENV C_INCLUDE_PATH="${CUDA_HOME}/include"

ENV LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/site-packages/nvidia/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/code/TRELLIS/

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"



# --------------- Code/Setup File Transfer --------------- #

COPY /trellis /code/trellis
COPY /x2sim/requirements.txt /code/requirements.txt



# --------------- System Installs --------------- #

RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y python3.10 \
&& apt-get install -y python3-pip \
&& apt-get install -y git \
&& apt-get install -y git-lfs \
&& apt-get install -y wget \
&& apt-get install -y mpich

RUN pip3 install --upgrade pip



# --------------- Python Package Installs --------------- #

RUN pip3 install -r /code/requirements.txt
RUN export MAX_JOBS=1 && /code/trellis/setup.sh

###
