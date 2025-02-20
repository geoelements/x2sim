FROM nvidia/cuda:12.2.2-devel-rockylinux9

RUN yum update -y && yum install -y python-devel python3.9 && yum install -y python-pip && yum install -y git && yum install -y which

RUN pip3 install --upgrade pip

COPY TRELLIS /code/TRELLIS
RUN chmod ugo+wrx /code
RUN chmod ugo+wrx /code/TRELLIS

ENV PATH="/code:$PATH"
ENV PATH="/code/TRELLIS:$PATH"
ENV CUDA_HOME=/usr/local/cuda-12
ENV LIBRARY_PATH="${CUDA_HOME}/lib64"
ENV C_INCLUDE_PATH="${CUDA_HOME}/include"
ENV LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.9/site-packages/nvidia/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.9/site-packages/nvidia/cuda_cupti/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/code/TRELLIS/

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6+PTX"

RUN /code/TRELLIS/setup.sh && pip3 install openai


### test
