FROM safarmirek/accelergy-timeloop-infrastructure:latest-amd64

LABEL maintainer="xsafar23@vutbr.cz"

WORKDIR /opt/app-root

COPY src/ .

RUN pip3 install -U setuptools wheel pip packaging
RUN pip3 install tensorflow-model-optimization==0.7.3
RUN pip3 install tensorflow-metadata==1.12.0
RUN pip3 install protobuf==3.19.6
RUN pip3 install tensorflow-datasets==4.8.2
RUN pip3 install py-paretoarchive==0.19

RUN mkdir -p nsga_runs
RUN mkdir -p cache
RUN mkdir -p logs
RUN mkdir -p checkpoints

ENTRYPOINT /bin/bash

