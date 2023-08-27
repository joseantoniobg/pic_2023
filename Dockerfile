FROM ubuntu:22.04

WORKDIR /app

COPY . .

RUN apt-get update \
    && apt-get install python3 -y \
    && apt-get install python3-pip -y \
    && pip install -U numpy \
    && pip install -U matplotlib \
    && pip install -U pandas \
    && pip install -U scikit-learn

ENTRYPOINT ["tail", "-f", "/dev/null"]