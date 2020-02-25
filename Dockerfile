FROM ubuntu:latest

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip \
  && apt-get update && apt-get install -y git

COPY requirements.txt /app/

RUN pip install -r app/requirements.txt

RUN apt-get update \
    && apt-get install -y libsm6 libxext6 libxrender-dev \
    && pip install opencv-python


COPY . /app

WORKDIR /app

CMD python -u app.py

