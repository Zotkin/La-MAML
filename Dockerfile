FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
RUN mkdir /code && mkdir /code/data && mkdir /code/logs
COPY . /code
WORKDIR /code
RUN pip install -r requirements.txt