FROM python:3.6-stretch

RUN pip install numpy==1.16.6
RUN pip install flask==1.1.1
RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install numba==0.48
RUN pip install librosa==0.7.2
RUN pip install python-speech-features==0.6

RUN apt-get update -y
RUN apt-get install -y libsndfile1

COPY fbank_net/demo /fbank_net/demo
COPY fbank_net/model_training /fbank_net/model_training
COPY fbank_net/weights /fbank_net/weights
RUN touch /fbank_net/__init__.py

RUN mkdir /fbank_net/data_files

ENV PYTHONPATH="/fbank_net"
ENV FLASK_APP="demo/app.py"

WORKDIR /fbank_net

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]
