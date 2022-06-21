FROM python:3.6-stretch

RUN pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
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
EXPOSE 443
EXPOSE 80

# Install Caddy
RUN apt install -y debian-keyring debian-archive-keyring apt-transport-https
RUN curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
RUN curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
RUN apt update
RUN apt install caddy

COPY entrypoint.sh entrypoint.sh
RUN chmod u+x entrypoint.sh

CMD ["bash", "entrypoint.sh"]