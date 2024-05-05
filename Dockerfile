FROM python:3.10.14-bookworm as python-base

RUN pip install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install git+https://github.com/m-bain/whisperx.git
RUN pip install yt-dlp
RUN pip install moviepy


FROM python-base as python-base
RUN apt update 
RUN apt install ffmpeg -y

FROM python-base as python-base
COPY ./preload_model.py /opt/preload_model.py
COPY ./test.aac /opt/test.aac
WORKDIR /opt
RUN python preload_model.py

FROM python-base as python-app
COPY ./app /app
WORKDIR /app
RUN mkdir -p /app/output && mkdir -p /app/source

CMD ["python3", "app.py"]

