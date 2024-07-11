FROM python:3.12.4

WORKDIR /face-detection

RUN apt-get update && apt-get install -y libgl-dev

ADD requirements.txt .
RUN python3 -m pip install -r requirements.txt

ADD face-detection.py .

CMD ["python3", "face-detection.py"]
