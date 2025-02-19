FROM python:3.8.16-slim-bullseye
RUN apt-get update
RUN apt-get install -y curl vim wget nano
RUN apt-get install -y libgl1-mesa-dev
COPY requirements.txt /
RUN pip install --trusted-host pypi.python.org -r requirements.txt
COPY / /
CMD ["python", "main.py"]
