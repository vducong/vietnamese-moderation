# Use a lightweight base image
FROM openjdk:slim
COPY --from=python:3.9-slim / /

# Set the working directory
WORKDIR /app

# Copy the application files to the container
COPY . /app

# Install required packages
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install make automake gcc g++ subversion python3-dev gfortran
RUN apt-get -y install cmake build-essential pkg-config libgoogle-perftools-dev google-perftools
RUN pip install --upgrade pip setuptools
RUN pip3 install sentencepiece
RUN PIP_DEFAULT_TIMEOUT=1000 && pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install torch==2.0.0 torchaudio==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install tensorflow
RUN pip3 install py_vncorenlp

EXPOSE 8080

ENV GOOGLE_APPLICATION_CREDENTIALS resources/firebase.dev.json

# Start the app
CMD exec gunicorn main:app --workers 1 --worker-class gthread --bind :5000 --timeout 0
