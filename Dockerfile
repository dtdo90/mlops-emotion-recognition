FROM amazon/aws-lambda-python:3.11

# Install system dependencies using yum
# Install libGL and other necessary build tools using yum
RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y cmake libGL libGL-devel && \
    yum clean all

# Set the working directory
WORKDIR /app


# configure AWS as remote storage
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
# aws credentials
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Set language environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Copy the requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Install DVC with S3 support
RUN pip install "dvc[s3]"   # since s3 is the remote storage

# Copy the rest of the application files to /app
COPY . /app

# Initialize DVC and configure remote storage
RUN dvc init --no-scm
RUN dvc remote add -d model-store s3://models-dvc-remote/trained_models/


# pull the trained model
RUN dvc pull models/trained_model.dvc

# Specify the handler
#RUN python lambda_handler.py
CMD [ "lambda_handler.lambda_handler"]
