# Use the TensorFlow 2.12 base image
FROM tensorflow/tensorflow:2.12.0-gpu

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Your additional customizations here
# ...

