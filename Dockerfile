# Use the TensorFlow GPU image as the base
FROM tensorflow/tensorflow:latest-gpu

# Copy requirements.txt to the container
COPY requirements.txt /tmp/

# Install additional Python packages
RUN cat /tmp/requirements.txt | xargs -n 1 pip install
# Any additional custom setup commands can go here

# Set the working directory (optional)
WORKDIR /app

