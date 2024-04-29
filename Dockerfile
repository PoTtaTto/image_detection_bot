# Use the official Python 3.10 image as the base image
FROM python:3.10

# Copy the current directory contents into the /usr/src/app/image_detection_bot/ directory of the container
COPY . /usr/src/app/image_detection_bot/

# Set the working directory to /usr/src/app/image_detection_bot/
WORKDIR /usr/src/app/image_detection_bot/

# Install the Python dependencies specified in the requirements.txt file
# Use cache for pip to speed up the process
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /usr/src/app/image_detection_bot/requirements.txt

# Command to run the Python script start.py when the container starts
CMD ["python", "start.py"]
