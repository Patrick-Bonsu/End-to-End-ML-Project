# Use the official Python image as the base image
FROM python:3.8-slim

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Create and set the working directory
WORKDIR /usr/app

# Copy the project code into the container
COPY . /usr/app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Define the command to run your Flask app
CMD ["flask", "run"]
