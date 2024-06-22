# Use the official Python image as a base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=main.py

# Create a working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose the port the app runs on
EXPOSE 80

# Start the Flask app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:80", "main:app", "--worker-class", "eventlet", "-w", "1"]
# 
