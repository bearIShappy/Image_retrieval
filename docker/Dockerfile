FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for opencv-python and other vision libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the typo'd requirements file and install dependencies
COPY requiremnents.txt .
RUN pip install --no-cache-dir -r requiremnents.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Command to run the backend application
CMD ["python", "app.py"]
