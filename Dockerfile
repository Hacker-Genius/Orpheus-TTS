# Use Python 3.10 as the base image
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Final stage
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy only the necessary files from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app/main.py .
COPY --from=builder /app/requirements.txt .

# Start the container
CMD ["python3", "-u", "main.py"] 
