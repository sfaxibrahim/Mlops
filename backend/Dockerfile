FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Accept the build argument for the model URL
ARG MODEL_URL
# Set the URL as an environment variable for runtime use
ENV MODEL_URL=${MODEL_URL}

# Download the model directly from the provided URL
RUN mkdir -p /app/models && \
    curl -L ${MODEL_URL} -o /app/models/model.pkl

# Expose the port for the backend service
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]