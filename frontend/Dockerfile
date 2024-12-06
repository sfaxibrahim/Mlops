FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app/frontend


COPY requirements.txt /app/frontend/
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
