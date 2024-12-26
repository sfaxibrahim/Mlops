FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install "dvc[s3]"

COPY . /app/

RUN git init && \
    rm -rf .dvc && \
    dvc init --no-scm && \
    dvc remote add -d storage https://dagshub.com/sfaxibrahim/Mlops.s3 && \
    dvc remote modify storage auth basic && \
    dvc remote modify storage user ${c35b367581e71547e09d600ef0c0fb2e64b01f5a} && \
    dvc pull

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]