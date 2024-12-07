FROM python:3.12-slim

WORKDIR /app/frontend

COPY requirements.txt /app/frontend/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/frontend/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
