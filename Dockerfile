FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p cache output data

EXPOSE 8501

CMD ["streamlit", "run", "src/main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]