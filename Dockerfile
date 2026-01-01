FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model path trên Render (persistent disk)
ENV MODEL_DIR=/var/data/models/finetuning-phase2
RUN mkdir -p "$MODEL_DIR"

EXPOSE 8000

# QUAN TRỌNG: tải model trước rồi mới start app
CMD ["sh", "-c", "python scripts/ensure_model.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
