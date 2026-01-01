# FROM python:3.11-slim

# # Không tạo pyc, log ra thẳng
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# WORKDIR /app

# # (optional) nếu bạn cần build một số lib (thường torch/transformers không cần)
# # RUN apt-get update && apt-get install -y --no-install-recommends \
# #     git curl && rm -rf /var/lib/apt/lists/*

# # Cài deps trước để tận dụng cache
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy code
# COPY . .

# # Expose port uvicorn
# EXPOSE 8000

# # Chạy server
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# (tuỳ lib) có thể cần build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
