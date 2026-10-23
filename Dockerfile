FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Added --timeout 1000 and --retries 5 to prevent network dropouts during heavy downloads
RUN pip install --no-cache-dir --timeout 1000 --retries 5 --prefix=/install -r requirements.txt

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY ./app /app/app

# Fixed duplicate copy and safely added results directory
COPY ./results /app/results

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]