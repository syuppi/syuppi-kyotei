FROM python:3.12-slim

WORKDIR /app

# LightGBM 実行に必要
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/start_render.sh

ENV PYTHONPATH=src
ENV BOATRACE_PREDICT_ONLY=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["bash", "scripts/start_render.sh"]
