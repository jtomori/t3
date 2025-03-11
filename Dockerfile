FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends sox ffmpeg git gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY t3 /app/t3/

COPY requirements.txt /app/

COPY tip-toi-reveng/libtiptoi.c /app/

WORKDIR /app/

RUN gcc libtiptoi.c -o libtiptoi && \
    pip install --no-cache-dir numpy typing_extensions && \
    pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "-m", "t3"]
