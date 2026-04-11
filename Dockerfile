FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends lua5.4 curl \
    && ln -sf /usr/bin/luac5.4 /usr/bin/luac \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
