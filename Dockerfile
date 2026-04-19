FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY dependency_info.txt .
RUN pip install --no-cache-dir -r dependency_info.txt

COPY . .

CMD ["python", "-u", "app/semantic_search.py"]