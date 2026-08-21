FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
RUN pip install --no-cache-dir fastapi uvicorn pydantic qdrant-client pytest

COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "examples.enterprise_rag_platform.server:app", "--host", "0.0.0.0", "--port", "8000"]
