# Build for x86_64 in container
FROM --platform=linux/amd64 python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

COPY pyproject.toml uv.lock ./


RUN uv sync --frozen --no-cache


COPY src ./src


CMD ["uv", "run", "fastapi", "run", "src/main.py", "--port", "8000"]

