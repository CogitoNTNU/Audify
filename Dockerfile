# Bygg for x86_64 inne i containeren
FROM --platform=linux/amd64 python:3.12-slim

# uv (pakkehåndterer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# VIKTIG: Kopier prosjektfilene før uv sync
COPY pyproject.toml uv.lock ./

# Installer låste avhengigheter
RUN uv sync --frozen --no-cache

# Kopier kilden etterpå (bedre lagdeling)
COPY src ./src

# Start appen
CMD ["fastapi", "run", "src/main.py", "--port", "80"]
