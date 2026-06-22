ARG SKIP_FRONTEND=false
ARG COMMON_IMAGE_TAG=tero-common

FROM ${COMMON_IMAGE_TAG} AS frontend-builder
ARG SKIP_FRONTEND
ARG VITE_APP_VERSION=0.0.0

WORKDIR /app
COPY pnpm-workspace.yaml ./

WORKDIR /app/frontend
COPY frontend/package.json ./

RUN if [ "$SKIP_FRONTEND" = "false" ]; then pnpm install --include=dev; fi

COPY frontend/ ./

ENV VITE_APP_VERSION=$VITE_APP_VERSION

RUN if [ "$SKIP_FRONTEND" = "false" ]; then pnpm run build; else mkdir dist; fi

FROM python:3.13

RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources; \
    fi; \
    apt-get update -o Acquire::ForceIPv4=true; \
    apt-get install -y --no-install-recommends netcat-traditional curl ca-certificates gnupg; \
    curl -fsSL https://packages.microsoft.com/keys/microsoft-2025.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg; \
    debian_version="$(. /etc/os-release && echo "$VERSION_ID")"; \
    curl -fsSL "https://packages.microsoft.com/config/debian/${debian_version}/prod.list" -o /etc/apt/sources.list.d/mssql-release.list; \
    apt-get update -o Acquire::ForceIPv4=true; \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 unixodbc; \
    rm -rf /var/lib/apt/lists/*

# This avoids unnecessary warning from transformers library due to missing pytorch and other libraries that are not really necessary.
ENV TRANSFORMERS_VERBOSITY=error

RUN pip install poetry

RUN poetry config installer.max-workers 10

WORKDIR /usr/src/app
ENV PYTHONPATH=/usr/src/app

COPY docker/wait-for-it.sh wait-for-it.sh
COPY backend/pyproject.toml backend/poetry.lock ./

RUN poetry install --no-root

COPY docker/preload_tokenizers.py preload_tokenizers.py
ARG PRELOADED_TOKENIZER_MODELS=
ENV PRELOADED_TOKENIZER_MODELS=${PRELOADED_TOKENIZER_MODELS}
RUN poetry run python preload_tokenizers.py

COPY docker/entrypoint.sh entrypoint.sh
COPY sample.env .env
COPY backend/alembic.ini alembic.ini
COPY backend/alembic ./alembic
COPY backend/tero ./tero
COPY --from=frontend-builder /app/frontend/dist ./frontend

RUN chmod +x wait-for-it.sh entrypoint.sh
ENV FRONTEND_PATH=frontend

ENTRYPOINT [ "./entrypoint.sh" ]

EXPOSE 8000

CMD ["poetry", "run", "python", "-m", "tero"]
