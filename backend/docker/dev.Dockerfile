FROM python:3.12-slim-bookworm

ARG APP_HOME=/app
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR $APP_HOME

ENV PIPENV_CACHE_DIR=/root/.cache/pip

# Install build-essential package (includes gcc, make, etc.)
RUN apt-get update && apt-get install -y build-essential ffmpeg curl \
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/*

# use pipenv to manage virtualenv
ENV PATH=/.venv/bin:$PATH
RUN python -m venv /.venv
RUN --mount=type=cache,target=/root/.cache/pip pip install pipenv==2025.0.3

COPY Pipfile Pipfile.lock $APP_HOME/
RUN --mount=type=cache,target=/root/.cache/pip pipenv  install --system --categories "packages dev-packages" --deploy

# Copy the rest of the application code
COPY . $APP_HOME/

EXPOSE 8000

# Command to run the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
