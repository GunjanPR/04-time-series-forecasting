# Reproducible benchmark + dashboard image (Linux, so Prophet's bundled cmdstan just works).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[dashboard]" && useradd --create-home --uid 10001 bench \
    && mkdir -p data results && chown bench data results
COPY docs ./docs

USER bench
EXPOSE 8501
CMD ["python", "-m", "ts_forecasting.benchmark"]
