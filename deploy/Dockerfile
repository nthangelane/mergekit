# syntax=docker/dockerfile:1.6
FROM python:3.10-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY mergekit ./mergekit
COPY docs ./docs
COPY examples ./examples
COPY tests ./tests

RUN pip install --upgrade pip \
    && pip install .[evolve-ga,eks]

COPY . .

ENTRYPOINT ["mergekit-evolve-ga"]
