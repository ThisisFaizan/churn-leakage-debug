#FROM ubuntu:24.04

# Install or copy over any environment dependencies here.

#WORKDIR /app
FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends tmux asciinema \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.lock \
    && rm -f /tmp/requirements.lock

COPY churn/ /app/churn/

WORKDIR /app/churn
