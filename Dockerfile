FROM python:3.9-slim

# Repository cloning options
ARG REPO_URL=https://github.com/yiino1222/eegemg_2025.git
ARG REPO_REF=main

WORKDIR /app

# Install git for cloning and clean up apt cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository directly during build
RUN git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" /tmp/repo \
    && cp -a /tmp/repo/. /app \
    && rm -rf /tmp/repo

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "run_pipeline.py"]
CMD ["--config", "/config/pipeline.json"]
