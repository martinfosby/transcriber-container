# Use Debian Slim as the base image
FROM python:3.13-slim

# Set environment variables
ENV PATH="/usr/local/bin:$PATH"
ENV LANG="C.UTF-8"

# Install dependencies in a single layer
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ffmpeg; \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Create a non-root user for security
# RUN addgroup --system appgroup && adduser --system --group appuser
# USER appuser
ENV HF_HOME="/app/.cache"
# COPY .cache /.cache

# Set working directory
WORKDIR /app

# Set environment variables (avoid hardcoding secrets)

# Copy files and set ownership
# COPY --chown=appuser:appgroup process.py /app/
# COPY --chown=appuser:appgroup user_sound_recordings /app/user_sound_recordings
# COPY --chown=appuser:appgroup .cache /app/.cache
COPY app .

# Set the entry point
ENTRYPOINT ["python3", "main.py", "--use-call-recording"]
