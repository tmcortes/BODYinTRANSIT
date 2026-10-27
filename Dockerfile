# Default base image for modern GPUs (RTX 3090/5090)
# For TITAN X (Pascal), build using:
# docker build --build-arg PYTORCH_TAG=2.4.0-cuda12.1-cudnn9-devel -t deep_learning .
# For newer model archs like 5090 use:
# docker build -t deep_learning .

ARG PYTORCH_TAG=2.10.0-cuda13.0-cudnn9-devel
FROM pytorch/pytorch:${PYTORCH_TAG}

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install essential system tools
RUN apt-get update && apt-get install -y \
    git \
    qpdf \
    ffmpeg \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user 'dev' with sudo access
RUN useradd -m -s /bin/bash dev && \
    usermod -aG sudo dev && \
    echo "dev ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Set the working directory
WORKDIR /workspaces/body_in_transit

# Ensure directory is writable by the dev user
RUN chown -R dev:dev /workspaces

# Copy and install Python dependencies
# Note: Using --break-system-packages to handle PEP 668 in modern Python versions
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Switch to the dev user
USER dev
