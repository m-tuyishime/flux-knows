FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-pip \
    python3-venv \
    openssh-server \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create and set up a Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip 

# Configure SSH
# Create the directory for the SSH server to run
RUN mkdir -p /var/run/sshd 

# Disable password authentication for security, allow only key-based auth
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# expose SSH port
EXPOSE 22

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]