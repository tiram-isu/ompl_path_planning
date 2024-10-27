# Use the existing OMPL Docker image as the base
FROM deividtesch/ompl:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    libfcl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries with pip
RUN pip3 install --no-cache-dir trimesh pyrender fcl
