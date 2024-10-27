# Use the existing OMPL Docker image as the base
FROM deividtesch/ompl:latest

# Install system dependencies including OpenGL
RUN apt-get update && apt-get install -y \
    python3-pip \
    libfcl-dev \
    python3-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries
RUN pip3 install --no-cache-dir "numpy<2.0.0" trimesh pyrender open3d rtree python-fcl
