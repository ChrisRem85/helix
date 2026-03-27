# Dockerfile (GPU, Python 3.10 via PyTorch runtime)
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only requirements first to maximize cache reuse
COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install --upgrade pip \
 && pip install -r /workspace/requirements.txt

WORKDIR /workspace
CMD ["python", "match_pairs.py"]
