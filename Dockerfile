FROM docker.io/pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

WORKDIR /workspace/OmniLearned

COPY pyproject.toml .
COPY README.md .
COPY src/ src/

# eventually, copy over saved checkpoints, etc

RUN pip3 install --no-cache-dir --upgrade pip --break-system-packages && \
    pip3 install --no-cache-dir . --break-system-packages