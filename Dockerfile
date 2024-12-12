# Build stage
FROM python:3.12.7-slim-bullseye as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only required TensorFlow packages
COPY requirements.txt .
RUN pip install --no-cache-dir \
    tensorflow-cpu==2.16.1 \
    keras==3.4.1 \
    Flask==3.1.0 \
    cvzone==1.6.1 \
    dm-tree==0.1.8 \
    pyglet==2.0.18 \
    mediapipe \
    && find /opt/venv -type d -name "tests" -exec rm -rf {} + \
    && find /opt/venv -type d -name "examples" -exec rm -rf {} + \
    && find /opt/venv -name "*.pyc" -delete \
    && find /opt/venv -name "*.pyo" -delete \
    && find /opt/venv -name "*.pyd" -delete

# Runtime stage
FROM python:3.12.7-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install only absolute minimum runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy only the virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Copy only necessary application files
COPY server.py StayAwake.keras ./
# Add any other necessary application files here

EXPOSE 5000

CMD ["python", "server.py"]