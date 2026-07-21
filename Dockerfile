FROM python:3.11-slim

# poppler-utils: required by pdf2image to rasterize PDFs.
# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1: required by opencv-python.
# libgomp1: required by onnxruntime.
# fonts-dejavu-core: Vietnamese-capable TTF used to label debug bbox overlays.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install the CPU-only torch/torchvision wheels instead of the default PyPI
# build (which bundles CUDA/cuDNN we never use -- TextRecognizer forces
# device='cpu' in code anyway). Installing them first means the later
# `-r requirements.txt` pass sees them already satisfied and leaves them alone.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download PP-DocLayout weights at build time so the container's first
# request doesn't stall on a cold download.
RUN python -c "from paddleocr import LayoutDetection; LayoutDetection(model_name='PP-DocLayout_plus-L')"

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
