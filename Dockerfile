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
# device='cpu' in code anyway). Installing the EXACT same pinned versions
# that requirements.txt itself lists means the later `-r requirements.txt`
# pass sees them already satisfied and leaves them alone -- if these two
# ever drift apart (e.g. only bumping requirements.txt), pip must resolve
# torch/torchvision fresh from the default (CUDA-bundled) PyPI index instead
# of the CPU one, which pulls in gigabytes of unneeded nvidia-*-cu* wheels
# and can even fail outright once another pinned dependency (e.g.
# surya-ocr's `torch>=2.7.0,<3.0`) constrains the resolution differently
# than whatever the CPU index's unpinned "latest" would have picked.
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# surya-ocr==0.17.1 declares `pillow<11.0.0`, which directly conflicts with
# mineru[pipeline]'s (unconditional, every version) `pillow>=11.0.0` -- no
# overlapping version exists, so it can't go in requirements.txt's normal
# resolved install. Install it with --no-deps (skip pip re-checking/
# reinstalling its own pillow/torch/transformers pins against what's already
# installed above) and separately install the rest of ITS OWN direct deps
# that aren't already covered by the main requirements.txt install --
# excluding pillow (keep mineru's >=11.0.0), opencv-python-headless (we
# already have opencv-python), torch, and transformers (mineru's own pin
# already satisfies surya's floor). This assumes surya's code works fine
# against Pillow 11 despite its conservative upper pin -- only its
# TableRecPredictor is used in this project (basic PIL Image open/convert/
# resize), so this should be safe, but re-verify against real table crops
# after building.
RUN pip install --no-cache-dir --no-deps surya-ocr==0.17.1 \
    && pip install --no-cache-dir \
        "click<9.0.0,>=8.1.8" \
        "filetype<2.0.0,>=1.2.0" \
        "platformdirs<5.0.0,>=4.3.6" \
        "pydantic<3.0.0,>=2.5.3" \
        "pydantic-settings<3.0.0,>=2.1.0" \
        "pypdfium2==4.30.0" \
        "python-dotenv<2.0.0,>=1.0.0"

COPY . .

# Pre-download PP-DocLayout weights, and RapidAI's table_cls QAnything
# classifier weight, at build time so the container's first request doesn't
# stall on a cold download (table_cls fetches its .onnx from modelscope.cn
# on first use). model_type="q" must match module/table/mineru_processor.py.
RUN python -c "from paddleocr import LayoutDetection; LayoutDetection(model_name='PP-DocLayout_plus-L')" \
    && python -c "from table_cls import TableCls; TableCls(model_type='q')"

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
