# FastOCR — DeepDoc + VietOCR

Pipeline OCR tiếng Việt cho tài liệu PDF (giấy tờ đất, giấy chứng nhận...): nhận PDF, trả về **JSON có cấu trúc** + **Markdown** (giữ định dạng bảng, tiêu đề, danh sách). Chạy được cả **CPU** lẫn **GPU** trên cùng 1 codebase.

## 1. Pipeline hoạt động thế nào

Với mỗi PDF, mỗi trang được xử lý qua các model sau:

| Bước | Model | Vai trò |
|---|---|---|
| Layout detection | PP-DocLayout_plus-L (PaddleOCR) | Tìm vùng text/bảng/hình/tiêu đề... trên trang |
| Text detection | DeepDoc `det.onnx` | Tìm bbox từng dòng chữ |
| Text recognition | VietOCR `vgg_seq2seq` | Đọc nội dung chữ tiếng Việt |
| Table structure | DeepDoc `tsr.onnx` | Dựng lại bảng thành Markdown |

Sau đó pipeline map các box chữ vào đúng khối layout, sắp thứ tự đọc (trái→phải, có xử lý trang ghép đôi), rồi build ra `output.json` + `output.md`.

Chi tiết code nằm ở `module/pipeline/document_pipeline.py` (orchestration), `module/layout/`, `module/ocr/`, `module/tsr/`.

## 2. Cấu trúc thư mục

```
module/           # Toàn bộ logic pipeline (layout, ocr, tsr, mapping, reading-order...)
server/           # FastAPI app + giao diện web đơn giản (server/static/index.html)
conf/             # Cấu hình pipeline (pipeline_conf.yaml + local.pipeline_conf.yaml override riêng máy)
vietocr/          # Code VietOCR vendor riêng (tool/model/config) + thư mục weight/
onnx/             # Model .onnx (det.onnx, tsr.onnx...) -- quản lý qua Git LFS
utils/            # read_config(), get_project_base_directory() -- tối giản, chỉ giữ phần thật sự dùng
Dockerfile / docker-compose.yml          # Build & chạy bản CPU
Dockerfile.gpu / docker-compose.gpu.yml  # Build & chạy bản GPU (Docker)
requirements.txt / requirements-gpu.txt  # Dependency tương ứng CPU/GPU
vastai_gpu_deploy_SUCCESS.txt            # Quy trình chạy GPU trực tiếp (không Docker) đã verify thành công
```

## 3. Cấu hình trước khi chạy (bắt buộc, cả CPU lẫn GPU)

Tạo `conf/local.pipeline_conf.yaml` (không commit vào git, riêng theo máy) trỏ đúng weight VietOCR đang dùng:

```yaml
ocr:
  vietocr_weight_path: "vietocr/weight/<ten-file-weight>.pth"
  vietocr_base_config: "vgg_seq2seq"
```

Các config khác (threshold, DPI, crop viền trắng, debug ảnh...) nằm ở `conf/pipeline_conf.yaml`, có thể override tương tự qua `local.pipeline_conf.yaml` mà không cần sửa file chung.

---

## 4. Chạy trên CPU (Docker)

Yêu cầu: đã cài Docker Desktop.

```bash
git clone https://github.com/daothihuyen64/deepdoc_vietocr_dh.git
cd deepdoc_vietocr_dh
# tạo conf/local.pipeline_conf.yaml như mục 3

docker compose up --build
```

Mở `http://localhost:8000/` — giao diện FastOCR: kéo thả PDF, xem kết quả JSON/Markdown trực tiếp.

Sửa `conf/local.pipeline_conf.yaml` sau này (đổi weight...) → chỉ cần `docker compose restart`, không cần build lại (file này được mount live qua `docker-compose.yml`). `vietocr/weight/` cũng được mount live tương tự.

---

## 5. Chạy trên GPU

Model trong pipeline khá nhẹ (VietOCR ~90MB, det/tsr onnx vài MB) — **không cần GPU cao cấp**, 8GB VRAM (vd RTX 3060/3060 Ti) là dư dùng. Yêu cầu driver NVIDIA hỗ trợ **CUDA ≥ 12.6**.

Có 2 cách — khuyến nghị dùng cách A.

### Cách A — Cài trực tiếp bằng pip (khuyến nghị)

Xem đầy đủ, chi tiết từng bước tại **[`vastai_gpu_deploy_SUCCESS.txt`](vastai_gpu_deploy_SUCCESS.txt)**. Tóm tắt:

```bash
# 1. Trên máy GPU (vd thuê Vast.ai, dùng image có sẵn CUDA/Python như vastai/pytorch)
apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 fonts-dejavu-core git

# 2. Clone (bỏ qua Git LFS lúc clone để tránh lỗi hết quota, xem bước 4)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/daothihuyen64/deepdoc_vietocr_dh.git
cd deepdoc_vietocr_dh

# 3. Cài package -- THỨ TỰ QUAN TRỌNG: paddle trước, torch SAU CÙNG không ghim version
#    (paddlepaddle-gpu tự ghim cứng version nvidia-nccl-cu12/cudnn-cu12; cài torch
#    trước sẽ bị paddle đè version làm vỡ torch)
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install -r requirements-gpu.txt
pip install --force-reinstall torch torchvision

# 4. Copy riêng onnx/ + weight + config (không qua Git LFS, scp thẳng từ máy có sẵn)
#    scp -P <PORT> -r onnx/ conf/local.pipeline_conf.yaml vietocr/weight/*.pth root@<HOST>:/path/to/repo/...

# 5. Kiểm tra cả 2 framework trước khi chạy
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "from paddleocr import LayoutDetection; LayoutDetection(model_name='PP-DocLayout_plus-L')"

# 6. Chạy
nohup uvicorn server.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

Model tự động detect GPU (`torch.cuda.is_available()`) — không cần chỉnh gì thêm trong `conf/`, kể cả layout PP-DocLayout cũng tự chuyển `gpu:0` nếu có GPU.

### Cách B — Docker

```bash
docker compose -f docker-compose.gpu.yml up --build
```

`Dockerfile.gpu` build từ base `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04`, cùng ghim version như cách A. Cần Docker daemon hỗ trợ GPU passthrough cho container lồng bên trong instance thuê (không phải nhà cung cấp/loại image nào cũng hỗ trợ tốt — xem thêm ghi chú trong `docker-compose.gpu.yml`).

---

## 6. Gọi API

```bash
curl -X POST http://<host>:8000/api/v1/ocr/pdf -F "file=@document.pdf"
```

Response:
```json
{
  "file": "document.pdf",
  "json": { "file": "...", "pages": [ { "page": 1, "blocks": [...] } ] },
  "markdown": "# ...\n\n..."
}
```
