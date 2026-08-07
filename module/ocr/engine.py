#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import copy
import threading
import time
import os

from huggingface_hub import snapshot_download

from utils.file_utils import get_project_base_directory
from utils.settings import PARALLEL_DEVICES
from ..operators import *  # noqa: F403
from .. import operators
import math
import numpy as np
import cv2
import onnxruntime as ort
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torch

from ..postprocess import build_post_process

from fastocr.tool.predictor import Predictor
from fastocr.tool.config import Cfg

loaded_models = {}

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(
        op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = getattr(operators, op_name)(**param)
        ops.append(op)
    return ops


def load_model(model_dir, nm, device_id: int | None = None):
    model_file_path = os.path.join(model_dir, nm + ".onnx")
    model_cached_tag = model_file_path + str(device_id) if device_id is not None else model_file_path

    global loaded_models
    loaded_model = loaded_models.get(model_cached_tag)
    if loaded_model:
        logging.info(f"[device] load_model {model_file_path} reuses cached model")
        return loaded_model

    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))

    def cuda_is_available():
        try:
            import torch
            dev = device_id if device_id is not None else 0
            if torch.cuda.is_available() and torch.cuda.device_count() > dev:
                return True
        except Exception:
            return False
        return False

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2

    # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
    # Shrink GPU memory after execution
    run_options = ort.RunOptions()
    if cuda_is_available():
        resolved_device_id = device_id if device_id is not None else 0
        cuda_provider_options = {
            "device_id": resolved_device_id, # Use specific GPU
            "gpu_mem_limit": 2048 * 1024 * 1024, # Limit gpu memory
            "arena_extend_strategy": "kNextPowerOfTwo",  # gpu memory allocation strategy
        }
        sess = ort.InferenceSession(
            model_file_path,
            sess_options=options,
            providers=['CUDAExecutionProvider'],
            provider_options=[cuda_provider_options]
            )
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:" + str(resolved_device_id))
        logging.info(f"[device] load_model {model_file_path} uses GPU (cuda:{resolved_device_id})")
    else:
        sess = ort.InferenceSession(
            model_file_path,
            sess_options=options,
            providers=['CPUExecutionProvider'])
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        logging.info(f"[device] load_model {model_file_path} uses CPU")
    loaded_model = (sess, run_options)
    loaded_models[model_cached_tag] = loaded_model
    return loaded_model

class TextRecognizer:
    def __init__(self, model_dir=None, device_id: int | None = None,
                 weight_path: str = None, base_config_name: str = None,
                 max_batch_size: int = 64):
        assert base_config_name is not None, (
            "base_config_name bắt buộc phải truyền vào ('vgg_transformer' hoặc 'vgg_seq2seq')")
        assert weight_path is not None and os.path.exists(weight_path), (
            f"weight_path không hợp lệ hoặc file không tồn tại: {weight_path!r}")

        config = Cfg.load_config_from_name(base_config_name)
        config['weights'] = weight_path
        config['cnn']['pretrained'] = False
        import torch
        dev = device_id if device_id is not None else 0
        config['device'] = f'cuda:{dev}' if torch.cuda.is_available() else 'cpu'
        logging.info(f"[device] FastOCR ({base_config_name}): device={config['device']}")
        self.detector = Predictor(config)
        # Caps how many text-line crops __call__() stacks into one
        # predict_batch() forward pass at a time -- a dense page can have
        # 100+ detected boxes, which would otherwise all get batched
        # unbounded in one call and risk a GPU OOM.
        self._max_batch_size = max_batch_size

    def __call__(self, img_list):
        if not img_list:
            return [], 0.0
        pil_imgs = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if isinstance(img, np.ndarray) else img
            for img in img_list
        ]
        # predict_batch() recognizes every crop in ONE encoder+decoder
        # forward pass instead of one call per crop -- score is still
        # discarded (hardcoded 1.0) here, matching this method's existing
        # behavior/contract; callers that need the real confidence use
        # self.detector.predict_batch()/predict() directly (see
        # MinerUTableProcessor._recognize_from_boxes).
        results = []
        for start in range(0, len(pil_imgs), self._max_batch_size):
            chunk = pil_imgs[start : start + self._max_batch_size]
            results.extend((text, 1.0) for text, _prob in self.detector.predict_batch(chunk))
        return results, 0.0


class TextDetector:
    def __init__(self, model_dir, device_id: int | None = None):
        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': 960,
                'limit_type': "max",
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {"name": "DBPostProcess", "thresh": 0.3, "box_thresh": 0.5, "max_candidates": 1000,
                              "unclip_ratio": 1.5, "use_dilation": False, "score_mode": "fast", "box_type": "quad"}

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.run_options = load_model(model_dir, 'det', device_id)
        self.input_tensor = self.predictor.get_inputs()[0]

        img_h, img_w = self.input_tensor.shape[2:]
        if isinstance(img_h, str) or isinstance(img_w, str):
            pass
        elif img_h is not None and img_w is not None and img_h > 0 and img_w > 0:
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'image_shape': [img_h, img_w]
                }
            }
        self.preprocess_op = create_operators(pre_process_list)

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if isinstance(box, list):
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {'image': img}

        st = time.time()
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        input_dict = {}
        input_dict[self.input_tensor.name] = img
        for i in range(100000):
            try:
                outputs = self.predictor.run(None, input_dict, self.run_options)
                break
            except Exception as e:
                if i >= 3:
                    raise e
                time.sleep(5)

        post_result = self.postprocess_op({"maps": outputs[0]}, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, ori_im.shape)

        return dt_boxes, time.time() - st


def _build_mineru_text_detector(device: str):
    """Builds MinerU's own PP-OCRv6 text detector (same DB/DBNet algorithm family
    as the onnx TextDetector above, different trained checkpoint) -- lazy import so
    installs that never select ocr.det_backend: mineru don't need `mineru`."""
    from mineru.model.utils.tools.infer import pytorchocr_utility as utility
    from mineru.model.utils.tools.infer.predict_det import TextDetector as MineruTextDetector
    from mineru.utils.enum_class import ModelPath
    from mineru.utils.models_download_utils import auto_download_and_get_model_root_path

    det_filename = "ch_PP-OCRv6_small_det_infer.safetensors"
    det_rel = f"{ModelPath.pytorch_paddle}/{det_filename}"
    det_path = os.path.join(auto_download_and_get_model_root_path(det_rel), det_rel)

    parser = utility.init_args()
    args = parser.parse_args([])
    args.device = device
    args.det_model_path = det_path

    return MineruTextDetector(args)


class OCR:
    def __init__(self, model_dir=None, fastocr_weight_path: str = None, fastocr_base_config: str = None,
                 crop_pad_px_h: float = 3.0, crop_pad_px_w: float = 0.0, det_backend: str = "onnx",
                 det_max_batch_size: int = 8, fastocr_max_batch_size: int = 64):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """

        if not model_dir:
            model_dir = os.path.join(get_project_base_directory(), "onnx")

        devices = list(range(PARALLEL_DEVICES)) if PARALLEL_DEVICES else [0]

        if det_backend == "mineru":
            devices_resolved = [f"cuda:{d}" if torch.cuda.is_available() else "cpu" for d in devices]
            self.text_detector = [
                _build_mineru_text_detector(dev) for dev in devices_resolved
            ]
            logging.info(f"[device] OCR detector (mineru PP-OCRv6): device={devices_resolved}")
        elif det_backend == "onnx":
            # Only the DeepDoc ONNX detector depends on `model_dir` containing the
            # right files -- fall back to downloading them from HuggingFace if
            # missing. FastOCR weight resolution below is independent of this and
            # must fail loudly (not get masked by this fallback) if misconfigured.
            # (load_model() itself already logs "uses GPU"/"uses CPU" per device below.)
            try:
                self.text_detector = [TextDetector(model_dir, d) for d in devices]
            except Exception:
                model_dir = snapshot_download(repo_id="InfiniFlow/deepdoc",
                                              local_dir=os.path.join(get_project_base_directory(), "onnx"),
                                              local_dir_use_symlinks=False)
                self.text_detector = [TextDetector(model_dir, d) for d in devices]
        else:
            raise ValueError(f"Unknown ocr det_backend: {det_backend!r}")

        self.text_recognizer = [TextRecognizer(
            model_dir, d,
            weight_path=fastocr_weight_path,
            base_config_name=fastocr_base_config,
            max_batch_size=fastocr_max_batch_size) for d in devices]

        self.drop_score = 0.5
        self.crop_image_res_index = 0
        # Expands each detected text quad outward along its own local height
        # axis (FIXED px, not a ratio of the box's own height -- a ratio
        # over-pads tall boxes) before cropping for recognition -- Vietnamese
        # diacritics sit just above/below the detector's tight box and were
        # getting clipped without this.
        self.crop_pad_px_h = crop_pad_px_h
        self.crop_pad_px_w = crop_pad_px_w
        # Caps how many images detect_raw_batch()/detect_sorted_batch() ask
        # the detector to stack into one forward pass at a time (only takes
        # effect for backends whose batch_predict() honors it, i.e. mineru's
        # own detector -- see detect_raw_batch) -- keeps a document with
        # many pages/table crops from OOMing GPU memory just because every
        # image got handed to one batched call.
        self._det_max_batch_size = det_max_batch_size
        # self.text_detector[i] is shared across every request the server
        # handles concurrently -- confirmed via real crashes that shared
        # Paddle/PyTorch model instances are NOT necessarily safe to call
        # from multiple threads at once (layout's Paddle predictor, Surya's
        # decoder), so each detector's own forward pass is serialized here
        # too. One lock PER DEVICE, not one global lock -- if PARALLEL_DEVICES
        # spans multiple independent GPUs, requests hitting DIFFERENT devices
        # should still run fully in parallel.
        self._detect_locks = [threading.Lock() for _ in self.text_detector]

    @staticmethod
    def _pad_quad(points, pad_px_h, pad_px_w):
        """Expands a 4-point quad (clockwise: top-left, top-right,
        bottom-right, bottom-left) outward along its OWN local width/height
        axes, by a FIXED number of pixels each side -- works for rotated
        boxes, unlike padding in raw x/y."""
        points = np.array(points, dtype=np.float32)
        if pad_px_h <= 0 and pad_px_w <= 0:
            return points
        w_vec = points[1] - points[0]
        h_vec = points[3] - points[0]
        w_len = np.linalg.norm(w_vec)
        h_len = np.linalg.norm(h_vec)
        if w_len == 0 or h_len == 0:
            return points
        w_unit = w_vec / w_len
        h_unit = h_vec / h_len
        padded = points.copy()
        padded[0] = points[0] - h_unit * pad_px_h - w_unit * pad_px_w
        padded[1] = points[1] - h_unit * pad_px_h + w_unit * pad_px_w
        padded[2] = points[2] + h_unit * pad_px_h + w_unit * pad_px_w
        padded[3] = points[3] + h_unit * pad_px_h - w_unit * pad_px_w
        return padded

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        assert len(points) == 4, "shape of points must be 4*2"
        orig_points = np.array(points, dtype=np.float32)
        # Rotate-90 decision below must reflect what the DETECTOR actually
        # found (vertical vs horizontal text), not the post-padding shape --
        # padding a narrow single-character box (e.g. a 1-digit table cell)
        # can push height/width past the 1.5 ratio on its own and cause a
        # wrongly-rotated crop even though the original box was horizontal.
        orig_width = max(
            np.linalg.norm(orig_points[0] - orig_points[1]),
            np.linalg.norm(orig_points[2] - orig_points[3]))
        orig_height = max(
            np.linalg.norm(orig_points[0] - orig_points[3]),
            np.linalg.norm(orig_points[1] - orig_points[2]))

        points = self._pad_quad(points, self.crop_pad_px_h, self.crop_pad_px_w)
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        if orig_height * 1.0 / max(orig_width, 1e-6) > 2:
            dst_img = np.rot90(dst_img)
        return dst_img

    def sorted_boxes(self, dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                        (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        return _boxes

    def detect_raw(self, img_bgr: np.ndarray, device_id: int | None = None) -> np.ndarray | None:
        """Runs the shared text detector on ONE image, normalized to a
        single result shape callers can rely on: None when nothing was
        detected (the detector itself inconsistently returns either None
        or an empty array), or the raw (unsorted) dt_boxes otherwise.

        This is the ONE place `self.text_detector[device_id](...)` gets
        called from -- every call site in the pipeline (deskew_page,
        page-orientation scoring, run_ocr_page, table OCR) used to
        reimplement this same detect+None-check pattern independently;
        centralizing it here means a future change to HOW detection runs
        (e.g. batching multiple images in one call) only has to happen in
        one place.
        """
        if device_id is None:
            device_id = 0
        with self._detect_locks[device_id]:
            dt_boxes, _elapse = self.text_detector[device_id](img_bgr)
        if dt_boxes is None or len(dt_boxes) == 0:
            return None
        return dt_boxes

    def detect_sorted(self, img_bgr: np.ndarray, device_id: int | None = None) -> np.ndarray | None:
        """detect_raw() + sorted_boxes() -- the reading-order-sorted variant
        every caller except deskew_page() (which only needs box shapes for
        its tilt-angle estimate, not reading order) actually wants."""
        dt_boxes = self.detect_raw(img_bgr, device_id)
        if dt_boxes is None:
            return None
        return self.sorted_boxes(dt_boxes)

    def detect_raw_batch(
        self,
        img_list: list[np.ndarray],
        device_id: int | None = None,
        max_batch_size: int | None = None,
    ) -> list[np.ndarray | None]:
        """Detects text boxes for MULTIPLE images in one call.

        Uses the detector's own `.batch_predict()` when available
        (MinerU's own PP-OCRv6 detector, `ocr.det_backend: mineru`) -- a
        real batched forward pass that ALREADY groups images by their own
        preprocessed shape internally
        (mineru/model/utils/tools/infer/predict_det.py::batch_predict),
        so mixed-size images never get padded against each other -- no
        shape-grouping needed on our side here, unlike the layout backend
        (PaddleX's LayoutDetection does NOT do this itself, see
        PPDocLayoutBackend.detect_batch).

        Falls back to one call per image (via detect_raw) for detector
        backends that don't expose `batch_predict` (the onnx det.onnx
        detector) -- same result, just no batching speedup.

        `max_batch_size`, when given, overrides the instance-level
        `ocr.max_batch_size` (self._det_max_batch_size) for THIS call only
        -- lets a caller batching much SMALLER images (e.g. table crops,
        see MinerUTableProcessor.process_batch) use its own cap instead of
        whatever this OCR instance's page-level default happens to be,
        since a sane batch size for full pages and for small crops isn't
        the same number.

        Returns one result per input image, in the SAME ORDER as
        `img_list` -- either the raw dt_boxes array, or None if nothing
        was detected.
        """
        if device_id is None:
            device_id = 0
        if not img_list:
            return []

        effective_max_batch_size = max_batch_size if max_batch_size is not None else self._det_max_batch_size

        detector = self.text_detector[device_id]
        if not hasattr(detector, "batch_predict"):
            return [self.detect_raw(img, device_id) for img in img_list]

        with self._detect_locks[device_id]:
            batch_results = detector.batch_predict(img_list, max_batch_size=effective_max_batch_size)
        if len(batch_results) != len(img_list):
            raise RuntimeError(
                f"Detector batch_predict returned {len(batch_results)} results "
                f"for {len(img_list)} input images -- refusing to guess how they line up."
            )
        return [
            dt_boxes if dt_boxes is not None and len(dt_boxes) > 0 else None
            for dt_boxes, _elapse in batch_results
        ]

    def detect_sorted_batch(
        self,
        img_list: list[np.ndarray],
        device_id: int | None = None,
        max_batch_size: int | None = None,
    ) -> list[np.ndarray | None]:
        """detect_raw_batch() + sorted_boxes() per image. `max_batch_size`
        -- see detect_raw_batch()."""
        raw_results = self.detect_raw_batch(img_list, device_id, max_batch_size)
        return [
            self.sorted_boxes(dt_boxes) if dt_boxes is not None else None
            for dt_boxes in raw_results
        ]

    def detect(self, img, device_id: int | None = None):
        if device_id is None:
            device_id = 0

        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            return None, None, time_dict

        start = time.time()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        return zip(self.sorted_boxes(dt_boxes), [
                   ("", 0) for _ in range(len(dt_boxes))])

    def recognize(self, ori_im, box, device_id: int | None = None):
        if device_id is None:
            device_id = 0

        img_crop = self.get_rotate_crop_image(ori_im, box)

        rec_res, elapse = self.text_recognizer[device_id]([img_crop])
        text, score = rec_res[0]
        if score < self.drop_score:
            return ""
        return text

    def recognize_batch(self, img_list, device_id: int | None = None):
        if device_id is None:
            device_id = 0
        rec_res, elapse = self.text_recognizer[device_id](img_list)
        texts = []
        for i in range(len(rec_res)):
            text, score = rec_res[i]
            if score < self.drop_score:
                text = ""
            texts.append(text)
        return texts

    def __call__(self, img, device_id = 0, cls=True, crop_debug_dir=None):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        if device_id is None:
            device_id = 0

        if img is None:
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector[device_id](img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        img_crop_list = []

        dt_boxes = self.sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # Dumps the EXACT crops handed to the recognizer below (post
        # TSR/layout detect, post crop padding) -- for debugging what
        # FastOCR actually sees per box, not a reconstruction of it.
        if crop_debug_dir:
            os.makedirs(crop_debug_dir, exist_ok=True)
            for bno, img_crop in enumerate(img_crop_list):
                cv2.imwrite(os.path.join(crop_debug_dir, f"{bno}.png"), img_crop)

        rec_res, elapse = self.text_recognizer[device_id](img_crop_list)

        time_dict['rec'] = elapse

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start

        # for bno in range(len(img_crop_list)):
        #    print(f"{bno}, {rec_res[bno]}")

        return list(zip([a.tolist() for a in filter_boxes], filter_rec_res))
