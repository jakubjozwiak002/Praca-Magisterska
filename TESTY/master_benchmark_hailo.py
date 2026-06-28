#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
master_benchmark_hailo.py

Benchmark dla Raspberry Pi 5 + Hailo AI HAT+ / Hailo-8.

Zakładana struktura katalogów:
00_TESTY/
├── master_benchmark_hailo.py
├── test/
│   ├── images/
│   │   └── dataset_config.json
│   └── labels/
└── hef/
    ├── yolov8_640_hailo8_valcalib.hef
    ├── yolov8_320_hailo8_valcalib.hef
    ├── yolov11_640_hailo8_valcalib.hef
    └── yolov11_320_hailo8_valcalib.hef

Najważniejsze zmienne środowiskowe:
export BENCHMARK_DATASET_DIR="/home/kubat/Downloads/00_TESTY/test"
export BENCHMARK_HAILO_DIR="/home/kubat/Downloads/00_TESTY/hef"
export BENCHMARK_RESULTS_CSV="/home/kubat/Downloads/00_TESTY/benchmark_results_rpi5_hailo.csv"
export BENCHMARK_PLATFORM="RPI5_HAILO8"
export BENCHMARK_HAILO_CONF="0.4"
export BENCHMARK_RUN_CLASSICAL="1"
export BENCHMARK_OPENCV_THREADS="4"
export BENCHMARK_LIMIT_IMAGES="0"

Uwaga metodologiczna:
- Hailo przyspiesza wyłącznie modele HEF uruchamiane przez HailoRT.
- Metody Canny/PP/TopHat pozostają metodami CPU/OpenCV.
- Umieszczono je w tym skrypcie, aby zmierzyć je w tym samym środowisku
  sprzętowo-termicznym co testy Hailo, nie dlatego, że Hailo je akceleruje.
"""

from __future__ import annotations

import os
from pathlib import Path
from contextlib import ExitStack
import cv2
import json
import csv
import time
import glob
import psutil
import itertools
import numpy as np
import gc
import unicodedata
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from hailo_platform import (
        HEF,
        ConfigureParams,
        FormatType,
        HailoStreamInterface,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        VDevice,
    )
    HAILO_AVAILABLE = True
except Exception as exc:
    HAILO_AVAILABLE = False
    HAILO_IMPORT_ERROR = exc


# ==============================================================================
# KONFIGURACJA ZBIORU I ŚCIEŻEK
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = Path(os.environ.get("BENCHMARK_DATASET_DIR", BASE_DIR / "test")).expanduser().resolve()
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

CONFIG_FILE = Path(os.environ.get("BENCHMARK_CONFIG_FILE", IMAGES_DIR / "dataset_config.json")).expanduser().resolve()
RESULTS_CSV = Path(os.environ.get("BENCHMARK_RESULTS_CSV", BASE_DIR / "benchmark_results_rpi5_hailo.csv")).expanduser().resolve()

PLATFORM_NAME = os.environ.get("BENCHMARK_PLATFORM", "RPI5_HAILO8")

HAILO_DIR = Path(os.environ.get("BENCHMARK_HAILO_DIR", BASE_DIR / "hef")).expanduser().resolve()
HAILO_CONF = float(os.environ.get("BENCHMARK_HAILO_CONF", "0.4"))
HAILO_BOX_ORDER = os.environ.get("BENCHMARK_HAILO_BOX_ORDER", "yxyx").strip().lower()

RUN_CLASSICAL = os.environ.get("BENCHMARK_RUN_CLASSICAL", "1").strip() not in {"0", "false", "False", "no", "NO"}
LIMIT_IMAGES = int(os.environ.get("BENCHMARK_LIMIT_IMAGES", "0"))

OPENCV_THREADS = int(os.environ.get("BENCHMARK_OPENCV_THREADS", "0"))
if OPENCV_THREADS > 0:
    cv2.setNumThreads(OPENCV_THREADS)

if HAILO_BOX_ORDER not in {"yxyx", "xyxy"}:
    raise ValueError("BENCHMARK_HAILO_BOX_ORDER musi mieć wartość 'yxyx' albo 'xyxy'.")

CLASS_ID_TO_COLOR = {
    0: "Zielony",
    1: "Zolty",
    2: "Czerwony",
    3: "Bialy",
}

BRIGHTNESS_THRESHOLD = 150
HI_WARNING_THRESHOLD = 0.80
HI_ERROR_THRESHOLD = 0.40

RESOLUTIONS = [(640, 480), (320, 240)]
PHOTOMETRY_STATES = [False, True]
PRESETS = ["Dzien", "Noc"]

ALGORITHMS = ["HAILO_YOLO"]
if RUN_CLASSICAL:
    ALGORITHMS += ["Canny", "PP", "TopHat"]

HAILO_MODELS = [
    {
        "name": "yolov8_640_hailo8_valcalib",
        "path": HAILO_DIR / "yolov8_640_hailo8_valcalib.hef",
        "resolution": (640, 480),
    },
    {
        "name": "yolov8_320_hailo8_valcalib",
        "path": HAILO_DIR / "yolov8_320_hailo8_valcalib.hef",
        "resolution": (320, 240),
    },
    {
        "name": "yolov11_640_hailo8_valcalib",
        "path": HAILO_DIR / "yolov11_640_hailo8_valcalib.hef",
        "resolution": (640, 480),
    },
    {
        "name": "yolov11_320_hailo8_valcalib",
        "path": HAILO_DIR / "yolov11_320_hailo8_valcalib.hef",
        "resolution": (320, 240),
    },
]


# ==============================================================================
# FUNKCJE MATEMATYCZNE, GT I KOLORY
# ==============================================================================
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0.0


def normalize_color_name(value):
    """Ujednolica nazwy klas kolorów pochodzące z YOLO, GT i klasyfikatora HSV."""
    if value is None:
        return "Inny"

    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ł", "l")

    aliases = {
        "bialy": "Bialy",
        "white": "Bialy",
        "czerwony": "Czerwony",
        "red": "Czerwony",
        "zolty": "Zolty",
        "zółty": "Zolty",
        "yellow": "Zolty",
        "zielony": "Zielony",
        "green": "Zielony",
    }
    return aliases.get(text, "Inny")


def get_lamp_color(frame_bgr, cx, cy, inner_r=3, outer_r=12):
    """Klasyfikacja koloru lampy na podstawie głosowania pikseli w pierścieniu HSV."""
    h, w = frame_bgr.shape[:2]

    x1 = max(0, cx - outer_r)
    y1 = max(0, cy - outer_r)
    x2 = min(w, cx + outer_r)
    y2 = min(h, cy + outer_r)

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return "Inny"

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    local_cx = cx - x1
    local_cy = cy - y1

    mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (local_cx, local_cy), outer_r, 255, -1)
    cv2.circle(mask, (local_cx, local_cy), inner_r, 0, -1)

    s_pixels = roi_hsv[:, :, 1][mask == 255]
    h_pixels = roi_hsv[:, :, 0][mask == 255]

    if len(s_pixels) == 0:
        return "Inny"

    s_thresh = 50
    white_votes = np.sum(s_pixels < s_thresh)

    color_mask = s_pixels >= s_thresh
    valid_h = h_pixels[color_mask]

    red_votes = np.sum((valid_h <= 8) | (valid_h >= 160))
    yellow_votes = np.sum((valid_h > 8) & (valid_h <= 30))
    green_votes = np.sum((valid_h > 30) & (valid_h <= 85))

    votes = {
        "Bialy": white_votes,
        "Czerwony": red_votes,
        "Zolty": yellow_votes,
        "Zielony": green_votes,
    }

    best_color = max(votes, key=votes.get)
    max_votes = votes[best_color]

    if max_votes < len(s_pixels) * 0.2:
        return "Inny"

    return best_color


def load_ground_truth(txt_path, target_w, target_h):
    """Wczytuje etykiety YOLO jako ramki oraz klasy kolorów."""
    objects = []
    if not Path(txt_path).exists():
        return objects

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                x1 = int((cx - w / 2) * target_w)
                y1 = int((cy - h / 2) * target_h)
                x2 = int((cx + w / 2) * target_w)
                y2 = int((cy + h / 2) * target_h)

                objects.append({
                    "box": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "color": normalize_color_name(CLASS_ID_TO_COLOR.get(class_id, "Inny")),
                })
    return objects


def filter_ground_truths_by_roi(gt_objects, mask):
    """Odrzuca GT, których środek znajduje się poza maską ROI."""
    filtered_gt = []
    for gt in gt_objects:
        box = gt["box"]
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0] and mask[cy, cx] == 255:
            filtered_gt.append(gt)
    return filtered_gt


def evaluate_predictions(preds, ground_truths, iou_threshold=0.45):
    """
    Dopasowuje predykcje do GT globalnie po największym IoU.
    Zwraca również statystyki klasyfikacji koloru dla poprawnie dopasowanych detekcji.
    """
    pairs = []

    for pred_idx, pred in enumerate(preds):
        p_box = pred["box"]
        for gt_idx, gt in enumerate(ground_truths):
            gt_box = gt["box"]
            iou = calculate_iou(p_box, gt_box)
            if iou >= iou_threshold:
                pairs.append((iou, pred_idx, gt_idx))

    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_preds = set()
    matched_gts = set()
    matched_ious = []

    class_evaluable = 0
    class_correct = 0
    class_unknown = 0

    for iou, pred_idx, gt_idx in pairs:
        if pred_idx not in matched_preds and gt_idx not in matched_gts:
            matched_preds.add(pred_idx)
            matched_gts.add(gt_idx)
            matched_ious.append(iou)

            pred_color = normalize_color_name(preds[pred_idx].get("color", "Inny"))
            gt_color = normalize_color_name(ground_truths[gt_idx].get("color", "Inny"))

            if gt_color != "Inny":
                class_evaluable += 1
                if pred_color == "Inny":
                    class_unknown += 1
                if pred_color == gt_color:
                    class_correct += 1

    TP = len(matched_ious)
    FP = len(preds) - TP
    FN = len(ground_truths) - TP

    return TP, FP, FN, matched_ious, class_evaluable, class_correct, class_unknown


# ==============================================================================
# HAILO: PREPROCESSING I DEKODOWANIE NMS
# ==============================================================================
def normalize_shape(shape: Any) -> Tuple[int, int, int]:
    s = tuple(int(x) for x in shape)
    if len(s) == 3:
        return s[0], s[1], s[2]
    if len(s) == 4:
        return s[-3], s[-2], s[-1]
    raise ValueError(f"Nieobsługiwany kształt wejścia Hailo: {shape}")


def letterbox_bgr_to_rgb(
    frame_bgr: np.ndarray,
    input_h: int,
    input_w: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h0, w0 = frame_bgr.shape[:2]
    scale = min(input_w / w0, input_h / h0)
    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_h, input_w, 3), color, dtype=np.uint8)

    pad_left = (input_w - new_w) // 2
    pad_top = (input_h - new_h) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb, scale, (pad_left, pad_top)


def undo_letterbox_box(
    box: Iterable[float],
    scale: float,
    pad: Tuple[int, int],
    orig_w: int,
    orig_h: int,
    input_w: int,
    input_h: int,
) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]

    # Format znormalizowany 0..1.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0:
        x1 *= input_w
        x2 *= input_w
        y1 *= input_h
        y2 *= input_h

    pad_left, pad_top = pad
    x1 = (x1 - pad_left) / scale
    x2 = (x2 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    y2 = (y2 - pad_top) / scale

    x1 = int(np.clip(round(x1), 0, orig_w - 1))
    x2 = int(np.clip(round(x2), 0, orig_w - 1))
    y1 = int(np.clip(round(y1), 0, orig_h - 1))
    y2 = int(np.clip(round(y2), 0, orig_h - 1))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def try_decode_hailo_nms(
    outputs: Dict[str, Any],
    scale: float,
    pad: Tuple[int, int],
    orig_w: int,
    orig_h: int,
    input_w: int,
    input_h: int,
    conf_thres: float,
    box_order: str = "yxyx",
) -> List[Dict[str, Any]]:
    """
    Dekoder wyjścia Hailo NMS.

    Obsługiwany format zaobserwowany w smoke-testach:
    output shape=(4, 5, 100), gdzie:
    - 4 = klasy,
    - 5 = ymin/xmin/ymax/xmax/conf albo xmin/ymin/xmax/ymax/conf,
    - 100 = maksymalna liczba detekcji na klasę.
    """
    detections: List[Dict[str, Any]] = []

    def normalize_box_order(box: Iterable[float]) -> List[float]:
        vals = [float(v) for v in box]
        if len(vals) < 4:
            return vals
        if box_order == "yxyx":
            y1, x1, y2, x2 = vals[:4]
            return [x1, y1, x2, y2]
        return vals[:4]

    def add_det(cls_id: int, score: float, box: Iterable[float]) -> None:
        try:
            score_f = float(score)
        except Exception:
            return

        if not np.isfinite(score_f) or score_f < conf_thres:
            return

        box_xyxy = normalize_box_order(box)
        if len(box_xyxy) < 4:
            return

        mapped = undo_letterbox_box(box_xyxy, scale, pad, orig_w, orig_h, input_w, input_h)
        if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
            return

        cx = int((mapped[0] + mapped[2]) / 2)
        cy = int((mapped[1] + mapped[3]) / 2)
        detections.append({
            "box": mapped,
            "cx": cx,
            "cy": cy,
            "class_id": int(cls_id),
            "color": normalize_color_name(CLASS_ID_TO_COLOR.get(int(cls_id), "Inny")),
            "conf": score_f,
        })

    def safe_array(obj: Any) -> Optional[np.ndarray]:
        try:
            return np.asarray(obj)
        except Exception:
            return None

    def decode_array(a: np.ndarray, default_cls: Optional[int] = None) -> bool:
        if a is None or a.size == 0:
            return False

        a = np.squeeze(a)

        # Hailo NMS: (classes, 5, max_boxes)
        if a.ndim == 3 and a.shape[1] == 5:
            num_classes, _, max_boxes = a.shape
            for cls_id in range(num_classes):
                for j in range(max_boxes):
                    row = a[cls_id, :, j].astype(float)
                    add_det(cls_id, row[4], row[:4])
            return True

        # Hailo NMS: (classes, max_boxes, 5)
        if a.ndim == 3 and a.shape[2] == 5:
            num_classes, max_boxes, _ = a.shape
            for cls_id in range(num_classes):
                for j in range(max_boxes):
                    row = a[cls_id, j, :].astype(float)
                    add_det(cls_id, row[4], row[:4])
            return True

        # Pojedyncza klasa: (5, max_boxes)
        if default_cls is not None and a.ndim == 2 and a.shape[0] == 5:
            for j in range(a.shape[1]):
                row = a[:, j].astype(float)
                add_det(default_cls, row[4], row[:4])
            return True

        # Pojedyncza klasa: (max_boxes, 5)
        if default_cls is not None and a.ndim == 2 and a.shape[1] == 5:
            for j in range(a.shape[0]):
                row = a[j, :].astype(float)
                add_det(default_cls, row[4], row[:4])
            return True

        # Klasyczny format detekcji: (N, 6+)
        if a.ndim == 1 and a.shape[0] >= 6:
            a = a.reshape(1, -1)

        if a.ndim == 2 and a.shape[1] >= 6:
            recognized_any = False
            for row in a:
                row = row.astype(float)

                # x1,y1,x2,y2,score,class
                score_a = row[4]
                cls_a = int(round(row[5]))
                if 0 <= cls_a <= 1000 and 0 <= score_a <= 1.5:
                    add_det(cls_a, score_a, row[:4])
                    recognized_any = True
                    continue

                # class,score,x1,y1,x2,y2
                cls_b = int(round(row[0]))
                score_b = row[1]
                if 0 <= cls_b <= 1000 and 0 <= score_b <= 1.5:
                    add_det(cls_b, score_b, row[2:6])
                    recognized_any = True
                    continue

            return recognized_any

        return False

    def walk(obj: Any, default_cls: Optional[int] = None, depth: int = 0) -> None:
        if depth > 6:
            return

        if isinstance(obj, dict):
            for v in obj.values():
                walk(v, default_cls=default_cls, depth=depth + 1)
            return

        a = safe_array(obj)
        if a is not None and a.dtype != object:
            if decode_array(a, default_cls=default_cls):
                return

        if isinstance(obj, (list, tuple)):
            # Dla 4 klas HailoRT może zwrócić listę per klasa.
            if len(obj) == 4 and default_cls is None:
                for cls_id, item in enumerate(obj):
                    walk(item, default_cls=cls_id, depth=depth + 1)
            else:
                for item in obj:
                    walk(item, default_cls=default_cls, depth=depth + 1)

    walk(outputs)
    detections.sort(key=lambda d: d["conf"], reverse=True)
    return detections


class HailoYOLORunner:
    """Stały runner HailoRT dla jednego pliku HEF."""

    def __init__(self, hef_path: Path, conf: float = 0.4, box_order: str = "yxyx"):
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                "Nie udało się zaimportować hailo_platform. "
                f"Szczegóły: {repr(HAILO_IMPORT_ERROR)}"
            )

        self.hef_path = Path(hef_path)
        self.conf = conf
        self.box_order = box_order
        self.stack = ExitStack()

        self.hef = HEF(str(self.hef_path))
        input_infos = self.hef.get_input_vstream_infos()
        output_infos = self.hef.get_output_vstream_infos()

        if len(input_infos) != 1:
            raise RuntimeError(f"Skrypt zakłada jedno wejście modelu. Wykryto: {len(input_infos)}")

        self.input_name = input_infos[0].name
        self.input_h, self.input_w, self.input_c = normalize_shape(input_infos[0].shape)
        self.output_names = [info.name for info in output_infos]

        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)

        self.vdevice = self.stack.enter_context(VDevice())
        network_groups = self.vdevice.configure(self.hef, configure_params)
        if not network_groups:
            raise RuntimeError("Nie udało się skonfigurować Hailo network group.")

        self.network_group = network_groups[0]
        self.network_group_params = self.network_group.create_params()

        input_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        output_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        self.infer_pipeline = self.stack.enter_context(InferVStreams(self.network_group, input_params, output_params))
        self.stack.enter_context(self.network_group.activate(self.network_group_params))

        print(f"  HEF: {self.hef_path.name}")
        print(f"  Hailo input: {self.input_name}, HWC={self.input_h}x{self.input_w}x{self.input_c}")
        print(f"  Hailo outputs: {self.output_names}")

    def close(self):
        self.stack.close()

    def detect(self, frame_bgr: np.ndarray, mask: np.ndarray, photometry_on: bool, apply_health_index_func):
        """
        Zwraca: detections, inference_ms, postprocess_ms, hi_stats.
        Inference_ms obejmuje przygotowanie wejścia Hailo + inferencję HailoRT.
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        t0 = time.perf_counter()
        inp_rgb, scale, pad = letterbox_bgr_to_rgb(frame_bgr, self.input_h, self.input_w)
        input_data = {self.input_name: np.expand_dims(inp_rgb, axis=0)}
        outputs = self.infer_pipeline.infer(input_data)
        t_infer = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        raw_detections = try_decode_hailo_nms(
            outputs,
            scale=scale,
            pad=pad,
            orig_w=orig_w,
            orig_h=orig_h,
            input_w=self.input_w,
            input_h=self.input_h,
            conf_thres=self.conf,
            box_order=self.box_order,
        )

        valid_detections = []
        for det in raw_detections:
            cx, cy = det["cx"], det["cy"]
            if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0] and mask[cy, cx] == 255:
                valid_detections.append(det)

        hi_stats = (
            apply_health_index_func(frame_bgr, valid_detections, padding=0)
            if photometry_on
            else {"warn_count": 0, "error_count": 0, "hi_values": []}
        )

        t_post = (time.perf_counter() - t1) * 1000.0
        return valid_detections, t_infer, t_post, hi_stats


# ==============================================================================
# GŁÓWNA KLASA BENCHMARKU
# ==============================================================================
class MasterBenchmark:
    def validate_paths(self):
        missing = []

        if not DATASET_DIR.exists():
            missing.append(f"DATASET_DIR: {DATASET_DIR}")
        if not IMAGES_DIR.exists():
            missing.append(f"IMAGES_DIR: {IMAGES_DIR}")
        if not LABELS_DIR.exists():
            missing.append(f"LABELS_DIR: {LABELS_DIR}")
        if not CONFIG_FILE.exists():
            missing.append(f"CONFIG_FILE: {CONFIG_FILE}")
        if not HAILO_DIR.exists():
            missing.append(f"HAILO_DIR: {HAILO_DIR}")

        for model_info in HAILO_MODELS:
            if not model_info["path"].exists():
                missing.append(f"HEF: {model_info['path']}")

        if missing:
            print("\nBŁĄD: Nie znaleziono wymaganych plików lub katalogów:")
            for item in missing:
                print(f"  - {item}")

            print("\nOczekiwana domyślna struktura katalogów:")
            print("  00_TESTY/")
            print("  ├── master_benchmark_hailo.py")
            print("  ├── test/images/dataset_config.json")
            print("  ├── test/labels/")
            print("  └── hef/*.hef")
            raise FileNotFoundError("Brak wymaganych plików/katalogów benchmarku Hailo.")

        if not any(Path(p).suffix.lower() in {".jpg", ".jpeg", ".png"} for p in IMAGES_DIR.iterdir() if p.is_file()):
            raise FileNotFoundError(f"Nie znaleziono obrazów .jpg/.jpeg/.png w katalogu: {IMAGES_DIR}")

    def __init__(self):
        self.validate_paths()

        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.image_files = sorted(
            glob.glob(str(IMAGES_DIR / "*.jpg")) +
            glob.glob(str(IMAGES_DIR / "*.jpeg")) +
            glob.glob(str(IMAGES_DIR / "*.png"))
        )

        if LIMIT_IMAGES > 0:
            self.image_files = self.image_files[:LIMIT_IMAGES]

        print("Konfiguracja benchmarku Hailo:")
        print(f"  BASE_DIR: {BASE_DIR}")
        print(f"  DATASET_DIR: {DATASET_DIR}")
        print(f"  IMAGES_DIR: {IMAGES_DIR}")
        print(f"  LABELS_DIR: {LABELS_DIR}")
        print(f"  CONFIG_FILE: {CONFIG_FILE}")
        print(f"  HAILO_DIR: {HAILO_DIR}")
        print(f"  RESULTS_CSV: {RESULTS_CSV}")
        print(f"  PLATFORM_NAME: {PLATFORM_NAME}")
        print(f"  HAILO_CONF: {HAILO_CONF}")
        print(f"  HAILO_BOX_ORDER: {HAILO_BOX_ORDER}")
        print(f"  RUN_CLASSICAL: {RUN_CLASSICAL}")
        print(f"  OPENCV_THREADS: {OPENCV_THREADS}")
        print(f"  LIMIT_IMAGES: {LIMIT_IMAGES}")
        print(f"  Liczba obrazów: {len(self.image_files)}")

        with open(RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                "Platform", "Algorithm", "Model_Path", "Resolution", "Preset", "Photometry",
                "Time_of_Day", "Weather", "Total_Images", "Total_GT_Lamps_in_ROI",
                "TP", "FP", "FN", "Precision", "Recall", "F1_Score", "Avg_IoU",
                "Class_Evaluable", "Class_Correct", "Class_Unknown", "Class_Accuracy",
                "HI_Warn_Count", "HI_Error_Count", "Avg_HI",
                "Preprocess_ms", "Inference_ms", "Postprocess_ms",
                "Avg_FPS", "1%_Low_FPS", "CPU_Usage_%", "RAM_Usage_MB",
            ])

    def create_roi_mask(self, w, h, roi_params):
        roi_top_y = int(h * (roi_params["roi_top"] / 100.0))
        top_w = int(w * (roi_params["top_width"] / 100.0))
        bottom_w = int(w * (roi_params["bottom_width"] / 100.0))
        center_x = (w // 2) + int((roi_params["offset_x"] - 100) * (w / 640.0))

        pts = np.array([
            [center_x - top_w // 2, roi_top_y],
            [center_x + top_w // 2, roi_top_y],
            [center_x + bottom_w // 2, h],
            [center_x - bottom_w // 2, h],
        ], np.int32)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return pts, mask

    # --------------------------------------------------------------------------
    # Moduł względnego Health Index
    # --------------------------------------------------------------------------
    def apply_health_index(self, frame, detections, padding=0):
        for det in detections:
            det["raw_br"] = 0.0
            det["hi"] = 1.0

        if len(detections) < 3:
            return {"warn_count": 0, "error_count": 0, "hi_values": []}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cys = []
        brs = []

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            roi = gray[y1:y2, x1:x2]
            raw_br = float(np.sum(roi[roi > BRIGHTNESS_THRESHOLD])) if roi.size > 0 else 0.0
            det["raw_br"] = raw_br
            cys.append(det.get("cy", int((det["box"][1] + det["box"][3]) / 2)))
            brs.append(raw_br)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                poly_coefs = np.polyfit(cys, brs, deg=2)
            ref_curve = np.poly1d(poly_coefs)
        except Exception:
            mean_br = float(np.mean(brs)) if len(brs) > 0 else 1.0
            ref_curve = lambda _y: mean_br

        warn_count = 0
        error_count = 0
        hi_values = []

        for det in detections:
            expected = float(ref_curve(det.get("cy", int((det["box"][1] + det["box"][3]) / 2))))
            if expected <= 0:
                expected = 1.0

            det["hi"] = float(det["raw_br"] / expected)
            hi_values.append(det["hi"])

            if det["hi"] <= HI_ERROR_THRESHOLD:
                error_count += 1
            elif det["hi"] <= HI_WARNING_THRESHOLD:
                warn_count += 1

        return {"warn_count": warn_count, "error_count": error_count, "hi_values": hi_values}

    # --------------------------------------------------------------------------
    # Klasyczne metody OpenCV
    # --------------------------------------------------------------------------
    def detect_canny_headless(self, frame, mask, preset, photometry_on):
        t0 = time.perf_counter()
        if preset == "Dzien":
            canny_t, min_l, ers, cl_k = 255, 0, 20, 1
        else:
            canny_t, min_l, ers, cl_k = 150, 100, 10, 5

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 5)
        edges = cv2.Canny(gray_blurred, canny_t // 2, canny_t)
        edges_roi = cv2.bitwise_and(edges, edges, mask=mask)

        lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180, threshold=40, minLineLength=min_l, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edges_roi, (x1, y1), (x2, y2), 0, ers)

        if cl_k > 0:
            if cl_k % 2 == 0:
                cl_k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cl_k, cl_k))
            edges_roi = cv2.morphologyEx(edges_roi, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        t_infer = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        valid_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 <= area <= 1000 and len(cnt) >= 5:
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = int(x + w / 2), int(y + h / 2)
                valid_detections.append({
                    "box": [x, y, x + w, y + h],
                    "cx": cx,
                    "cy": cy,
                    "color": normalize_color_name(get_lamp_color(frame, cx, cy)),
                    "conf": 1.0,
                })

        hi_stats = self.apply_health_index(frame, valid_detections, padding=15) if photometry_on else {"warn_count": 0, "error_count": 0, "hi_values": []}
        t_post = (time.perf_counter() - t1) * 1000
        return valid_detections, t_infer, t_post, hi_stats

    def detect_pp_headless(self, frame, mask, photometry_on):
        t0 = time.perf_counter()
        thresh_v, working_k_size = 250, 3
        min_a, max_a, min_bright_ratio = 10, 1000, 0.20

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_roi = cv2.bitwise_and(gray, gray, mask=mask)

        _, binary = cv2.threshold(gray_roi, thresh_v, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (working_k_size, working_k_size))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_infer = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        valid_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_a <= area <= max_a:
                obj_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(obj_mask, [cnt], -1, 255, -1)

                pixel_values = gray[obj_mask == 255]
                if len(pixel_values) > 0:
                    bright_ratio = np.sum(pixel_values > 200) / len(pixel_values)
                    if bright_ratio >= min_bright_ratio:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cx, cy = int(x + w / 2), int(y + h / 2)
                        valid_detections.append({
                            "box": [x, y, x + w, y + h],
                            "cx": cx,
                            "cy": cy,
                            "color": normalize_color_name(get_lamp_color(frame, cx, cy)),
                            "conf": 1.0,
                        })

        hi_stats = self.apply_health_index(frame, valid_detections, padding=15) if photometry_on else {"warn_count": 0, "error_count": 0, "hi_values": []}
        t_post = (time.perf_counter() - t1) * 1000
        return valid_detections, t_infer, t_post, hi_stats

    def detect_tophat_headless(self, frame, mask, preset, photometry_on):
        t0 = time.perf_counter()
        if preset == "Dzien":
            tophat_k, thresh_val, min_a, max_a, max_aspect = 9, 125, 5, 1500, 3.0
        else:
            tophat_k, thresh_val, min_a, max_a, max_aspect = 50, 125, 5, 1500, 3.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_k, tophat_k))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        tophat_roi = cv2.bitwise_and(tophat, tophat, mask=mask)

        _, binary = cv2.threshold(tophat_roi, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t_infer = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        valid_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_a <= area <= max_a:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 0 and h > 0:
                    aspect_ratio = float(w) / h
                    if aspect_ratio < 1:
                        aspect_ratio = 1 / aspect_ratio
                    if aspect_ratio <= max_aspect:
                        cx, cy = int(x + w / 2), int(y + h / 2)
                        valid_detections.append({
                            "box": [x, y, x + w, y + h],
                            "cx": cx,
                            "cy": cy,
                            "color": normalize_color_name(get_lamp_color(frame, cx, cy)),
                            "conf": 1.0,
                        })

        hi_stats = self.apply_health_index(frame, valid_detections, padding=15) if photometry_on else {"warn_count": 0, "error_count": 0, "hi_values": []}
        t_post = (time.perf_counter() - t1) * 1000
        return valid_detections, t_infer, t_post, hi_stats

    # --------------------------------------------------------------------------
    # Silnik testowy
    # --------------------------------------------------------------------------
    def run(self):
        print(f"Rozpoczynam benchmark. Zajętość RAM: {psutil.virtual_memory().percent}%")

        combinations = list(itertools.product(ALGORITHMS, RESOLUTIONS, PRESETS, PHOTOMETRY_STATES))

        for algo, res, preset, photo_on in combinations:
            if algo in ["HAILO_YOLO", "PP"] and preset == "Noc":
                continue

            if algo == "HAILO_YOLO":
                models_to_test = [m for m in HAILO_MODELS if m["resolution"] == res]
            else:
                models_to_test = [None]

            if not models_to_test:
                continue

            for model_info in models_to_test:
                model_name = model_info["path"].name if model_info else "N/A"
                print_preset = "N/A" if algo in ["HAILO_YOLO", "PP"] else preset
                print(f"--- TESTUJĘ: {algo} | Res: {res} | Preset: {print_preset} | Photo: {photo_on} | Model: {model_name} ---")

                hailo_net = None
                try:
                    if algo == "HAILO_YOLO":
                        hailo_net = HailoYOLORunner(model_info["path"], conf=HAILO_CONF, box_order=HAILO_BOX_ORDER)

                        # Rozgrzewka HailoRT. Nie zapisujemy jej do wyników.
                        dummy = np.zeros((res[1], res[0], 3), dtype=np.uint8)
                        dummy_mask = np.full((res[1], res[0]), 255, dtype=np.uint8)
                        for _ in range(3):
                            _ = hailo_net.detect(dummy, dummy_mask, False, self.apply_health_index)

                    stats_by_condition = {}
                    psutil.cpu_percent(interval=None)

                    for img_path in self.image_files:
                        img_name = os.path.basename(img_path)
                        if img_name not in self.config:
                            continue

                        tag_time = self.config[img_name]["tags"]["time"]
                        tag_weather = self.config[img_name]["tags"]["weather"]
                        cond_key = (tag_time, tag_weather)

                        if cond_key not in stats_by_condition:
                            stats_by_condition[cond_key] = {
                                "TP": 0, "FP": 0, "FN": 0, "Total_GT": 0, "img_count": 0,
                                "ious": [],
                                "class_evaluable": 0, "class_correct": 0, "class_unknown": 0,
                                "hi_warn": 0, "hi_error": 0, "hi_values": [],
                                "prep_times": [], "infer_times": [], "post_times": [],
                                "frame_times": [],
                                "cpu_samples": [], "ram_samples": [],
                            }

                        frame_raw = cv2.imread(str(img_path))
                        if frame_raw is None:
                            print(f"Nie udało się wczytać obrazu: {img_path}")
                            continue

                        # 1. Preprocessing wspólny: resize do konfiguracji benchmarku + ROI.
                        t_prep_start = time.perf_counter()
                        frame = cv2.resize(frame_raw, res)
                        _, mask = self.create_roi_mask(res[0], res[1], self.config[img_name]["roi"])
                        t_prep = (time.perf_counter() - t_prep_start) * 1000

                        # 2. GT przeskalowany do tej samej rozdzielczości co frame.
                        txt_name = img_name.rsplit(".", 1)[0] + ".txt"
                        gt_txt_path = LABELS_DIR / txt_name
                        all_ground_truths = load_ground_truth(gt_txt_path, res[0], res[1])
                        valid_ground_truths = filter_ground_truths_by_roi(all_ground_truths, mask)

                        stats_by_condition[cond_key]["Total_GT"] += len(valid_ground_truths)
                        stats_by_condition[cond_key]["img_count"] += 1

                        # 3. Detekcja.
                        if algo == "HAILO_YOLO":
                            preds, t_inf, t_post, hi_stats = hailo_net.detect(frame, mask, photo_on, self.apply_health_index)
                        elif algo == "Canny":
                            preds, t_inf, t_post, hi_stats = self.detect_canny_headless(frame, mask, preset, photo_on)
                        elif algo == "PP":
                            preds, t_inf, t_post, hi_stats = self.detect_pp_headless(frame, mask, photo_on)
                        elif algo == "TopHat":
                            preds, t_inf, t_post, hi_stats = self.detect_tophat_headless(frame, mask, preset, photo_on)
                        else:
                            raise ValueError(f"Nieznany algorytm: {algo}")

                        # 4. Ocena.
                        tp, fp, fn, matched_ious, class_eval, class_correct, class_unknown = evaluate_predictions(preds, valid_ground_truths)

                        stats_by_condition[cond_key]["TP"] += tp
                        stats_by_condition[cond_key]["FP"] += fp
                        stats_by_condition[cond_key]["FN"] += fn
                        stats_by_condition[cond_key]["ious"].extend(matched_ious)
                        stats_by_condition[cond_key]["class_evaluable"] += class_eval
                        stats_by_condition[cond_key]["class_correct"] += class_correct
                        stats_by_condition[cond_key]["class_unknown"] += class_unknown
                        stats_by_condition[cond_key]["hi_warn"] += hi_stats["warn_count"]
                        stats_by_condition[cond_key]["hi_error"] += hi_stats["error_count"]
                        stats_by_condition[cond_key]["hi_values"].extend(hi_stats["hi_values"])

                        stats_by_condition[cond_key]["prep_times"].append(t_prep)
                        stats_by_condition[cond_key]["infer_times"].append(t_inf)
                        stats_by_condition[cond_key]["post_times"].append(t_post)

                        total_frame_ms = t_prep + t_inf + t_post
                        if total_frame_ms > 0:
                            stats_by_condition[cond_key]["frame_times"].append(total_frame_ms)

                        stats_by_condition[cond_key]["cpu_samples"].append(psutil.cpu_percent(interval=None))
                        stats_by_condition[cond_key]["ram_samples"].append(psutil.virtual_memory().used / (1024 * 1024))

                    self.write_condition_rows(algo, model_name, res, print_preset, photo_on, stats_by_condition)
                    self.write_all_row(algo, model_name, res, print_preset, photo_on, stats_by_condition)

                    print("--> Zakończono konfigurację.")

                finally:
                    if hailo_net is not None:
                        hailo_net.close()
                    gc.collect()

    def write_condition_rows(self, algo, model_name, res, print_preset, photo_on, stats_by_condition):
        for cond_key, stats in stats_by_condition.items():
            tag_time, tag_weather = cond_key
            row = self.compute_row_stats(stats)

            with open(RESULTS_CSV, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    PLATFORM_NAME, algo, model_name, f"{res[0]}x{res[1]}", print_preset, photo_on,
                    tag_time, tag_weather, stats["img_count"], stats["Total_GT"],
                    row["TP"], row["FP"], row["FN"],
                    row["precision"], row["recall"], row["f1"], row["iou"],
                    row["class_evaluable"], row["class_correct"], row["class_unknown"], row["class_accuracy"],
                    row["hi_warn"], row["hi_error"], row["avg_hi"],
                    row["avg_prep"], row["avg_inf"], row["avg_post"],
                    row["avg_fps"], row["low_1_fps"],
                    row["cpu_use"], row["ram_use"],
                ])

    def write_all_row(self, algo, model_name, res, print_preset, photo_on, stats_by_condition):
        merged = {
            "TP": 0, "FP": 0, "FN": 0, "Total_GT": 0, "img_count": 0,
            "ious": [],
            "class_evaluable": 0, "class_correct": 0, "class_unknown": 0,
            "hi_warn": 0, "hi_error": 0, "hi_values": [],
            "prep_times": [], "infer_times": [], "post_times": [],
            "frame_times": [],
            "cpu_samples": [], "ram_samples": [],
        }

        for s in stats_by_condition.values():
            for key in ["TP", "FP", "FN", "Total_GT", "img_count", "class_evaluable", "class_correct", "class_unknown", "hi_warn", "hi_error"]:
                merged[key] += s[key]
            for key in ["ious", "hi_values", "prep_times", "infer_times", "post_times", "frame_times", "cpu_samples", "ram_samples"]:
                merged[key].extend(s[key])

        row = self.compute_row_stats(merged)

        with open(RESULTS_CSV, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                PLATFORM_NAME, algo, model_name, f"{res[0]}x{res[1]}", print_preset, photo_on,
                "All", "All", merged["img_count"], merged["Total_GT"],
                row["TP"], row["FP"], row["FN"],
                row["precision"], row["recall"], row["f1"], row["iou"],
                row["class_evaluable"], row["class_correct"], row["class_unknown"], row["class_accuracy"],
                row["hi_warn"], row["hi_error"], row["avg_hi"],
                row["avg_prep"], row["avg_inf"], row["avg_post"],
                row["avg_fps"], row["low_1_fps"],
                row["cpu_use"], row["ram_use"],
            ])

    @staticmethod
    def compute_row_stats(stats):
        total_tp = stats["TP"]
        total_fp = stats["FP"]
        total_fn = stats["FN"]

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(stats["ious"]) if len(stats["ious"]) > 0 else 0.0

        class_accuracy = (stats["class_correct"] / stats["class_evaluable"]) if stats["class_evaluable"] > 0 else 0.0
        avg_hi = np.mean(stats["hi_values"]) if len(stats["hi_values"]) > 0 else 0.0

        avg_prep = np.mean(stats["prep_times"]) if stats["prep_times"] else 0.0
        avg_inf = np.mean(stats["infer_times"]) if stats["infer_times"] else 0.0
        avg_post = np.mean(stats["post_times"]) if stats["post_times"] else 0.0

        avg_frame_time = np.mean(stats["frame_times"]) if stats["frame_times"] else 0.0
        avg_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0

        frame_times_sorted = np.sort(stats["frame_times"])
        one_percent_idx = max(1, int(len(frame_times_sorted) * 0.01))
        slowest_times = frame_times_sorted[-one_percent_idx:] if len(frame_times_sorted) > 0 else []
        low_1_fps = 1000.0 / np.mean(slowest_times) if len(slowest_times) > 0 else 0.0

        cpu_use = np.mean(stats["cpu_samples"]) if stats["cpu_samples"] else 0.0
        ram_use = np.mean(stats["ram_samples"]) if stats["ram_samples"] else 0.0

        return {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "iou": round(mean_iou, 4),
            "class_evaluable": stats["class_evaluable"],
            "class_correct": stats["class_correct"],
            "class_unknown": stats["class_unknown"],
            "class_accuracy": round(class_accuracy, 4),
            "hi_warn": stats["hi_warn"],
            "hi_error": stats["hi_error"],
            "avg_hi": round(avg_hi, 4),
            "avg_prep": round(avg_prep, 2),
            "avg_inf": round(avg_inf, 2),
            "avg_post": round(avg_post, 2),
            "avg_fps": round(avg_fps, 1),
            "low_1_fps": round(low_1_fps, 1),
            "cpu_use": round(cpu_use, 1),
            "ram_use": round(ram_use, 1),
        }


if __name__ == "__main__":
    benchmark = MasterBenchmark()
    benchmark.run()
    print("\nBenchmark zakończony! Wyniki zapisano do:", str(RESULTS_CSV))
