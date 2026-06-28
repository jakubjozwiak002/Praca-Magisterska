import os
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
from ultralytics import YOLO

# ==============================================================================
# KONFIGURACJA ZBIORU
# ==============================================================================
DATASET_DIR = r"C:\Users\kubat\OneDrive\Pulpit\STUDIA\MAGISTERKA\Skrypty\00_TESTY\test"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Twój plik JSON zapisał się w folderze images:
CONFIG_FILE = os.path.join(IMAGES_DIR, "dataset_config.json") 
RESULTS_CSV = "benchmark_results.csv"

# Nazwa platformy zapisywana w pliku wynikowym.
# Można ją nadpisać bez edycji kodu, np.:
# Windows PowerShell: $env:BENCHMARK_PLATFORM="RPI5_CPU"
# Linux/Raspberry Pi: export BENCHMARK_PLATFORM="RPI5_CPU"
PLATFORM_NAME = os.environ.get("BENCHMARK_PLATFORM", "MSI_Katana")

# Mapowanie identyfikatorów klas z plików etykiet YOLO na nazwy kolorów.
# WAŻNE: przed finalnym benchmarkiem należy zweryfikować kolejność klas
# względem pliku dataset.yaml użytego do treningu modeli.
CLASS_ID_TO_COLOR = {
    0: "Zielony",
    1: "Zolty",
    2: "Czerwony",
    3: "Bialy",
}

BRIGHTNESS_THRESHOLD = 150
HI_WARNING_THRESHOLD = 0.80
HI_ERROR_THRESHOLD = 0.40

# ==============================================================================
# PRZESTRZEŃ ZMIENNYCH (MACIERZ TESTOWA)
# ==============================================================================
ALGORITHMS = ['YOLO', 'Canny', 'PP', 'TopHat'] 
RESOLUTIONS = [(640, 480), (320, 240)] # Testujemy pełną i obniżoną jakość
PHOTOMETRY_STATES = [False, True]
PRESETS = ['Dzien', 'Noc']
YOLO_MODELS = [
    r"C:\Users\kubat\OneDrive\Pulpit\STUDIA\MAGISTERKA\Skrypty\00_TESTY\modele\8\yolov8.pt",
    r"C:\Users\kubat\OneDrive\Pulpit\STUDIA\MAGISTERKA\Skrypty\00_TESTY\modele\11\yolov11.pt",
    r"C:\Users\kubat\OneDrive\Pulpit\STUDIA\MAGISTERKA\Skrypty\00_TESTY\modele\12\yolov12.pt",
    r"C:\Users\kubat\OneDrive\Pulpit\STUDIA\MAGISTERKA\Skrypty\00_TESTY\modele\26\yolov26.pt"
]

# ==============================================================================
# FUNKCJE MATEMATYCZNE (METRYKI IOU I GT)
# ==============================================================================
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def normalize_color_name(value):
    """Ujednolica nazwy klas kolorów pochodzące z YOLO, etykiet GT i klasyfikatora HSV."""
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


def get_model_class_name(model, class_id):
    """Zwraca nazwę klasy modelu YOLO dla podanego identyfikatora klasy.

    Funkcja obsługuje najczęstsze formaty `model.names` używane przez Ultralytics:
    słownik `{id: nazwa}` albo listę/tuplę nazw. W przypadku braku nazwy
    stosowane jest mapowanie CLASS_ID_TO_COLOR jako bezpieczne rozwiązanie awaryjne.
    """
    try:
        class_id = int(class_id)
    except (TypeError, ValueError):
        return "Inny"

    if class_id < 0:
        return "Inny"

    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)

    try:
        if isinstance(names, dict):
            return names.get(class_id, names.get(str(class_id), CLASS_ID_TO_COLOR.get(class_id, "Inny")))

        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return names[class_id]
    except Exception:
        pass

    return CLASS_ID_TO_COLOR.get(class_id, "Inny")


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
    if not os.path.exists(txt_path):
        return objects

    with open(txt_path, 'r') as f:
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
    """Odrzuca obiekty referencyjne, których środek znajduje się poza maską ROI."""
    filtered_gt = []
    for gt in gt_objects:
        box = gt["box"]
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]:
            if mask[cy, cx] == 255:
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
# GŁÓWNA KLASA BENCHMARKU
# ==============================================================================
class MasterBenchmark:
    def __init__(self):
        with open(CONFIG_FILE, 'r') as f:
            self.config = json.load(f)
            
        self.image_files = sorted(
            glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) +
            glob.glob(os.path.join(IMAGES_DIR, "*.png"))
        )

        # Plik wynikowy jest nadpisywany przy każdym uruchomieniu benchmarku.
        # Dzięki temu nie dojdzie do przypadkowego dopisania nowych wyników
        # do rezultatów z poprzedniej próby.
        with open(RESULTS_CSV, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                'Platform', 'Algorithm', 'Model_Path', 'Resolution', 'Preset', 'Photometry', 
                'Time_of_Day', 'Weather', 'Total_Images', 'Total_GT_Lamps_in_ROI',
                'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1_Score', 'Avg_IoU', 'Class_Evaluable', 'Class_Correct', 'Class_Unknown', 'Class_Accuracy',
                'HI_Warn_Count', 'HI_Error_Count', 'Avg_HI',
                'Preprocess_ms', 'Inference_ms', 'Postprocess_ms', 
                'Avg_FPS', '1%_Low_FPS', 'CPU_Usage_%', 'RAM_Usage_MB'
            ])

    def create_roi_mask(self, w, h, roi_params):
        roi_top_y = int(h * (roi_params['roi_top'] / 100.0))
        top_w = int(w * (roi_params['top_width'] / 100.0))
        bottom_w = int(w * (roi_params['bottom_width'] / 100.0))
        center_x = (w // 2) + int((roi_params['offset_x'] - 100) * (w / 640.0))
        
        pts = np.array([
            [center_x - top_w // 2, roi_top_y], 
            [center_x + top_w // 2, roi_top_y],
            [center_x + bottom_w // 2, h], 
            [center_x - bottom_w // 2, h]
        ], np.int32)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return pts, mask

    # --- PURE HEADLESS ALGORITHMS ---
    def apply_health_index(self, frame, detections, padding=0):
        """
        Wyznacza pełny względny wskaźnik jasności dla detekcji w sposób zgodny
        z modułem używanym w aplikacji: suma pikseli > progu, dopasowanie krzywej
        odniesienia i obliczenie HI = jasność surowa / jasność oczekiwana.
        """
        for det in detections:
            det['raw_br'] = 0.0
            det['hi'] = 1.0

        if len(detections) < 3:
            return {"warn_count": 0, "error_count": 0, "hi_values": []}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cys = []
        brs = []

        for det in detections:
            x1, y1, x2, y2 = det['box']
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)

            roi = gray[y1:y2, x1:x2]
            raw_br = float(np.sum(roi[roi > BRIGHTNESS_THRESHOLD])) if roi.size > 0 else 0.0
            det['raw_br'] = raw_br
            cys.append(det.get('cy', int((det['box'][1] + det['box'][3]) / 2)))
            brs.append(raw_br)

        # Domyślnie używany jest wielomian drugiego stopnia, jak w module interaktywnym.
        # Jeśli dane są osobliwe, stosowana jest wartość średnia jako bezpieczne odniesienie.
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
            expected = float(ref_curve(det.get('cy', int((det['box'][1] + det['box'][3]) / 2))))
            if expected <= 0:
                expected = 1.0
            det['hi'] = float(det['raw_br'] / expected)
            hi_values.append(det['hi'])

            if det['hi'] <= HI_ERROR_THRESHOLD:
                error_count += 1
            elif det['hi'] <= HI_WARNING_THRESHOLD:
                warn_count += 1

        return {"warn_count": warn_count, "error_count": error_count, "hi_values": hi_values}

    def detect_yolo_headless(self, model, frame, mask, photometry_on):
        t0 = time.perf_counter()
        imgsz = max(frame.shape[:2])
        results = model.predict(source=frame, conf=0.4, imgsz=imgsz, verbose=False)
        t_infer = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        valid_detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
            if 0 <= cx < mask.shape[1] and 0 <= cy < mask.shape[0]:
                if mask[cy, cx] == 255:
                    cls_id = int(box.cls[0]) if box.cls is not None else -1
                    cls_name = get_model_class_name(model, cls_id)
                    valid_detections.append({
                        "box": [x1, y1, x2, y2],
                        "cx": cx,
                        "cy": cy,
                        "class_id": cls_id,
                        "color": normalize_color_name(cls_name),
                        "conf": float(box.conf[0]) if box.conf is not None else 1.0,
                    })

        hi_stats = self.apply_health_index(frame, valid_detections, padding=0) if photometry_on else {"warn_count": 0, "error_count": 0, "hi_values": []}

        t_post = (time.perf_counter() - t1) * 1000
        return valid_detections, t_infer, t_post, hi_stats

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

        lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, threshold=40, minLineLength=min_l, maxLineGap=10)
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
                    "box": [x, y, x+w, y+h],
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
                            "box": [x, y, x+w, y+h],
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
                            "box": [x, y, x+w, y+h],
                            "cx": cx,
                            "cy": cy,
                            "color": normalize_color_name(get_lamp_color(frame, cx, cy)),
                            "conf": 1.0,
                        })

        hi_stats = self.apply_health_index(frame, valid_detections, padding=15) if photometry_on else {"warn_count": 0, "error_count": 0, "hi_values": []}

        t_post = (time.perf_counter() - t1) * 1000
        return valid_detections, t_infer, t_post, hi_stats

    # ==============================================================================
    # SILNIK TESTOWY (RUNNER)
    # ==============================================================================
    def run(self):
        print(f"Rozpoczynam Benchmark. Zablokowano RAM: {psutil.virtual_memory().percent}%")
        
        combinations = list(itertools.product(ALGORITHMS, RESOLUTIONS, PRESETS, PHOTOMETRY_STATES))
        
        for algo, res, preset, photo_on in combinations:
            
            # Filtrowanie bezsensownych kombinacji
            # YOLO i PP nie używają presetów Dzień/Noc, więc omijamy dla nich pętlę "Noc", żeby nie dublować testu
            if algo in ['YOLO', 'PP'] and preset == 'Noc':
                continue 
            
            # Dla YOLO iterujemy też po jego modelach (dla klasyki lista ma 1 element: None)
            models_to_test = YOLO_MODELS if algo == 'YOLO' else [None]
            
            for yolo_model_path in models_to_test:
                model_name = os.path.basename(yolo_model_path) if yolo_model_path else "N/A"
                print_preset = "N/A" if algo in ['YOLO', 'PP'] else preset
                print(f"--- TESTUJĘ: {algo} | Res: {res} | Preset: {print_preset} | Photo: {photo_on} | Model: {model_name} ---")
                
                yolo_net = YOLO(yolo_model_path) if algo == 'YOLO' else None

                # Rozgrzewka modelu YOLO.
                # Pierwsze wywołanie predykcji często obejmuje inicjalizację wewnętrzną,
                # dlatego nie powinno być uwzględniane w pomiarach właściwego benchmarku.
                if algo == 'YOLO':
                    dummy = np.zeros((res[1], res[0], 3), dtype=np.uint8)
                    for _ in range(3):
                        _ = yolo_net.predict(source=dummy, conf=0.4, imgsz=max(res), verbose=False)

                # Zamiast jednego słownika stats tworzony jest słownik grupujący wyniki
                # według tagów sceny, np. pory dnia i warunków pogodowych.
                stats_by_condition = {}

                # Pierwsze wywołanie cpu_percent inicjalizuje pomiar procentowego użycia CPU.
                psutil.cpu_percent(interval=None)

                for img_path in self.image_files:
                    img_name = os.path.basename(img_path)
                    if img_name not in self.config: continue 
                    
                    # Pobieranie tagów aktualnego zdjęcia
                    tag_time = self.config[img_name]['tags']['time']
                    tag_weather = self.config[img_name]['tags']['weather']
                    cond_key = (tag_time, tag_weather)
                    
                    # Inicjalizacja słownika dla nowej kombinacji warunków
                    if cond_key not in stats_by_condition:
                        stats_by_condition[cond_key] = {
                            'TP': 0, 'FP': 0, 'FN': 0, 'Total_GT': 0, 'img_count': 0,
                            'ious': [],
                            'class_evaluable': 0, 'class_correct': 0, 'class_unknown': 0,
                            'hi_warn': 0, 'hi_error': 0, 'hi_values': [],
                            'prep_times': [], 'infer_times': [], 'post_times': [],
                            'frame_times': [],
                            'cpu_samples': [], 'ram_samples': []
                        }
                    
                    frame_raw = cv2.imread(img_path)
                    if frame_raw is None:
                        print(f"Nie udało się wczytać obrazu: {img_path}")
                        continue
                    
                    # 1. PRE-PROCESSING
                    t_prep_start = time.perf_counter()
                    frame = cv2.resize(frame_raw, res)
                    _, mask = self.create_roi_mask(res[0], res[1], self.config[img_name]['roi'])
                    t_prep = (time.perf_counter() - t_prep_start) * 1000
                    
                    # 2. WCZYTYWANIE ETYKIET I FILTROWANIE PO ROI
                    txt_name = img_name.rsplit('.', 1)[0] + ".txt"
                    gt_txt_path = os.path.join(LABELS_DIR, txt_name)
                    
                    all_ground_truths = load_ground_truth(gt_txt_path, res[0], res[1])
                    valid_ground_truths = filter_ground_truths_by_roi(all_ground_truths, mask)
                    
                    stats_by_condition[cond_key]['Total_GT'] += len(valid_ground_truths)
                    stats_by_condition[cond_key]['img_count'] += 1
                    
                    # 3. INFERENCJA
                    if algo == 'YOLO':
                        preds, t_inf, t_post, hi_stats = self.detect_yolo_headless(yolo_net, frame, mask, photo_on)
                    elif algo == 'Canny':
                        preds, t_inf, t_post, hi_stats = self.detect_canny_headless(frame, mask, preset, photo_on)
                    elif algo == 'PP':
                        preds, t_inf, t_post, hi_stats = self.detect_pp_headless(frame, mask, photo_on)
                    elif algo == 'TopHat':
                        preds, t_inf, t_post, hi_stats = self.detect_tophat_headless(frame, mask, preset, photo_on)
                    
                    # 4. OCENA DLA KLATKI
                    tp, fp, fn, matched_ious, class_eval, class_correct, class_unknown = evaluate_predictions(preds, valid_ground_truths)
                    
                    stats_by_condition[cond_key]['TP'] += tp
                    stats_by_condition[cond_key]['FP'] += fp
                    stats_by_condition[cond_key]['FN'] += fn
                    stats_by_condition[cond_key]['ious'].extend(matched_ious)
                    stats_by_condition[cond_key]['class_evaluable'] += class_eval
                    stats_by_condition[cond_key]['class_correct'] += class_correct
                    stats_by_condition[cond_key]['class_unknown'] += class_unknown
                    stats_by_condition[cond_key]['hi_warn'] += hi_stats['warn_count']
                    stats_by_condition[cond_key]['hi_error'] += hi_stats['error_count']
                    stats_by_condition[cond_key]['hi_values'].extend(hi_stats['hi_values'])
                    
                    stats_by_condition[cond_key]['prep_times'].append(t_prep)
                    stats_by_condition[cond_key]['infer_times'].append(t_inf)
                    stats_by_condition[cond_key]['post_times'].append(t_post)
                    
                    total_frame_ms = t_prep + t_inf + t_post
                    if total_frame_ms > 0:
                        stats_by_condition[cond_key]['frame_times'].append(total_frame_ms)

                    # Próbki obciążenia systemu zbierane są po przetworzeniu każdej klatki.
                    # Dzięki temu zapisane wartości CPU/RAM lepiej opisują rzeczywisty przebieg
                    # danej konfiguracji niż jednorazowy odczyt przed testem.
                    stats_by_condition[cond_key]['cpu_samples'].append(psutil.cpu_percent(interval=None))
                    stats_by_condition[cond_key]['ram_samples'].append(psutil.virtual_memory().used / (1024 * 1024))

                # =======================================================
                # ZAPIS WYNIKÓW Z PODZIAŁEM NA TAGI (ZDJĘCIA DZIEŃ / NOC)
                # =======================================================
                for cond_key, stats in stats_by_condition.items():
                    tag_time, tag_weather = cond_key
                    
                    total_tp = stats['TP']
                    total_fp = stats['FP']
                    total_fn = stats['FN']
                    
                    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
                    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    mean_iou = np.mean(stats['ious']) if len(stats['ious']) > 0 else 0.0
                    class_accuracy = (stats['class_correct'] / stats['class_evaluable']) if stats['class_evaluable'] > 0 else 0.0
                    avg_hi = np.mean(stats['hi_values']) if len(stats['hi_values']) > 0 else 0.0
                    
                    avg_prep = np.mean(stats['prep_times']) if stats['prep_times'] else 0.0
                    avg_inf = np.mean(stats['infer_times']) if stats['infer_times'] else 0.0
                    avg_post = np.mean(stats['post_times']) if stats['post_times'] else 0.0

                    # FPS liczony jest z uśrednionego czasu klatki, a nie jako średnia
                    # z chwilowych wartości FPS. Ogranicza to zawyżanie wyniku przy
                    # zmiennych czasach przetwarzania.
                    avg_frame_time = np.mean(stats['frame_times']) if stats['frame_times'] else 0.0
                    avg_fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0

                    # 1% Low FPS liczony jest na podstawie 1% najwolniejszych klatek,
                    # czyli tych o największym czasie przetwarzania.
                    frame_times_sorted = np.sort(stats['frame_times'])
                    one_percent_idx = max(1, int(len(frame_times_sorted) * 0.01))
                    slowest_times = frame_times_sorted[-one_percent_idx:] if len(frame_times_sorted) > 0 else []
                    low_1_fps = 1000.0 / np.mean(slowest_times) if len(slowest_times) > 0 else 0.0

                    cpu_use = np.mean(stats['cpu_samples']) if stats['cpu_samples'] else 0.0
                    ram_use = np.mean(stats['ram_samples']) if stats['ram_samples'] else 0.0

                    with open(RESULTS_CSV, 'a', newline='') as f:
                        writer = csv.writer(f, delimiter=';')
                        writer.writerow([
                            PLATFORM_NAME, algo, model_name, f"{res[0]}x{res[1]}", print_preset, photo_on,
                            tag_time, tag_weather, stats['img_count'], stats['Total_GT'],
                            total_tp, total_fp, total_fn, 
                            round(precision, 4), round(recall, 4), round(f1, 4), round(mean_iou, 4),
                            stats['class_evaluable'], stats['class_correct'], stats['class_unknown'], round(class_accuracy, 4),
                            stats['hi_warn'], stats['hi_error'], round(avg_hi, 4),
                            round(avg_prep, 2), round(avg_inf, 2), round(avg_post, 2), 
                            round(avg_fps, 1), round(low_1_fps, 1), 
                            round(cpu_use, 1), round(ram_use, 1)
                        ])

                # =======================================================
                # ZAPIS WYNIKU ZBIORCZEGO ("All") DLA CAŁEGO ZBIORU
                # =======================================================
                total_tp_all = sum(s['TP'] for s in stats_by_condition.values())
                total_fp_all = sum(s['FP'] for s in stats_by_condition.values())
                total_fn_all = sum(s['FN'] for s in stats_by_condition.values())
                total_gt_all = sum(s['Total_GT'] for s in stats_by_condition.values())
                total_img_all = sum(s['img_count'] for s in stats_by_condition.values())
                
                # Zbieranie wszystkich list do jednej
                all_ious = []
                all_frame_times = []
                all_hi_values = []
                all_cpu_samples, all_ram_samples = [], []
                all_prep, all_inf, all_post = [], [], []
                total_class_evaluable_all = 0
                total_class_correct_all = 0
                total_class_unknown_all = 0
                total_hi_warn_all = 0
                total_hi_error_all = 0
                for s in stats_by_condition.values():
                    all_ious.extend(s['ious'])
                    all_frame_times.extend(s['frame_times'])
                    all_hi_values.extend(s['hi_values'])
                    total_class_evaluable_all += s['class_evaluable']
                    total_class_correct_all += s['class_correct']
                    total_class_unknown_all += s['class_unknown']
                    total_hi_warn_all += s['hi_warn']
                    total_hi_error_all += s['hi_error']
                    all_cpu_samples.extend(s['cpu_samples'])
                    all_ram_samples.extend(s['ram_samples'])
                    all_prep.extend(s['prep_times'])
                    all_inf.extend(s['infer_times'])
                    all_post.extend(s['post_times'])

                prec_all = total_tp_all / (total_tp_all + total_fp_all) if (total_tp_all + total_fp_all) > 0 else 0.0
                rec_all = total_tp_all / (total_tp_all + total_fn_all) if (total_tp_all + total_fn_all) > 0 else 0.0
                f1_all = 2 * (prec_all * rec_all) / (prec_all + rec_all) if (prec_all + rec_all) > 0 else 0.0
                iou_all = np.mean(all_ious) if len(all_ious) > 0 else 0.0
                class_accuracy_all = (total_class_correct_all / total_class_evaluable_all) if total_class_evaluable_all > 0 else 0.0
                avg_hi_all = np.mean(all_hi_values) if len(all_hi_values) > 0 else 0.0
                
                avg_frame_time_all = np.mean(all_frame_times) if len(all_frame_times) > 0 else 0.0
                fps_avg_all = 1000.0 / avg_frame_time_all if avg_frame_time_all > 0 else 0.0

                frame_times_sorted_all = np.sort(all_frame_times)
                one_percent_idx_all = max(1, int(len(frame_times_sorted_all) * 0.01))
                slowest_times_all = frame_times_sorted_all[-one_percent_idx_all:] if len(frame_times_sorted_all) > 0 else []
                low_1_fps_all = 1000.0 / np.mean(slowest_times_all) if len(slowest_times_all) > 0 else 0.0

                cpu_use_all = np.mean(all_cpu_samples) if len(all_cpu_samples) > 0 else 0.0
                ram_use_all = np.mean(all_ram_samples) if len(all_ram_samples) > 0 else 0.0

                with open(RESULTS_CSV, 'a', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow([
                        PLATFORM_NAME, algo, model_name, f"{res[0]}x{res[1]}", print_preset, photo_on,
                        "All", "All", total_img_all, total_gt_all,
                        total_tp_all, total_fp_all, total_fn_all, 
                        round(prec_all, 4), round(rec_all, 4), round(f1_all, 4), round(iou_all, 4),
                        total_class_evaluable_all, total_class_correct_all, total_class_unknown_all, round(class_accuracy_all, 4),
                        total_hi_warn_all, total_hi_error_all, round(avg_hi_all, 4),
                        round(np.mean(all_prep) if all_prep else 0, 2), 
                        round(np.mean(all_inf) if all_inf else 0, 2), 
                        round(np.mean(all_post) if all_post else 0, 2), 
                        round(fps_avg_all, 1), round(low_1_fps_all, 1), 
                        round(cpu_use_all, 1), round(ram_use_all, 1)
                    ])

                # Wypisujemy podsumowanie z wariantu warunków, żeby konsola żyła
                print(f"--> Zakończono konfigurację (Zapisano tagi oraz wiersz zbiorczy All)")
                
                if yolo_net:
                    del yolo_net
                gc.collect()

if __name__ == "__main__":
    benchmark = MasterBenchmark()
    benchmark.run()
    print("\nBenchmark zakończony! Wyniki zapisano do:", RESULTS_CSV)