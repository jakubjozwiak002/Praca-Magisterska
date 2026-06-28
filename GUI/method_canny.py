import cv2
import numpy as np
import time
from color_detector import get_lamp_color

def run_method_canny(source_path, source_type, get_params_callback, root):
    is_video = source_type in [0, 1] 
    
    frame_times_dict = {}
    detection_data_dict = {}
    frame_counter = 1
    
    if is_video:
        cap = cv2.VideoCapture(source_path if source_type == 1 else 0)
        ret, current_frame = cap.read()
        if not ret:
            print("[Canny+Elipsy] Błąd odczytu strumienia wideo.")
            return [], [] 
        is_playing = False
    else:
        current_frame = cv2.imread(source_path)
        if current_frame is None:
            print("[Canny+Elipsy] Błąd odczytu obrazu.")
            return [], [] 
        is_playing = False

    cv2.namedWindow("Wynik Koncowy (Canny + Elipsy)", cv2.WINDOW_NORMAL)

    while True:
        root.update()
        
        start_time = time.time()
        current_frame_detections = []
        valid_detections_info = [] # Lista robocza na znalezione lampy

        params = get_params_callback()
        
        # Parametry ROI
        roi_top_perc = params.get('roi_top', 30)
        top_width_perc = params.get('top_width', 10)
        bottom_width_perc = params.get('bottom_width', 50)
        offset_x = params.get('offset_x', 0)
        
        # Parametry Detekcji
        canny_thresh = params.get('canny_thresh', 150)
        min_line_len = params.get('min_line_len', 100)
        eraser_thick = params.get('eraser_thick', 10)
        min_a = params.get('min_a', 10)
        max_a = params.get('max_a', 1000)
        closing_k = params.get('closing_k', 5)
        
        # Flaga fotometrii przekazywana z GUI
        enable_photometry = params.get('enable_photometry', False)

        if canny_thresh < 1: canny_thresh = 1
        if min_line_len < 1: min_line_len = 1
        if eraser_thick < 1: eraser_thick = 1
        if max_a <= min_a: max_a = min_a + 1

        frame_640 = cv2.resize(current_frame, (640, 480))
        h, w = 480, 640
        gray = cv2.cvtColor(frame_640, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.medianBlur(gray, 5)

        # Budowa maski ROI
        roi_top_y = int(h * (roi_top_perc / 100.0))
        top_w_pixels = int(w * (top_width_perc / 100.0))
        bottom_w_pixels = int(w * (bottom_width_perc / 100.0))
        center_x = (w // 2) + offset_x
        
        x_top_left = center_x - top_w_pixels // 2
        x_top_right = center_x + top_w_pixels // 2
        x_bottom_left = center_x - bottom_w_pixels // 2
        x_bottom_right = center_x + bottom_w_pixels // 2

        pts = np.array([[x_top_left, roi_top_y], [x_top_right, roi_top_y],
                        [x_bottom_right, h], [x_bottom_left, h]], np.int32)
        
        mask_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_roi, [pts], 255)

        edges_full = cv2.Canny(gray_blurred, canny_thresh // 2, canny_thresh)
        edges_roi = cv2.bitwise_and(edges_full, edges_full, mask=mask_roi)
        edges_cleaned = edges_roi.copy()

        lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, threshold=40, 
                                minLineLength=min_line_len, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edges_cleaned, (x1, y1), (x2, y2), 0, eraser_thick)

        edges_closed = edges_cleaned.copy()
        if closing_k > 0:
            if closing_k % 2 == 0: closing_k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_k, closing_k))
            edges_closed = cv2.morphologyEx(edges_cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # --- ZBIERANIE POPRAWNYCH DETEKCJI ---
        for cnt in contours:
            if len(cnt) >= 5: 
                area = cv2.contourArea(cnt)
                if min_a <= area <= max_a:
                    ellipse = cv2.fitEllipse(cnt)
                    (cx_float, cy_float), (a, b), angle = ellipse
                    cx, cy = int(cx_float), int(cy_float)
                    
                    x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
                    
                    # SZTUCZNY PADDING DLA METOD KLASYCZNYCH (łapanie poświaty)
                    PADDING = 15
                    x1_pad = max(0, x_rect - PADDING)
                    y1_pad = max(0, y_rect - PADDING)
                    x2_pad = min(w, x_rect + w_rect + PADDING)
                    y2_pad = min(h, y_rect + h_rect + PADDING)
                    
                    lamp_color = get_lamp_color(frame_640, cx, cy)
                    
                    valid_detections_info.append({
                        'ellipse': ellipse,
                        'cx': cx, 'cy': cy,
                        'x_rect': x_rect, 'y_rect': y_rect, 
                        'w_rect': w_rect, 'h_rect': h_rect,
                        'x1_pad': x1_pad, 'y1_pad': y1_pad,
                        'x2_pad': x2_pad, 'y2_pad': y2_pad,
                        'color_name': lamp_color,
                        'raw_br': 0, 'hi': 1.0 # Wartości domyślne
                    })

        # --- ANALIZA FOTOMETRYCZNA ---
        photometry_active = enable_photometry and len(valid_detections_info) >= 3
        
        if photometry_active:
            cys = []
            brs = []
            
            for det in valid_detections_info:
                # Wycinamy obszar z nałożonym paddingiem
                roi = gray[det['y1_pad']:det['y2_pad'], det['x1_pad']:det['x2_pad']]
                mask_br = roi > 150
                raw_br = np.sum(roi[mask_br])
                
                det['raw_br'] = raw_br
                cys.append(det['cy'])
                brs.append(raw_br)
                
            poly_coefs = np.polyfit(cys, brs, deg=2)
            ref_curve = np.poly1d(poly_coefs)
            
            for det in valid_detections_info:
                expected = ref_curve(det['cy'])
                if expected <= 0: expected = 1
                det['hi'] = det['raw_br'] / expected

        # --- WIZUALIZACJA I ZAPIS LOGÓW ---
        final_display = frame_640.copy()
        roi_only = cv2.bitwise_and(final_display, final_display, mask=mask_roi)
        inv_mask = cv2.bitwise_not(mask_roi)
        bg_dark = cv2.addWeighted(cv2.bitwise_and(final_display, final_display, mask=inv_mask), 0.3, np.zeros_like(frame_640), 0.7, 0)
        final_display = cv2.add(roi_only, bg_dark)
        cv2.polylines(final_display, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

        for det in valid_detections_info:
            color_ellipse = (0, 255, 0)
            warning_text = ""
            hi_log = ""
            br_log = ""
            
            if photometry_active:
                hi = det['hi']
                hi_log = round(hi, 2)
                br_log = det['raw_br']
                
                if hi <= 0.40:
                    color_ellipse = (0, 0, 255)
                    warning_text = f" [ERR: {int(hi*100)}%]"
                elif hi <= 0.80:
                    color_ellipse = (0, 255, 255)
                    warning_text = f" [WARN: {int(hi*100)}%]"

            # Rysowanie Elipsy na zadanym kolorze
            cv2.ellipse(final_display, det['ellipse'], color_ellipse, 2)
            cv2.circle(final_display, (det['cx'], det['cy']), 2, (0, 0, 255), -1)

            # Rysowanie testu
            text_color = (255, 255, 255) 
            if det['color_name'] == "Czerwony": 
                text_color = (0, 0, 255)
            elif det['color_name'] == "Zolty": 
                text_color = (0, 255, 255)
            elif det['color_name'] == "Zielony": 
                text_color = (0, 255, 0)
            
            display_str = f"{det['color_name']}{warning_text}"
            cv2.putText(final_display, display_str, (det['cx'] + 10, det['cy'] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

            # Dodanie logu do CSV (nowe pole hi_log i br_log dodane na końcu formatu)
            x2_csv = det['x_rect'] + det['w_rect']
            y2_csv = det['y_rect'] + det['h_rect']
            current_frame_detections.append([
                frame_counter, -1, det['color_name'], 1.0, 
                det['x_rect'], det['y_rect'], x2_csv, y2_csv, hi_log, br_log
            ])

        processing_time_ms = (time.time() - start_time) * 1000
        frame_times_dict[frame_counter] = [frame_counter, round(processing_time_ms, 2)]
        detection_data_dict[frame_counter] = current_frame_detections

        cv2.imshow("Wynik Koncowy (Canny + Elipsy)", final_display)

        delay = 30 if (is_video and is_playing) else 10
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == 32: 
            if is_video:
                is_playing = not is_playing

        if is_video and is_playing:
            ret, next_frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, current_frame = cap.read()
            else:
                current_frame = next_frame
            frame_counter += 1

    if is_video:
        cap.release()
    cv2.destroyAllWindows()

    frame_times = list(frame_times_dict.values())
    detection_data = []
    for dets in detection_data_dict.values():
        detection_data.extend(dets)
        
    return frame_times, detection_data