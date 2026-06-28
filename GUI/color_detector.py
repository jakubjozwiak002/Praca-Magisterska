import cv2
import numpy as np

def get_lamp_color(frame_bgr, cx, cy, inner_r=3, outer_r=12):
    """
    Rozpoznaje kolor lampy na podstawie sprawiedliwego głosowania pikseli w pierścieniu (flarze).
    """
    h, w = frame_bgr.shape[:2]
    
    # 1. Wycinamy kwadratowe ROI wokół środka lampy dla optymalizacji
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
    
    # 2. Tworzenie lokalnej maski pierścieniowej
    mask = np.zeros(roi_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (local_cx, local_cy), outer_r, 255, -1)
    cv2.circle(mask, (local_cx, local_cy), inner_r, 0, -1)
    
    s_pixels = roi_hsv[:, :, 1][mask == 255]
    h_pixels = roi_hsv[:, :, 0][mask == 255]
    
    if len(s_pixels) == 0:
        return "Inny"
        
    # --- SYSTEM GŁOSOWANIA PIKSELI ---
    
    # Krok A: Rygorystyczny próg bieli (Nasycenie S)
    # Wartość 50 oddziela czystą biel od wyblakłej flary kolorowej lampy.
    S_THRESH = 50 
    
    white_votes = np.sum(s_pixels < S_THRESH)
    
    # Krok B: Głosowanie kolorów (Tylko dla pikseli o nasyceniu >= S_THRESH)
    color_mask = s_pixels >= S_THRESH
    valid_h = h_pixels[color_mask]
    
    red_votes = np.sum((valid_h <= 8) | (valid_h >= 160))
    yellow_votes = np.sum((valid_h > 8) & (valid_h <= 30))
    green_votes = np.sum((valid_h > 30) & (valid_h <= 85))
    
    # Zbieranie głosów
    votes = {
        "Bialy": white_votes,
        "Czerwony": red_votes,
        "Zolty": yellow_votes,
        "Zielony": green_votes
    }
    
    # Wybór zwycięzcy
    best_color = max(votes, key=votes.get)
    max_votes = votes[best_color]
    
    # Zabezpieczenie: Jeśli dominujący kolor ma mniej niż 20% wszystkich pikseli
    # (np. lampa świeci na niebiesko albo jest to tło), odrzucamy klasyfikację.
    if max_votes < len(s_pixels) * 0.2:
        return "Inny"
        
    return best_color