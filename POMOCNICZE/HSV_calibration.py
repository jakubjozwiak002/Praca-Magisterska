import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def create_ring_mask(gray_img, inner_radius=3, outer_radius=15):
    """
    Znajduje najjaśniejszy punkt (rdzeń lampy) i tworzy maskę w kształcie pierścienia
    wokół niego, aby przechwycić flarę koloru, omijając prześwietlony biały środek.
    """
    # Znalezienie najjaśniejszego punktu na obrazie (zakładamy, że to środek lampy)
    _, _, _, max_loc = cv2.minMaxLoc(gray_img)
    center_x, center_y = max_loc

    mask = np.zeros_like(gray_img)
    # Rysujemy zewnętrzne koło (wypełnione na biało)
    cv2.circle(mask, (center_x, center_y), outer_radius, 255, -1)
    # Wygumkowujemy wewnętrzne koło (na czarno), tworząc pierścień
    cv2.circle(mask, (center_x, center_y), inner_radius, 0, -1)
    
    return mask

def main():
    base_dir = "dataset_kalibracja"
    colors = ["green", "yellow", "red", "white"]
    
    # Słowniki na zebrane wartości pikseli z wszystkich zdjęć dla danej klasy
    collected_data = {
        "green": [],
        "yellow": [],
        "red": [],
        "white": [] # Tutaj będziemy zbierać Nasycenie (S), a nie Odcień (H)
    }

    if not os.path.exists(base_dir):
        print(f"Błąd: Nie znaleziono folderu '{base_dir}'.")
        return

    print("Rozpoczynam analize flary lamp...")

    for color in colors:
        folder_path = os.path.join(base_dir, color)
        if not os.path.exists(folder_path):
            print(f"Brak folderu dla koloru: {color}, pomijam.")
            continue
            
        images = glob.glob(os.path.join(folder_path, "*.*"))
        if not images:
            continue
            
        print(f"Przetwarzanie klasy '{color}' ({len(images)} zdjec)...")
        
        for img_path in images:
            img = cv2.imread(img_path)
            if img is None: continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Tworzymy maskę pierścieniową (pomijamy środek 3px, bierzemy do 15px)
            ring_mask = create_ring_mask(gray, inner_radius=3, outer_radius=15)
            
            # Wyciągamy wartości pikseli, które pokrywają się z maską
            if color == "white":
                # Dla bieli interesuje nas kanał S (indeks 1)
                pixels = hsv[:, :, 1][ring_mask == 255]
            else:
                # Dla kolorowych interesuje nas kanał H (indeks 0)
                pixels = hsv[:, :, 0][ring_mask == 255]
                
            collected_data[color].extend(pixels)

    # --- ANALIZA STATYSTYCZNA I RYSOWANIE WYKRESÓW ---
    plt.figure(figsize=(15, 10))
    
    plot_idx = 1
    for color in colors:
        data = np.array(collected_data[color])
        if len(data) == 0:
            continue
            
        plt.subplot(2, 2, plot_idx)
        
        if color == "red":
            # Korekta zawijania skali dla czerwonego (wartości > 160 traktujemy jako ujemne)
            data_shifted = np.where(data > 90, data - 180, data)
            percentile_5 = np.percentile(data_shifted, 5)
            percentile_95 = np.percentile(data_shifted, 95)
            
            # Przywrócenie do formatu OpenCV (modulo 180)
            p5_real = int(percentile_5) % 180
            p95_real = int(percentile_95) % 180
            
            plt.hist(data_shifted, bins=40, color='red', alpha=0.7)
            plt.title(f"Czerwony (Kanal H) - Skompensowany\nZakres 90%: {p5_real} lub {p95_real}")
            plt.xlabel("Przesunięty Hue (-20 do 20)")
            
        elif color == "white":
            # Analiza Nasycenia (S) dla bieli
            percentile_95 = int(np.percentile(data, 95)) # Interesuje nas górny limit nasycenia
            
            plt.hist(data, bins=40, color='gray', alpha=0.7)
            plt.title(f"Bialy (Kanal S - Nasycenie)\nZalecany prog S < {percentile_95}")
            plt.xlabel("Nasycenie (0-255)")
            
        else:
            # Standardowa analiza dla zielonego i żółtego
            percentile_5 = int(np.percentile(data, 5))
            percentile_95 = int(np.percentile(data, 95))
            plot_color = 'green' if color == "green" else 'yellow'
            
            plt.hist(data, bins=40, color=plot_color, alpha=0.7)
            plt.title(f"{color.capitalize()} (Kanal H)\nZalecany przedzial: {percentile_5} - {percentile_95}")
            plt.xlabel("Hue (0-179)")
            
        plt.ylabel("Liczba pikseli")
        plot_idx += 1

    plt.tight_layout()
    plt.savefig("histograms_calibration.png")
    print("\nAnaliza zakonczona. Wygenerowano wykresy w pliku 'histograms_calibration.png'.")
    plt.show()

if __name__ == "__main__":
    main()