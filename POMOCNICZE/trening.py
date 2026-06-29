from ultralytics import YOLO

if __name__ == "__main__":

    # Wczytanie modelu wariantu "Small"
    model = YOLO('yolo26s.pt')

    results = model.train(
        data='DB_yolo26/data.yaml',
        epochs=150,                 # Długość treningu w epokach
        imgsz=1024,                 # Rozdzielczość obrazu - 1024 ze względu na małe obiekty
        batch=4,                    # Zmniejszony batch ze względu na ograniczenie pamięci VRAM
        device=0,                   # Wymuszenie użycia GPU
        save_period=10,             # Checkpoint co 10 epok
        patience=20,                # Early stopping po 20 epokach bez poprawy
    
        # -- Optymalizacja pod Small Object Detection (SOD) --
        mosaic=1.0,                 # 100% szans na uzycie Mosaic
        close_mosaic=15,            # Wyłącz mosaic na ostatnie 15 epok dla stabilizacji
        copy_paste=0.3,             # 30% szans na Copy-Paste
        mixup=0.1,                  # Lekki mixup dla uodpornienia na tło
    
        # -- Strojenie Hiperparametrów (Losses) --
        box=10.0,                   # Mocniejsza kara za błędy w lokalizacji (box loss)
        optimizer='auto',           
    
        project='Magisterka',
        name='yolo26s_colours1'
    )