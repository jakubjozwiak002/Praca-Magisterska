from ultralytics import YOLO

# Obowiązkowe zabezpieczenie dla systemu Windows!
if __name__ == '__main__':
    
    # 1. Wczytanie modelu last.pt z przerwanego treningu YOLO26
    # Upewnij się, że ścieżka do pliku last.pt jest poprawna dla Twojego komputera
    model = YOLO('runs/detect/Magisterka/yolo26s_colours1/weights/last.pt')
    
    # 2. Wznowienie treningu. 
    # Parametr batch=2 może tu pomóc zapobiec ponownemu wyrzuceniu błędu pamięci
    results = model.train(resume=True)