import cv2
import csv
import time
import itertools
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO

# Importujemy zewnętrzne metody klasyczne
from method_pp import run_method_pp
from method_canny import run_method_canny
from method_tophat import run_method_tophat

class AirportLightingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("System Detekcji Lamp Lotniskowych")
        self.root.geometry("600x780")
        self.root.eval('tk::PlaceWindow . center')

        # --- ZMIENNE AI ---
        self.model_path = tk.StringVar()
        self.ai_source_type = tk.IntVar(value=0) 
        self.ai_source_path = tk.StringVar()
        self.detection_data = []
        self.frame_times = [] # Lista przechowująca czas przetwarzania poszczególnych klatek
        
        # Zmienna włączająca algorytm fotometryczny (Współdzielona dla obu zakładek)
        self.enable_photometry = tk.BooleanVar(value=False)

        # --- ZMIENNE METOD KLASYCZNYCH ---
        self.class_source_type = tk.IntVar(value=0)
        self.class_source_path = tk.StringVar()
        
        # Zmienne suwaków ROI (Wspólne)
        self.roi_top = tk.IntVar(value=30)
        self.top_width = tk.IntVar(value=10)
        self.bottom_width = tk.IntVar(value=50)
        self.offset_x = tk.IntVar(value=100)
        
        # Zmienne Metody 2 (Dylatacja + Histogram)
        self.pp_thresh_v = tk.IntVar(value=250)
        self.pp_k_size = tk.IntVar(value=3)
        self.pp_min_a = tk.IntVar(value=10)
        self.pp_max_a = tk.IntVar(value=1000)
        self.pp_bright_ratio = tk.IntVar(value=20)

        # Zmienne Metody 1 (HoughLinesP + Elipsy)
        self.preset_var = tk.StringVar(value="Dzien")
        self.canny_thresh = tk.IntVar(value=255)
        self.canny_min_line = tk.IntVar(value=0)
        self.canny_eraser = tk.IntVar(value=20)
        self.canny_min_a = tk.IntVar(value=5)
        self.canny_max_a = tk.IntVar(value=1000)
        self.canny_closing = tk.IntVar(value=1)

        # Zmienne Metody 3 (SLD Top-Hat)
        self.tophat_k = tk.IntVar(value=9)
        self.tophat_thresh_v = tk.IntVar(value=125)
        self.tophat_min_a = tk.IntVar(value=5)
        self.tophat_max_a = tk.IntVar(value=1500)
        self.tophat_max_aspect = tk.IntVar(value=30)

        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Arial', 10, 'bold'), padding=[10, 5])
        
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.tab_ai = ttk.Frame(self.notebook)
        self.tab_classical = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_ai, text='Detekcja AI (YOLO)')
        self.notebook.add(self.tab_classical, text='Metody Klasyczne')

        self.build_ai_tab()
        self.build_classical_tab()

    # =========================================================================
    # BUDOWA ZAKŁADKI: AI (YOLO)
    # =========================================================================
    def build_ai_tab(self):
        tk.Label(self.tab_ai, text="1. Wybierz wyuczony model YOLO (.pt / .ncnn):", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        frame_model = tk.Frame(self.tab_ai)
        frame_model.pack()
        tk.Entry(frame_model, textvariable=self.model_path, width=45, state='readonly').pack(side=tk.LEFT, padx=5)
        tk.Button(frame_model, text="Przeglądaj", command=self.browse_model).pack(side=tk.LEFT)

        tk.Label(self.tab_ai, text="2. Wybierz źródło obrazu:", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        tk.Radiobutton(self.tab_ai, text="Kamera na żywo", variable=self.ai_source_type, value=0, command=self.toggle_ai_source_btn).pack()
        tk.Radiobutton(self.tab_ai, text="Plik wideo (.mp4, .avi)", variable=self.ai_source_type, value=1, command=self.toggle_ai_source_btn).pack()
        tk.Radiobutton(self.tab_ai, text="Pojedyncze zdjęcie (.jpg, .png)", variable=self.ai_source_type, value=2, command=self.toggle_ai_source_btn).pack()

        frame_source = tk.Frame(self.tab_ai)
        frame_source.pack(pady=2)
        self.entry_ai_source = tk.Entry(frame_source, textvariable=self.ai_source_path, width=45, state='readonly')
        self.entry_ai_source.pack(side=tk.LEFT, padx=5)
        self.btn_ai_source = tk.Button(frame_source, text="Przeglądaj", command=self.browse_ai_source, state=tk.DISABLED)
        self.btn_ai_source.pack(side=tk.LEFT)

        # --- SEKCJA ROI DLA YOLO ---
        tk.Label(self.tab_ai, text="3. Geometria ROI (Ograniczenie obszaru dla YOLO):", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        self.ai_roi_frame = tk.Frame(self.tab_ai)
        self.ai_roi_frame.pack(fill="x", padx=40)
        
        tk.Scale(self.ai_roi_frame, from_=0, to=100, orient="horizontal", label="Odcięcie Góry (%)", variable=self.roi_top).pack(fill="x")
        tk.Scale(self.ai_roi_frame, from_=0, to=100, orient="horizontal", label="Szerokość Góry (%)", variable=self.top_width).pack(fill="x")
        tk.Scale(self.ai_roi_frame, from_=0, to=100, orient="horizontal", label="Szerokość Dołu (%)", variable=self.bottom_width).pack(fill="x")
        tk.Scale(self.ai_roi_frame, from_=0, to=200, orient="horizontal", label="Przesunięcie X (100 = środek)", variable=self.offset_x).pack(fill="x")

        # --- FOTOMETRIA ---
        tk.Checkbutton(self.tab_ai, text="Włącz analizę fotometryczną (Wykrywanie uszkodzonych lamp)", 
                       variable=self.enable_photometry, font=("Arial", 9, "bold"), fg="#D84315").pack(pady=(10, 0))

        frame_buttons = tk.Frame(self.tab_ai)
        frame_buttons.pack(pady=15)

        tk.Button(frame_buttons, text="ROZPOCZNIJ DETEKCJĘ", font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", 
                  width=20, height=2, command=self.start_ai_inference).pack(side=tk.LEFT, padx=10)

        self.btn_export = tk.Button(frame_buttons, text="EKSPORTUJ .CSV", font=("Arial", 10, "bold"), bg="#2196F3", fg="white", 
                  width=15, height=2, command=self.export_csv, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=10)

    # =========================================================================
    # BUDOWA ZAKŁADKI: METODY KLASYCZNE
    # =========================================================================
    def build_classical_tab(self):
        tk.Label(self.tab_classical, text="1. Wybierz źródło obrazu dla klasyki:", font=("Arial", 10, "bold")).pack(pady=(10, 5))
        tk.Radiobutton(self.tab_classical, text="Kamera na żywo", variable=self.class_source_type, value=0, command=self.toggle_class_source_btn).pack()
        tk.Radiobutton(self.tab_classical, text="Plik wideo (.mp4, .avi)", variable=self.class_source_type, value=1, command=self.toggle_class_source_btn).pack()
        tk.Radiobutton(self.tab_classical, text="Pojedyncze zdjęcie (.jpg, .png)", variable=self.class_source_type, value=2, command=self.toggle_class_source_btn).pack()

        frame_source = tk.Frame(self.tab_classical)
        frame_source.pack(pady=5)
        tk.Entry(frame_source, textvariable=self.class_source_path, width=45, state='readonly').pack(side=tk.LEFT, padx=5)
        self.btn_class_source = tk.Button(frame_source, text="Przeglądaj", command=self.browse_class_source, state=tk.DISABLED)
        self.btn_class_source.pack(side=tk.LEFT)

        tk.Label(self.tab_classical, text="2. Geometria ROI:", font=("Arial", 10, "bold")).pack(pady=(5, 5))
        
        self.roi_frame = tk.Frame(self.tab_classical)
        self.roi_frame.pack(fill="x", padx=40)
        
        self.scale_roi_top = tk.Scale(self.roi_frame, from_=0, to=100, orient="horizontal", label="Odcięcie Góry (%)", variable=self.roi_top, state=tk.DISABLED)
        self.scale_roi_top.pack(fill="x")
        
        self.scale_top_width = tk.Scale(self.roi_frame, from_=0, to=100, orient="horizontal", label="Szerokość Góry (%)", variable=self.top_width, state=tk.DISABLED)
        self.scale_top_width.pack(fill="x")
        
        self.scale_bottom_width = tk.Scale(self.roi_frame, from_=0, to=100, orient="horizontal", label="Szerokość Dołu (%)", variable=self.bottom_width, state=tk.DISABLED)
        self.scale_bottom_width.pack(fill="x")
        
        self.scale_offset_x = tk.Scale(self.roi_frame, from_=0, to=200, orient="horizontal", label="Przesunięcie X (100 = środek)", variable=self.offset_x, state=tk.DISABLED)
        self.scale_offset_x.pack(fill="x")

        # --- FOTOMETRIA (Kopia do widoku klasycznego) ---
        tk.Checkbutton(self.tab_classical, text="Włącz analizę fotometryczną (Wykrywanie uszkodzonych lamp)", 
                       variable=self.enable_photometry, font=("Arial", 9, "bold"), fg="#D84315").pack(pady=(5, 0))

        # Selektor Presetów
        frame_presets = tk.Frame(self.tab_classical)
        frame_presets.pack(pady=(10, 0))
        tk.Label(frame_presets, text="Wybierz Preset (Dla Metody 1 i 3):", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(frame_presets, text="Dzień", variable=self.preset_var, value="Dzien", command=self.apply_preset_day, state=tk.DISABLED).pack(side=tk.LEFT)
        tk.Radiobutton(frame_presets, text="Noc", variable=self.preset_var, value="Noc", command=self.apply_preset_night, state=tk.DISABLED).pack(side=tk.LEFT)
        self.preset_radios = frame_presets.winfo_children()[1:]

        tk.Label(self.tab_classical, text="3. Uruchom algorytm:", font=("Arial", 10, "bold")).pack(pady=(5, 5))
        btn_width = 40
        
        self.btn_method_1 = ttk.Button(self.tab_classical, text="Metoda 1 (Canny + HoughLines + Elipsy)", width=btn_width, 
                   command=self.run_classical_method_1, state=tk.DISABLED)
        self.btn_method_1.pack(pady=3, ipady=3)
        
        self.btn_method_2 = ttk.Button(self.tab_classical, text="Metoda 2 (Dylatacja + Filtracja Histogramowa PP)", width=btn_width, 
                   command=self.run_classical_method_2, state=tk.DISABLED)
        self.btn_method_2.pack(pady=3, ipady=3)
        
        self.btn_method_3 = ttk.Button(self.tab_classical, text="Metoda 3 (Filtrowanie SLD Top-Hat)", width=btn_width, 
                   command=self.run_classical_method_3, state=tk.DISABLED)
        self.btn_method_3.pack(pady=3, ipady=3)

        self.btn_adv_params = ttk.Button(self.tab_classical, text="Dostosuj Parametry Detekcji", width=btn_width, 
                                         command=self.open_advanced_params_window, state=tk.DISABLED)
        self.btn_adv_params.pack(pady=(15, 5), ipady=3)

        self.btn_export_class = tk.Button(self.tab_classical, text="EKSPORTUJ .CSV", font=("Arial", 10, "bold"), 
                                          bg="#2196F3", fg="white", width=40, height=1, 
                                          command=self.export_csv, state=tk.DISABLED)
        self.btn_export_class.pack(pady=5)

    # =========================================================================
    # LOGIKA PRESETÓW (DZIEŃ / NOC)
    # =========================================================================
    def apply_preset_day(self):
        self.canny_thresh.set(255)
        self.canny_min_line.set(0)
        self.canny_eraser.set(20)
        self.canny_min_a.set(5)
        self.canny_max_a.set(1000)
        self.canny_closing.set(1)
        
        self.tophat_k.set(9)
        self.tophat_thresh_v.set(125)
        self.tophat_min_a.set(5)
        self.tophat_max_a.set(1500)
        self.tophat_max_aspect.set(30)

    def apply_preset_night(self):
        self.canny_thresh.set(150)
        self.canny_min_line.set(100)
        self.canny_eraser.set(10)
        self.canny_min_a.set(10)
        self.canny_max_a.set(1000)
        self.canny_closing.set(5)
        
        self.tophat_k.set(50)
        self.tophat_thresh_v.set(125)
        self.tophat_min_a.set(5)
        self.tophat_max_a.set(1500)
        self.tophat_max_aspect.set(30)

    # =========================================================================
    # OKNO PARAMETRÓW ZAAWANSOWANYCH (Z PODZIAŁEM NA METODY)
    # =========================================================================
    def open_advanced_params_window(self):
        if hasattr(self, 'adv_window') and self.adv_window.winfo_exists():
            self.adv_window.lift()
            return

        self.adv_window = tk.Toplevel(self.root)
        self.adv_window.title("Zaawansowane Parametry Detekcji")
        self.adv_window.geometry("450x480")
        self.adv_window.grab_set() 
        
        tk.Label(self.adv_window, text="Zmiany od razu wpływają na odpalony algorytm", font=("Arial", 9, "italic")).pack(pady=10)

        adv_notebook = ttk.Notebook(self.adv_window)
        adv_notebook.pack(expand=True, fill='both', padx=10, pady=5)

        tab_m1 = ttk.Frame(adv_notebook)
        tab_m2 = ttk.Frame(adv_notebook)
        tab_m3 = ttk.Frame(adv_notebook)

        adv_notebook.add(tab_m1, text="Metoda 1 (Canny)")
        adv_notebook.add(tab_m2, text="Metoda 2 (PP)")
        adv_notebook.add(tab_m3, text="Metoda 3 (Top-Hat)")

        # Suwaki dla Metody 1 (Canny)
        tk.Scale(tab_m1, from_=1, to=255, orient="horizontal", label="Próg Canny", variable=self.canny_thresh).pack(fill="x", padx=20)
        tk.Scale(tab_m1, from_=0, to=200, orient="horizontal", label="Min Długość Linii", variable=self.canny_min_line).pack(fill="x", padx=20)
        tk.Scale(tab_m1, from_=1, to=30, orient="horizontal", label="Grubość Gumki", variable=self.canny_eraser).pack(fill="x", padx=20)
        tk.Scale(tab_m1, from_=1, to=500, orient="horizontal", label="Min Pole", variable=self.canny_min_a).pack(fill="x", padx=20)
        tk.Scale(tab_m1, from_=10, to=2000, orient="horizontal", label="Max Pole", variable=self.canny_max_a).pack(fill="x", padx=20)
        tk.Scale(tab_m1, from_=0, to=20, orient="horizontal", label="Zamykanie Morfologiczne", variable=self.canny_closing).pack(fill="x", padx=20)

        # Suwaki dla Metody 2 (PP)
        tk.Scale(tab_m2, from_=0, to=255, orient="horizontal", label="Próg Binaryzacji (thresh_v)", variable=self.pp_thresh_v).pack(fill="x", padx=20)
        tk.Scale(tab_m2, from_=1, to=15, orient="horizontal", label="Rozmiar Jądra Dylatacji (k_size)", variable=self.pp_k_size).pack(fill="x", padx=20)
        tk.Scale(tab_m2, from_=1, to=500, orient="horizontal", label="Minimalne Pole (min_a)", variable=self.pp_min_a).pack(fill="x", padx=20)
        tk.Scale(tab_m2, from_=10, to=2000, orient="horizontal", label="Maksymalne Pole (max_a)", variable=self.pp_max_a).pack(fill="x", padx=20)
        tk.Scale(tab_m2, from_=0, to=100, orient="horizontal", label="Min % Jasnych Pikseli", variable=self.pp_bright_ratio).pack(fill="x", padx=20)

        # Suwaki dla Metody 3 (Top-Hat)
        tk.Scale(tab_m3, from_=1, to=100, orient="horizontal", label="Rozmiar Jądra (Top-Hat)", variable=self.tophat_k).pack(fill="x", padx=20)
        tk.Scale(tab_m3, from_=0, to=255, orient="horizontal", label="Próg Binaryzacji", variable=self.tophat_thresh_v).pack(fill="x", padx=20)
        tk.Scale(tab_m3, from_=1, to=500, orient="horizontal", label="Min Pole", variable=self.tophat_min_a).pack(fill="x", padx=20)
        tk.Scale(tab_m3, from_=10, to=2000, orient="horizontal", label="Max Pole", variable=self.tophat_max_a).pack(fill="x", padx=20)
        tk.Scale(tab_m3, from_=10, to=100, orient="horizontal", label="Max Proporcja Boku (x10)", variable=self.tophat_max_aspect).pack(fill="x", padx=20)

    # =========================================================================
    # LOGIKA INTERFEJSU
    # =========================================================================
    def browse_model(self):
        path = filedialog.askopenfilename(title="Wybierz plik modelu", filetypes=[("Modele YOLO", "*.pt *.ncnn"), ("Wszystkie pliki", "*.*")])
        if path: self.model_path.set(path)

    def browse_ai_source(self):
        filetypes = [("Wideo", "*.mp4 *.avi *.mov")] if self.ai_source_type.get() == 1 else [("Zdjecia", "*.jpg *.png *.jpeg *.bmp")]
        path = filedialog.askopenfilename(title="Wybierz plik", filetypes=filetypes)
        if path: self.ai_source_path.set(path)

    def browse_class_source(self):
        filetypes = [("Wideo", "*.mp4 *.avi *.mov")] if self.class_source_type.get() == 1 else [("Zdjecia", "*.jpg *.png *.jpeg *.bmp")]
        path = filedialog.askopenfilename(title="Wybierz plik", filetypes=filetypes)
        if path: 
            self.class_source_path.set(path)
            self.enable_classical_controls()

    def toggle_ai_source_btn(self):
        if self.ai_source_type.get() in [1, 2]:
            self.btn_ai_source.config(state=tk.NORMAL)
        else:
            self.btn_ai_source.config(state=tk.DISABLED)
            self.ai_source_path.set("")

    def toggle_class_source_btn(self):
        if self.class_source_type.get() in [1, 2]:
            self.btn_class_source.config(state=tk.NORMAL)
            self.disable_classical_controls()
        else:
            self.btn_class_source.config(state=tk.DISABLED)
            self.class_source_path.set("")
            self.enable_classical_controls()

    def enable_classical_controls(self):
        self.scale_roi_top.config(state=tk.NORMAL)
        self.scale_top_width.config(state=tk.NORMAL)
        self.scale_bottom_width.config(state=tk.NORMAL)
        self.scale_offset_x.config(state=tk.NORMAL)
        self.btn_method_1.config(state=tk.NORMAL)
        self.btn_method_2.config(state=tk.NORMAL)
        self.btn_method_3.config(state=tk.NORMAL)
        self.btn_adv_params.config(state=tk.NORMAL)
        for rb in self.preset_radios:
            rb.config(state=tk.NORMAL)

    def disable_classical_controls(self):
        self.scale_roi_top.config(state=tk.DISABLED)
        self.scale_top_width.config(state=tk.DISABLED)
        self.scale_bottom_width.config(state=tk.DISABLED)
        self.scale_offset_x.config(state=tk.DISABLED)
        self.btn_method_1.config(state=tk.DISABLED)
        self.btn_method_2.config(state=tk.DISABLED)
        self.btn_method_3.config(state=tk.DISABLED)
        self.btn_adv_params.config(state=tk.DISABLED)
        for rb in self.preset_radios:
            rb.config(state=tk.DISABLED)

    # --- SŁOWNIKI DANYCH DLA METOD ZEWNĘTRZNYCH ---
    def get_canny_params(self):
        return {
            'roi_top': self.roi_top.get(),
            'top_width': self.top_width.get(),
            'bottom_width': self.bottom_width.get(),
            'offset_x': self.offset_x.get() - 100, 
            'canny_thresh': self.canny_thresh.get(),
            'min_line_len': self.canny_min_line.get(),
            'eraser_thick': self.canny_eraser.get(),
            'min_a': self.canny_min_a.get(),
            'max_a': self.canny_max_a.get(),
            'closing_k': self.canny_closing.get(),
            'enable_photometry': self.enable_photometry.get()
        }

    def get_pp_params(self):
        return {
            'roi_top': self.roi_top.get(),
            'top_width': self.top_width.get(),
            'bottom_width': self.bottom_width.get(),
            'offset_x': self.offset_x.get() - 100, 
            'thresh_v': self.pp_thresh_v.get(),
            'k_size': self.pp_k_size.get(),
            'min_a': self.pp_min_a.get(),
            'max_a': self.pp_max_a.get(),
            'bright_ratio': self.pp_bright_ratio.get(),
            'enable_photometry': self.enable_photometry.get()
        }

    def get_tophat_params(self):
        return {
            'roi_top': self.roi_top.get(),
            'top_width': self.top_width.get(),
            'bottom_width': self.bottom_width.get(),
            'offset_x': self.offset_x.get() - 100,
            'tophat_k': self.tophat_k.get(),
            'thresh_val': self.tophat_thresh_v.get(),
            'min_a': self.tophat_min_a.get(),
            'max_a': self.tophat_max_a.get(),
            'max_aspect': self.tophat_max_aspect.get(),
            'enable_photometry': self.enable_photometry.get()
        }

    # =========================================================================
    # SILNIK AI (YOLO)
    # =========================================================================
    def apply_roi(self, frame):
        roi_top_perc = self.roi_top.get()
        top_width_perc = self.top_width.get()
        bottom_width_perc = self.bottom_width.get()
        offset_x = self.offset_x.get() - 100

        frame_640 = cv2.resize(frame, (640, 480))
        h, w = 480, 640
        
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
        
        return frame_640, pts, mask_roi

    def start_ai_inference(self):
        if not self.model_path.get():
            messagebox.showerror("Błąd", "Wybierz plik modelu przed startem!")
            return
        if self.ai_source_type.get() in [1, 2] and not self.ai_source_path.get():
            messagebox.showerror("Błąd", "Wybrano opcję pliku, ale nie wskazano ścieżki!")
            return

        source = self.ai_source_path.get() if self.ai_source_type.get() in [1, 2] else 0
        self.run_yolo(self.model_path.get(), source, self.ai_source_type.get())
        
        if self.frame_times or self.detection_data:
            self.btn_export.config(state=tk.NORMAL)

    def run_yolo(self, model_path, source, source_type):
        self.detection_data = []
        self.frame_times = [] 
        model = YOLO(model_path)
        
        is_video = source_type in [0, 1]
        
        frame_times_dict = {}
        detection_data_dict = {}
        frame_counter = 1

        if is_video:
            cap = cv2.VideoCapture(source)
            ret, current_frame = cap.read()
            if not ret:
                messagebox.showerror("Błąd", "Nie udało się otworzyć strumienia.")
                return
            is_playing = False
        else:
            current_frame = cv2.imread(source)
            if current_frame is None:
                messagebox.showerror("Błąd", "Nie udało się wczytać zdjęcia.")
                return
            is_playing = False

        cv2.namedWindow("Wizualizacja YOLO", cv2.WINDOW_NORMAL)
        target_fps = 30
        target_frame_time = 1.0 / target_fps

        while True:
            self.root.update()

            start_time = time.time()
            current_frame_detections = []
            valid_detections_info = []

            frame_640, pts, mask_roi = self.apply_roi(current_frame)

            if is_video and is_playing:
                results = model.track(source=frame_640, conf=0.4, imgsz=640, persist=True, tracker="bytetrack.yaml", verbose=False)
            else:
                results = model.predict(source=frame_640, conf=0.4, imgsz=640, verbose=False)
            
            # --- ZBIERANIE POPRAWNYCH DETEKCJI ---
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cx = max(0, min(cx, 639))
                cy = max(0, min(cy, 479))
                
                # Tylko detekcje z obszaru ROI
                if mask_roi[cy, cx] == 255:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])           
                    cls_name = model.names[cls_id]     
                    track_id = int(box.id[0]) if box.id is not None else -1
                    
                    valid_detections_info.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'cx': cx, 'cy': cy, 'conf': conf,
                        'cls_name': cls_name, 'track_id': track_id,
                        'raw_br': 0, 'hi': 1.0 # Domyślne wartości
                    })

            # --- ANALIZA FOTOMETRYCZNA (Jeśli włączona i >=3 lampy) ---
            photometry_active = self.enable_photometry.get() and len(valid_detections_info) >= 3
            
            if photometry_active:
                gray_img = cv2.cvtColor(frame_640, cv2.COLOR_BGR2GRAY)
                cys = []
                brs = []
                
                for det in valid_detections_info:
                    x1, y1 = max(0, det['x1']), max(0, det['y1'])
                    x2, y2 = min(639, det['x2']), min(479, det['y2'])
                    
                    roi = gray_img[y1:y2, x1:x2]
                    mask_br = roi > 150
                    raw_br = np.sum(roi[mask_br])
                    det['raw_br'] = raw_br
                    cys.append(det['cy'])
                    brs.append(raw_br)
                    
                # Dopasowanie krzywej referencyjnej
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
                color = (0, 255, 0)
                warning_text = ""
                hi_log = ""
                br_log = ""
                
                if photometry_active:
                    hi = det['hi']
                    hi_log = round(hi, 2)
                    br_log = det['raw_br']
                    
                    # Dodajemy opisy tylko dla uszkodzonych lamp
                    if hi <= 0.40:
                        color = (0, 0, 255)
                        warning_text = f" [ERR: {int(hi*100)}%]"
                    elif hi <= 0.80:
                        color = (0, 255, 255)
                        warning_text = f" [WARN: {int(hi*100)}%]"

                label = f"ID:{det['track_id']} {det['cls_name']} {det['conf']:.2f}{warning_text}" if det['track_id'] != -1 else f"{det['cls_name']} {det['conf']:.2f}{warning_text}"
                
                # Zapis do CSV
                current_frame_detections.append([
                    frame_counter, det['track_id'], det['cls_name'], round(det['conf'], 3), 
                    det['x1'], det['y1'], det['x2'], det['y2'], hi_log, br_log
                ])
                
                cv2.rectangle(final_display, (det['x1'], det['y1']), (det['x2'], det['y2']), color, 1)
                cv2.putText(final_display, label, (det['x2'] + 5, det['y1'] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            processing_time_ms = (time.time() - start_time) * 1000 
            frame_times_dict[frame_counter] = [frame_counter, round(processing_time_ms, 2)]
            detection_data_dict[frame_counter] = current_frame_detections

            cv2.imshow("Wizualizacja YOLO", final_display)
            
            if is_video and is_playing:
                elapsed_total = time.time() - start_time
                time_to_wait = target_frame_time - elapsed_total
                delay = max(1, int(time_to_wait * 1000))
            else:
                delay = 10
                
            key = cv2.waitKey(delay) & 0xFF

            if key == ord('q'):
                break
            elif key == 32: 
                if is_video:
                    is_playing = not is_playing

            if is_video and is_playing:
                ret, next_frame = cap.read()
                if not ret:
                    if source_type == 1: 
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        _, current_frame = cap.read()
                    else:
                        break 
                else:
                    current_frame = next_frame
                frame_counter += 1

        if is_video:
            cap.release()
        cv2.destroyAllWindows()

        self.frame_times = list(frame_times_dict.values())
        self.detection_data = []
        for dets in detection_data_dict.values():
            self.detection_data.extend(dets)

    def export_csv(self):
        save_path = filedialog.asksaveasfilename(
            title="Zapisz dane detekcji", defaultextension=".csv",
            filetypes=[("Pliki CSV", "*.csv"), ("Wszystkie pliki", "*.*")]
        )
        if save_path:
            try:
                with open(save_path, mode='w', newline='') as file:
                    writer = csv.writer(file, delimiter=';')
                    
                    if self.frame_times:
                        avg_time = sum(row[1] for row in self.frame_times) / len(self.frame_times)
                        writer.writerow(["Sredni czas przetwarzania klatki [ms]:", round(avg_time, 2)])
                        writer.writerow([]) 
                        
                    writer.writerow([
                        "WYDAJNOSC_Klatka", "Czas_ms", "", 
                        "DETEKCJA_Klatka", "ID_Obiektu", "Klasa", "Pewnosc", "X1", "Y1", "X2", "Y2", "Health_Index", "Raw_Brightness"
                    ])
                    
                    for time_row, det_row in itertools.zip_longest(self.frame_times, self.detection_data, fillvalue=None):
                        t_data = time_row if time_row is not None else ["", ""]
                        d_data = det_row if det_row is not None else ["", "", "", "", "", "", "", "", "", ""]
                        combined_row = list(t_data) + [""] + list(d_data)
                        writer.writerow(combined_row)
                    
                messagebox.showinfo("Sukces", f"Pomyślnie zapisano raport do pliku CSV.")
            except Exception as e:
                messagebox.showerror("Błąd Zapisu", f"Wystąpił błąd:\n{e}")

    # =========================================================================
    # SILNIK KLASYCZNY
    # =========================================================================
    def run_classical_method_1(self):
        source = self.class_source_path.get() if self.class_source_type.get() in [1, 2] else 0
        source_type = self.class_source_type.get()
        self.frame_times, self.detection_data = run_method_canny(source, source_type, self.get_canny_params, self.root)
        if self.frame_times or self.detection_data:
            self.btn_export_class.config(state=tk.NORMAL) 

    def run_classical_method_2(self):
        source = self.class_source_path.get() if self.class_source_type.get() in [1, 2] else 0
        source_type = self.class_source_type.get()
        self.frame_times, self.detection_data = run_method_pp(source, source_type, self.get_pp_params, self.root)
        if self.frame_times or self.detection_data:
            self.btn_export_class.config(state=tk.NORMAL)

    def run_classical_method_3(self):
        source = self.class_source_path.get() if self.class_source_type.get() in [1, 2] else 0
        source_type = self.class_source_type.get()
        self.frame_times, self.detection_data = run_method_tophat(source, source_type, self.get_tophat_params, self.root)
        if self.frame_times or self.detection_data:
            self.btn_export_class.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = AirportLightingApp(root)
    root.mainloop()