import customtkinter as ctk
import threading
import psutil
import pynvml

class HomeFrame(ctk.CTkFrame):
    def __init__(self, master, engine, show_settings_callback):
        super().__init__(master)
        self.engine = engine

        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=10, pady=(10,0))

        self.status = "Chargement des modèles"
        
        # status
        self.label_status = ctk.CTkLabel(self, text=f"État : {self.status}", font=("Helvetica", 18))
        self.label_status.pack(pady=20)

        # bouton paramètres
        self.btn_settings = ctk.CTkButton(
            self.header_frame,
            text="⚙️",
            width=40,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10","#DCE4EE"),
            command=show_settings_callback
        )
        self.btn_settings.pack(side="right")


        # Bouton Principal
        self.btn_talk = ctk.CTkButton(self, text="PARLER (ESPACE)", command=self.start_conversation_thread)
        self.btn_talk.pack(pady=20)

        # Indicateur visuel
        self.indicator = ctk.CTkFrame(self, width=20, height=20, fg_color="red")
        self.indicator.pack(pady=10)

        self.answer_label = ctk.CTkLabel(self, text="", font=("Helvetica", 16))
        self.answer_label.pack(pady=10)

        self.monitoring_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.monitoring_frame.pack(pady=10, fill="x", padx=20)

        # RAM Système
        self.ram_label = ctk.CTkLabel(self.monitoring_frame, text="RAM: 0%")
        self.ram_label.pack()
        self.ram_bar = ctk.CTkProgressBar(self.monitoring_frame, width=300)
        self.ram_bar.pack(pady=5)

        # VRAM (GPU)
        self.vram_label = ctk.CTkLabel(self.monitoring_frame, text="VRAM: 0MB / 0MB")
        self.vram_label.pack()
        self.vram_bar = ctk.CTkProgressBar(self.monitoring_frame, width=300)
        self.vram_bar.pack(pady=5)

        # Init NVML
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_ready = True
        except:
            self.gpu_ready = False

        self.monitoring_active = True
        self.update_stats()

    def update_stats(self):
        # 1. RAM Système
        ram = psutil.virtual_memory()
        self.ram_label.configure(text=f"RAM Système: {ram.percent}%")
        self.ram_bar.set(ram.percent / 100)

        # 2. VRAM NVIDIA
        if self.gpu_ready:
            info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            used_gb = info.used / 1024**3
            total_gb = info.total / 1024**3
            percent = (info.used / info.total)
            
            self.vram_label.configure(text=f"VRAM: {used_gb:.2f}GB / {total_gb:.2f}GB")
            self.vram_bar.set(percent)
            
            # Alerte couleur si > 90%
            if percent > 0.9:
                self.vram_bar.configure(progress_color="red")
            else:
                self.vram_bar.configure(progress_color="#1f538d")

        # Rappeler cette fonction dans 1000ms (1 seconde)
        self.after(1000, self.update_stats)

    def update_status(self, new_status, color, response_text = ""):
        self.status = new_status
        self.label_status.configure(text=f"État : {self.status}")
        self.indicator.configure(fg_color=color)
        if response_text:
            self.answer_label.configure(text=response_text)

    def start_conversation_thread(self):
        # On lance la logique IA dans un thread séparé pour ne pas bloquer la fenêtre
        thread = threading.Thread(target=self.engine.run_pipeline, args=(self.update_status,))
        thread.start()

    def toggle_monitoring(self, is_visible):
        self.monitoring_active = is_visible
        if is_visible:
            self.monitoring_frame.pack(pady=10, fill="x", padx=20)
        else:
            self.monitoring_frame.pack_forget()