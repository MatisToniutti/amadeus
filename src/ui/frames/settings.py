import customtkinter as ctk

class SettingsFrame(ctk.CTkFrame):
    def __init__(self, master, go_back_callback, toggle_monitor_callback):
        super().__init__(master)

        #Bouton retour
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", padx=10, pady=10)

        self.btn_back = ctk.CTkButton(
            self.header,
            text="<- Retour",
            width=80,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "#DCE4EE"),
            command = go_back_callback
        )
        self.btn_back.pack(side="left")

        self.title = ctk.CTkLabel(self.header,
                                   text="Paramètres",
                                   font=("Helvetica", 18, "bold"))
        self.title.pack(side="left", padx=20)

        # Option 1 : Display monitoring
        self.row1 = ctk.CTkFrame(self, fg_color="transparent")
        self.row1.pack(fill="x", padx=20, pady=20)

        self.lbl_monitor = ctk.CTkLabel(self.row1, 
                                        text="Afficher l'utilisation du système",
                                        font=("Helvetica",14))
        self.lbl_monitor.pack(side="left")

        self.switch_monitor = ctk.CTkSwitch(
            self.row1,
            text="",
            command=lambda: toggle_monitor_callback(self.switch_monitor.get())
        )
        #pour qu'il soit activé de base
        self.switch_monitor.select()
        self.switch_monitor.pack(side="right")