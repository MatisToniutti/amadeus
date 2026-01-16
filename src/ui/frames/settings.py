import customtkinter as ctk

class SettingsFrame(ctk.CTkFrame):
    def __init__(self,
                master,
                go_back_callback,
                toggle_monitor_callback,
                change_volume_callback,
                reset_history_callback,
                current_model,
                model_list,
                change_TG_model_callback):
        super().__init__(master)
        self.change_volume_callback = change_volume_callback
        self.reset_history_callback = reset_history_callback

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

        # Option 2 : Volume
        self.row2 = ctk.CTkFrame(self, fg_color="transparent")
        self.row2.pack(fill="x", padx=20, pady=10)

        self.lbl_vol = ctk.CTkLabel(self.row2,
                                    text="Volume",
                                    font=("Helvetica",14))
        self.lbl_vol.pack(side="left")

        self.lbl_vol_value = ctk.CTkLabel(self.row2, text="50%", width=40)
        self.lbl_vol_value.pack(side="right", padx=(10, 0))

        self.slider_vol = ctk.CTkSlider(
            self.row2,
            from_=0,
            to=1,
            number_of_steps=100,
            command=self.update_volume_event
        )
        self.slider_vol.set(0.5)
        self.slider_vol.pack(side="right")

        # Titre de section modèles dans les paramètres
        self.lbl_section_models = ctk.CTkLabel(
            self, 
            text="Modèles", 
            font=("Helvetica", 16, "bold")
        )
        self.lbl_section_models.pack(pady=(20, 0))

        # Option 3 : Reset de l'historique
        self.row4 = ctk.CTkFrame(self, fg_color="transparent")
        self.row4.pack(fill="x", padx=20, pady=20)

        self.lbl_TG_model = ctk.CTkLabel(self.row4,
                                      text="LLM",
                                      font=("Helvetica",14))
        self.lbl_TG_model.pack(side="left")
        self.combo_TG_model = ctk.CTkOptionMenu(
            self.row4,
            values=model_list,
            command=change_TG_model_callback
        )
        self.combo_TG_model.set(current_model)
        self.combo_TG_model.pack(side="right")

        self.row5 = ctk.CTkFrame(self, fg_color="transparent")
        self.row5.pack(fill="x", padx=20, pady=20)

        self.btn_reset = ctk.CTkButton(
            self.row5,
            text = "Rénitialiser l'historique",
            fg_color="#A83232",
            hover_color="#7A2424",
            command=reset_history_callback
        )
        self.btn_reset.pack(fill="x") # prend toute la largeur

    def update_volume_event(self, volume):
        self.lbl_vol_value.configure(text=f"{int(volume * 100)}%")
        self.change_volume_callback(volume*200)