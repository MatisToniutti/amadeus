import customtkinter as ctk
from src.ui.frames.home import HomeFrame
from src.ui.frames.settings import SettingsFrame


class LawAssistantGUI(ctk.CTk):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

        self.title("Amadeus")
        self.geometry("400x600")
        #la fenêtre reste devant
        self.attributes('-topmost', True)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.home_frame = HomeFrame(
            self,
            engine=engine,
            show_settings_callback=self.show_settings
        )
        
        self.settings_frame = SettingsFrame(
            self,
            go_back_callback=self.show_home,
            toggle_monitor_callback=self.home_frame.toggle_monitoring,
            change_volume_callback = self.engine.set_volume,
            reset_history_callback = self.engine.reset_history
        )
        
        # Bind de la touche Espace
        self.bind('<space>', lambda event: self.start_conversation_thread())

        self.show_home()

    def update_status(self, status_text, color):
        """pont entre le main et update_status de home"""
        # On vérifie que la home_frame existe bien avant de l'appeler
        if hasattr(self, 'home_frame'):
            # On passe None pour le texte de réponse car ici on met juste à jour le statut
            self.home_frame.update_status(status_text, color)

    def show_home(self):
        self.settings_frame.grid_forget()
        self.home_frame.grid(row=0, column=0, sticky="nsew")

    def show_settings(self):
        self.home_frame.grid_forget()
        self.settings_frame.grid(row=0, column=0, sticky="nsew")



    