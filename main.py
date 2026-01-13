import threading
from src.engine import Engine
from src.ui.main_window import LawAssistantGUI

def main():
    engine = Engine()
    app = LawAssistantGUI(engine=engine)

    def loading_task():
        engine.load_all_models()
        app.update_status("Prêt", "green")

    loader = threading.Thread(target = loading_task)
    loader.start()

    print("lancement de l'interface")
    app.mainloop() 

if __name__ == "__main__":
    main()