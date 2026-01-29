import threading
from src.engine import Engine
from src.ui.main_ui import LawAssistantGUI

def main():
    engine = Engine()
    app = LawAssistantGUI(engine=engine)

    def loading_task():
        engine.start_base_services()
        app.update_status("Prêt", "green")

    loader = threading.Thread(target = loading_task)
    loader.start()

    print("lancement de l'interface")
    app.mainloop() 

if __name__ == "__main__":
    main()