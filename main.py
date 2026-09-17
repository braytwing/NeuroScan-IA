import tkinter as tk
import sys
import os

# Asegurar que el directorio raíz está en el PATH de Python
# Esto evita el error "ModuleNotFoundError" al importar tus carpetas
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importar la interfaz principal desde nuestro módulo UI
from ui.app_window import NeuroScanApp

def main():
    print("Iniciando NeuroScan AI - Entorno Clínico...")
    
    # 1. Crear la ventana principal (Root)
    root = tk.Tk()
    
    # 2. Configurar dimensiones mínimas para evitar que la UI se colapse
    root.minsize(1024, 768)
    
    # 3. Forzar el enfoque en la ventana al iniciar
    root.focus_force()
    
    # 4. Instanciar nuestra aplicación
    app = NeuroScanApp(root)
    
    # 5. Capturar el evento de cierre de ventana (la 'X' roja)
    # Esto asegura que la cámara web y la base de datos se cierren correctamente
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # 6. Iniciar el bucle de eventos (Main Loop)
    root.mainloop()

if __name__ == "__main__":
    main()