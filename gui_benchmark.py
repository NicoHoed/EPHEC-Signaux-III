"""
--------------------------------------------------------------------------------
File: gui_benchmark.py
Author:
    Nicolas HOEDENAEKEN
    Théo MERTENS
    Baris OZCELIK
    Khassan AKTAMIROV
    
Description: 
    Interface graphique (GUI) basée sur CustomTkinter.
    Permet de lancer une analyse de masse (Benchmark) sur les images.
    Compare le layout détecté avec le nom du fichier (Vérité Terrain) 
    pour calculer un taux de précision (Accuracy) en temps réel.
--------------------------------------------------------------------------------
"""

import os
import threading
import time
import cv2
import easyocr
import customtkinter as ctk
from PIL import Image

from src.preprocessing import get_processed_images
from src.engine import run_ocr_pipeline, cluster_rows, score_layout

# Config du GUI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class BenchmarkApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Config de la fenetre
        self.title("Benchmark Clavier - Auto Validation")
        self.geometry("1100x700")
        
        # Variables d'état
        self.folder_path = "data/inputs" # Dossier par défaut
        self.is_running = False
        self.ocr_reader = None
        
        # Stats
        self.total_processed = 0
        self.success_count = 0
        self.fail_count = 0

        # --- LAYOUT PRINCIPAL ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. PANNEAU LATÉRAL (Contrôles & Stats)
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        ctk.CTkLabel(self.sidebar, text="KeyDetect", font=("Roboto", 20, "bold")).pack(pady=20)

        self.btn_load = ctk.CTkButton(self.sidebar, text="Charger l'OCR", command=self.load_model, fg_color="#E59F2F")
        self.btn_load.pack(pady=10, padx=20, fill="x")

        self.btn_run = ctk.CTkButton(self.sidebar, text="LANCER L'ANALYSE", command=self.start_benchmark, state="disabled", fg_color="#2CC985")
        self.btn_run.pack(pady=10, padx=20, fill="x")

        # Section Stats
        ctk.CTkLabel(self.sidebar, text="--- STATISTIQUES ---", font=("Roboto", 16)).pack(pady=(30, 10))
        
        self.lbl_total = ctk.CTkLabel(self.sidebar, text="Total: 0", font=("Roboto", 14))
        self.lbl_total.pack(pady=2)
        
        self.lbl_success = ctk.CTkLabel(self.sidebar, text="Succès: 0", text_color="#2CC985", font=("Roboto", 14, "bold"))
        self.lbl_success.pack(pady=2)
        
        self.lbl_fail = ctk.CTkLabel(self.sidebar, text="Échecs: 0", text_color="#FF4444", font=("Roboto", 14, "bold"))
        self.lbl_fail.pack(pady=2)

        self.lbl_accuracy = ctk.CTkLabel(self.sidebar, text="Précision: 0.0%", font=("Roboto", 24, "bold"))
        self.lbl_accuracy.pack(pady=20)

        self.progress_bar = ctk.CTkProgressBar(self.sidebar)
        self.progress_bar.pack(pady=10, padx=20, fill="x")
        self.progress_bar.set(0)

        # 2. ZONE CENTRALE (Tableau des résultats)
        self.result_frame = ctk.CTkScrollableFrame(self, label_text="Résultats en temps réel")
        self.result_frame.grid(row=0, column=1, rowspan=2, padx=20, pady=20, sticky="nsew")
        
        # En-têtes du tableau
        headers = ["Fichier", "Attendu (Nom)", "Détecté (Algo)", "Confiance", "Résultat"]
        for i, h in enumerate(headers):
            lbl = ctk.CTkLabel(self.result_frame, text=h, font=("Roboto", 14, "bold"))
            lbl.grid(row=0, column=i, padx=10, pady=5, sticky="w")
            
        self.current_row = 1

    def load_model(self):
        """Charge EasyOCR dans un thread pour ne pas figer l'interface"""
        self.btn_load.configure(text="Chargement en cours...", state="disabled")
        
        def _load():
            print("Chargement EasyOCR...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR chargé !")
            # Mise à jour UI depuis le thread principal
            self.after(0, lambda: self.btn_load.configure(text="Modèle Chargé", fg_color="green"))
            self.after(0, lambda: self.btn_run.configure(state="normal"))

        threading.Thread(target=_load, daemon=True).start()

    def parse_expected_layout(self, filename):
        """
        Extrait le layout attendu du nom de fichier.
        Convention: FORMAT-OS-LAYOUT-X.png (ex: ISO-WIN-AZERTY-1.png)
        """
        try:
            name_no_ext = os.path.splitext(filename)[0] # Enlève .png
            parts = name_no_ext.split('-')
            
            # On cherche AZERTY, QWERTY ou QWERTZ dans les parties du nom
            for part in parts:
                p = part.upper()
                if p in ["AZERTY", "QWERTY", "QWERTZ"]:
                    return p
            return "INCONNU"
        except Exception:
            return "ERREUR"

    def start_benchmark(self):
        if self.is_running: return
        self.is_running = True
        self.btn_run.configure(state="disabled", text="Analyse en cours...")
        
        # Reset stats
        self.total_processed = 0
        self.success_count = 0
        self.fail_count = 0
        self.current_row = 1
        
        # Nettoyage tableau (sauf headers)
        for widget in self.result_frame.winfo_children():
            info = widget.grid_info()
            if int(info['row']) > 0:
                widget.destroy()

        # Liste des fichiers
        if not os.path.exists(self.folder_path):
            print("Dossier introuvable")
            self.is_running = False
            return

        files = [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_files = len(files)

        # Lancement du worker thread
        threading.Thread(target=self.process_files, args=(files, total_files), daemon=True).start()

    def process_files(self, files, total_files):
        for i, filename in enumerate(files):
            file_path = os.path.join(self.folder_path, filename)
            
            # 1. Détermination de la vérité terrain (Ground Truth)
            expected = self.parse_expected_layout(filename)
            
            # 2. Analyse Algorithmique
            img = cv2.imread(file_path)
            detected_layout = "Erreur"
            confidence = 0
            
            if img is not None:
                # Pipeline complet
                processed = get_processed_images(img)
                chars = run_ocr_pipeline(self.ocr_reader, processed)
                
                # Check si assez de lettres
                if len(chars) >= 4:
                    rows = cluster_rows(chars)
                    if rows:
                        detected_layout, confidence, _ = score_layout(rows)
                    else:
                        detected_layout = "Echec Cluster"
                else:
                    detected_layout = "Pas assez de lettres"
            
            # 3. Comparaison
            is_success = (expected == detected_layout)
            
            # 4. Mise à jour des stats et de l'UI
            self.after(0, lambda f=filename, e=expected, d=detected_layout, c=confidence, s=is_success, idx=i, tot=total_files: 
                       self.update_ui_row(f, e, d, c, s, idx, tot))
            
            # Petit délai pour laisser l'UI respirer
            # time.sleep(0.1) 

        self.after(0, lambda: self.btn_run.configure(state="normal", text="RELANCER L'ANALYSE"))
        self.is_running = False

    def update_ui_row(self, filename, expected, detected, confidence, is_success, index, total):
        row = self.current_row
        
        # Update stats
        self.total_processed += 1
        if is_success: self.success_count += 1
        else: self.fail_count += 1
        
        accuracy = (self.success_count / self.total_processed) * 100
        
        # Update Labels stats
        self.lbl_total.configure(text=f"Total: {self.total_processed}/{total}")
        self.lbl_success.configure(text=f"Succès: {self.success_count}")
        self.lbl_fail.configure(text=f"Échecs: {self.fail_count}")
        self.lbl_accuracy.configure(text=f"Précision: {accuracy:.1f}%")
        self.progress_bar.set((index + 1) / total)

        # Couleurs
        color_res = "#2CC985" if is_success else "#FF4444"
        icon = "✅" if is_success else "❌"
        
        # Affichage Ligne Tableau
        ctk.CTkLabel(self.result_frame, text=filename, width=200, anchor="w").grid(row=row, column=0, padx=10, pady=2)
        ctk.CTkLabel(self.result_frame, text=expected, width=100, text_color="#AAA").grid(row=row, column=1, padx=10, pady=2)
        ctk.CTkLabel(self.result_frame, text=detected, width=100, text_color=color_res, font=("Roboto", 12, "bold")).grid(row=row, column=2, padx=10, pady=2)
        ctk.CTkLabel(self.result_frame, text=f"{confidence:.0f}%", width=80).grid(row=row, column=3, padx=10, pady=2)
        ctk.CTkLabel(self.result_frame, text=icon, width=50).grid(row=row, column=4, padx=10, pady=2)

        self.current_row += 1

if __name__ == "__main__":
    app = BenchmarkApp()
    app.mainloop()