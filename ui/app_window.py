import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import time
import os
from collections import deque

from core.vision_engine import VisionEngine
from core.clinical_logic import ClinicalLogic
from core.speech_engine import SpeechEngine
from data.database import DatabaseManager


BG_COLOR = "#0f172a"
PANEL_COLOR = "#1e293b"
TEXT_COLOR = "#f8fafc"
ACCENT_BLUE = "#3b82f6"
ALERT_RED = "#ef4444"
SUCCESS_GREEN = "#22c55e"
WARNING_YELLOW = "#eab308"
MUTED_COLOR = "#94a3b8"


class NeuroScanApp:
    """
    NeuroScan V4.

    La interfaz distingue:
    - Confiabilidad técnica: calidad de captura/tracking.
    - Margen dentro de parámetros: distancia heurística al threshold.

    Ninguno de los dos valores equivale a sensibilidad, especificidad,
    precisión diagnóstica ni probabilidad de ausencia de ACV.
    """

    MIN_TECH_QUALITY = 0.65
    GOOD_TECH_QUALITY = 0.80
    HIGH_TECH_QUALITY = 0.90
    QUALITY_WINDOW = 30
    PROCESS_WIDTH = 800
    PROCESS_HEIGHT = 600
    VIDEO_W = 960
    VIDEO_H = 540

    # Actualización visual aproximada: ~30 FPS
    TARGET_MS = 33

    # MediaPipe se ejecuta cada 2 frames.
    # El vídeo sigue actualizándose en todos los frames.
    INFERENCE_EVERY_N_FRAMES = 2

    MIN_Q = 0.65
    GOOD_Q = 0.80
    HIGH_Q = 0.90

    def __init__(self, root):
        self.root = root
        self.root.title("NeuroScan AI - Suite de Triaje Avanzado")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        window_w = min(1500, max(1150, int(screen_w * 0.94)))
        window_h = min(900, max(700, int(screen_h * 0.90)))

        self.root.geometry(f"{window_w}x{window_h}")
        self.root.minsize(1150, 700)
        self.root.configure(bg=BG_COLOR)

        self.vision = VisionEngine()
        self.logic = ClinicalLogic()
        self.speech = SpeechEngine()
        self.db = DatabaseManager()

        self.photo_dir = "captured_patients"
        os.makedirs(self.photo_dir, exist_ok=True)

        # Datos
        self.patient_id = tk.StringVar()
        self.patient_name = tk.StringVar()
        self.patient_age = tk.StringVar()
        self.has_hta = tk.BooleanVar()
        self.has_dm2 = tk.BooleanVar()
        self.has_anticoag = tk.BooleanVar()
        self.otros_comorb = tk.StringVar()
        self.nihss_score = tk.StringVar(value="0")

        # UI
        self.status_text = tk.StringVar(
            value="SISTEMA LISTO. Ingrese los datos del paciente."
        )
        self.res_face_text = tk.StringVar(value="Cara: Pendiente")
        self.res_eyes_text = tk.StringVar(value="Ojos: Pendiente")
        self.res_arms_text = tk.StringVar(value="Brazos: Pendiente")
        self.res_speech_text = tk.StringVar(value="Habla: Pendiente")
        self.res_quality_text = tk.StringVar(value="Calidad: --")

        # Indicadores individuales de las tarjetas de resultados
        self.card_vars = {
            "FACE": tk.StringVar(value="Pendiente"),
            "EYES": tk.StringVar(value="Pendiente"),
            "ARMS": tk.StringVar(value="Pendiente"),
            "SPEECH": tk.StringVar(value="Pendiente"),
        }
        self.card_detail_vars = {
            "FACE": tk.StringVar(value=""),
            "EYES": tk.StringVar(value=""),
            "ARMS": tk.StringVar(value=""),
            "SPEECH": tk.StringVar(value=""),
        }
        self.card_widgets = {}

        # Flujo
        self.state = "IDLE"
        self.flow_start_time = 0.0
        # Estado del procesamiento de vídeo.
        # Debe existir ANTES de que update_video() sea llamado.
        self.frame_counter = 0
        self.last_landmarks = {
            "face": None,
            "pose": None,
            "framing": {},
            "quality": {}
        }
        self.setup_retry_count = 0

        self.temp_scores = []
        self.alert_flags = []

        self.final_face = 0.0
        self.final_eyes = 0.0
        self.final_arm = 0.0
        self.final_speech = 0.0

        self.photo_saved_path = ""

        # Calidad técnica por modalidad
        self.quality_history = deque(maxlen=self.QUALITY_WINDOW)
        self.phase_quality_samples = []
        self.phase_quality = {}
        self.final_quality = 0.0

        # Margen heurístico dentro de parámetros
        self.phase_normality = {
            "FACE": None,
            "EYES": None,
            "ARMS": None,
            "SPEECH": None
        }

        self.img_smile_tk = self._load_ui_image("smile.png")
        self.img_arms_tk = self._load_ui_image("arms.png")
        self.img_blank_tk = ImageTk.PhotoImage(
            Image.new("RGB", (200, 200), color=BG_COLOR)
        )

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Cámara",
                "No se pudo abrir la cámara web."
            )

        # Pedimos al driver una resolución razonable si la soporta.
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception:
            pass

        self.setup_ui()
        self.update_video()

    # ============================================================
    # UI
    # ============================================================

    def _load_ui_image(self, filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)

        paths = [
            os.path.join(base_dir, "datasets", filename),
            os.path.join(base_dir, filename),
            filename,
        ]

        for p in paths:
            if os.path.exists(p):
                try:
                    img = Image.open(p).resize(
                        (180, 180),
                        Image.Resampling.LANCZOS
                    )
                    return ImageTk.PhotoImage(img)
                except Exception:
                    pass

        return None

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Panel.TFrame", background=PANEL_COLOR)
        style.configure("TLabel", background=PANEL_COLOR, foreground=TEXT_COLOR)
        style.configure("TCheckbutton", background=PANEL_COLOR, foreground=TEXT_COLOR)
        style.configure(
            "Clinical.TLabelframe",
            background=PANEL_COLOR,
            foreground=TEXT_COLOR
        )
        style.configure(
            "Clinical.TLabelframe.Label",
            background=PANEL_COLOR,
            foreground=ACCENT_BLUE,
            font=("Segoe UI", 10, "bold")
        )

        # ========================================================
        # PANEL IZQUIERDO
        # ========================================================
        left = ttk.Frame(self.root, style="Panel.TFrame", width=290)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        left.pack_propagate(False)

        tk.Label(
            left,
            text="Registro del paciente",
            font=("Segoe UI", 14, "bold"),
            bg=PANEL_COLOR,
            fg=ACCENT_BLUE
        ).pack(pady=(12, 10))

        fields = [
            ("ID / DNI", self.patient_id),
            ("Nombre completo", self.patient_name),
            ("Edad", self.patient_age),
        ]

        for label, var in fields:
            tk.Label(
                left, text=label, bg=PANEL_COLOR,
                fg=TEXT_COLOR, font=("Segoe UI", 9)
            ).pack(anchor=tk.W, padx=12)
            ttk.Entry(left, textvariable=var).pack(
                fill=tk.X, padx=12, pady=(2, 7)
            )

        # Menú desplegable de datos clínicos
        self.clinical_expanded = tk.BooleanVar(value=False)
        self.clinical_toggle = tk.Button(
            left,
            text="▶  Datos clínicos",
            command=self.toggle_clinical_panel,
            bg="#24324a",
            fg=TEXT_COLOR,
            activebackground="#334155",
            activeforeground=TEXT_COLOR,
            relief=tk.FLAT,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            padx=10
        )
        self.clinical_toggle.pack(fill=tk.X, padx=10, pady=(10, 4))

        self.clinical_panel = tk.Frame(
            left, bg=PANEL_COLOR
        )

        tk.Label(
            self.clinical_panel,
            text="Antecedentes / comorbilidades",
            bg=PANEL_COLOR,
            fg=MUTED_COLOR,
            font=("Segoe UI", 9, "bold")
        ).pack(anchor=tk.W, padx=4, pady=(4, 3))

        for text, variable in (
            ("Hipertensión arterial (HTA)", self.has_hta),
            ("Diabetes mellitus (DM2)", self.has_dm2),
            ("Toma anticoagulantes", self.has_anticoag),
        ):
            ttk.Checkbutton(
                self.clinical_panel,
                text=text,
                variable=variable
            ).pack(anchor=tk.W, pady=2)

        tk.Label(
            self.clinical_panel,
            text="Otros antecedentes",
            bg=PANEL_COLOR,
            fg=MUTED_COLOR,
            font=("Segoe UI", 9)
        ).pack(anchor=tk.W, pady=(6, 2))

        ttk.Entry(
            self.clinical_panel,
            textvariable=self.otros_comorb
        ).pack(fill=tk.X, pady=(0, 7))

        tk.Label(
            self.clinical_panel,
            text="NIHSS adicional (0–42)",
            bg=PANEL_COLOR,
            fg=MUTED_COLOR,
            font=("Segoe UI", 9)
        ).pack(anchor=tk.W)

        ttk.Entry(
            self.clinical_panel,
            textvariable=self.nihss_score,
            width=10
        ).pack(anchor=tk.W, pady=(2, 5))

        # Botón de inicio
        self.start_button = tk.Button(
            left,
            text="▶  INICIAR EXAMEN",
            bg=ACCENT_BLUE,
            fg=TEXT_COLOR,
            activebackground="#2563eb",
            activeforeground=TEXT_COLOR,
            font=("Segoe UI", 11, "bold"),
            command=self.start_unified_flow,
            relief=tk.FLAT,
            padx=8,
            pady=9
        )
        self.start_button.pack(
            fill=tk.X, padx=10, pady=14
        )

        tk.Button(
            left,
            text="💾  Guardar en base de datos",
            bg=SUCCESS_GREEN,
            fg=BG_COLOR,
            activebackground="#16a34a",
            font=("Segoe UI", 10, "bold"),
            command=self.save_to_database,
            relief=tk.FLAT,
            pady=7
        ).pack(
            side=tk.BOTTOM,
            fill=tk.X, padx=10, pady=12
        )

        # ========================================================
        # CENTRO: CÁMARA + GUÍAS ABAJO
        # ========================================================
        center = tk.Frame(self.root, bg=BG_COLOR)
        center.pack(
            side=tk.LEFT, fill=tk.BOTH,
            expand=True, padx=4, pady=8
        )

        camera_panel = tk.Frame(
            center, bg="#050b16",
            highlightbackground="#334155",
            highlightthickness=1
        )
        camera_panel.pack(
            fill=tk.BOTH, expand=True,
            padx=4, pady=(0, 5)
        )

        self.video_label = tk.Label(
            camera_panel, bg="black"
        )
        self.video_label.pack(
            fill=tk.BOTH, expand=True
        )

        # Las imágenes ya NO comparten espacio horizontal con la cámara.
        # Se mantienen siempre abajo a la izquierda.
        self.guide_panel = tk.Frame(
            center, bg=PANEL_COLOR,
            height=155
        )
        self.guide_panel.pack(
            fill=tk.X, side=tk.BOTTOM,
            padx=4, pady=(0, 5)
        )
        self.guide_panel.pack_propagate(False)

        tk.Label(
            self.guide_panel,
            text="GUÍA DEL EXAMEN",
            bg=PANEL_COLOR,
            fg=MUTED_COLOR,
            font=("Segoe UI", 8, "bold")
        ).pack(side=tk.LEFT, padx=(10, 8))

        self.guide_img_label = tk.Label(
            self.guide_panel,
            image=self.img_blank_tk,
            bg=PANEL_COLOR
        )
        self.guide_img_label.pack(
            side=tk.LEFT, padx=5, pady=5
        )

        self.guide_text_var = tk.StringVar(
            value="La imagen aparecerá durante la prueba correspondiente."
        )
        tk.Label(
            self.guide_panel,
            textvariable=self.guide_text_var,
            bg=PANEL_COLOR,
            fg=TEXT_COLOR,
            font=("Segoe UI", 9),
            justify=tk.LEFT
        ).pack(
            side=tk.LEFT, padx=12
        )

        self.status_bar = tk.Label(
            center,
            textvariable=self.status_text,
            font=("Segoe UI", 13, "bold"),
            bg=PANEL_COLOR,
            fg=ACCENT_BLUE,
            pady=9
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # ========================================================
        # PANEL DERECHO: TARJETAS DE RESULTADOS
        # ========================================================
        right = ttk.Frame(
            self.root, style="Panel.TFrame",
            width=345
        )
        right.pack(
            side=tk.RIGHT, fill=tk.Y,
            padx=8, pady=8
        )
        right.pack_propagate(False)

        tk.Label(
            right,
            text="Resultados del triaje",
            font=("Segoe UI", 15, "bold"),
            bg=PANEL_COLOR,
            fg=ACCENT_BLUE
        ).pack(pady=(12, 4))

        tk.Label(
            right,
            text="Indicadores técnicos y hallazgos",
            font=("Segoe UI", 8),
            bg=PANEL_COLOR,
            fg=MUTED_COLOR
        ).pack(pady=(0, 10))

        self._create_result_card(
            right, "FACE", "●  ROSTRO", ACCENT_BLUE
        )
        self._create_result_card(
            right, "EYES", "◉  OJOS", "#8b5cf6"
        )
        self._create_result_card(
            right, "ARMS", "↕  BRAZOS", "#06b6d4"
        )
        self._create_result_card(
            right, "SPEECH", "◌  HABLA", WARNING_YELLOW
        )

        self.arm_live_label = tk.Label(
            right,
            text="",
            bg=PANEL_COLOR,
            fg=MUTED_COLOR,
            font=("Consolas", 8),
            justify=tk.LEFT,
            anchor="w"
        )
        self.arm_live_label.pack(fill=tk.X, padx=12, pady=(2, 6))

        self.global_quality_frame = tk.Frame(
            right, bg="#162033",
            highlightbackground="#334155",
            highlightthickness=1
        )
        self.global_quality_frame.pack(
            fill=tk.X, padx=10, pady=(8, 5)
        )

        tk.Label(
            self.global_quality_frame,
            text="CALIDAD TÉCNICA GLOBAL",
            bg="#162033",
            fg=MUTED_COLOR,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=12, pady=(8, 1))

        self.global_quality_label = tk.Label(
            self.global_quality_frame,
            textvariable=self.res_quality_text,
            bg="#162033",
            fg=TEXT_COLOR,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            justify=tk.LEFT
        )
        self.global_quality_label.pack(
            fill=tk.X, padx=12, pady=(0, 8)
        )

    def _create_result_card(self, parent, key, title, accent):
        card = tk.Frame(
            parent,
            bg="#162033",
            highlightbackground="#334155",
            highlightthickness=1
        )
        card.pack(
            fill=tk.X, padx=10, pady=4
        )

        header = tk.Frame(card, bg="#162033")
        header.pack(fill=tk.X, padx=10, pady=(7, 1))

        tk.Label(
            header,
            text=title,
            bg="#162033",
            fg=accent,
            font=("Segoe UI", 9, "bold")
        ).pack(side=tk.LEFT)

        status_label = tk.Label(
            card,
            textvariable=self.card_vars[key],
            bg="#162033",
            fg=TEXT_COLOR,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
            justify=tk.LEFT,
            wraplength=300
        )
        status_label.pack(
            fill=tk.X, padx=10, pady=(2, 1)
        )

        detail_label = tk.Label(
            card,
            textvariable=self.card_detail_vars[key],
            bg="#162033",
            fg=MUTED_COLOR,
            font=("Segoe UI", 8),
            anchor="w",
            justify=tk.LEFT,
            wraplength=300
        )
        detail_label.pack(
            fill=tk.X, padx=10, pady=(0, 8)
        )

        self.card_widgets[key] = {
            "frame": card,
            "status": status_label,
            "detail": detail_label,
            "accent": accent
        }

    def toggle_clinical_panel(self):
        if self.clinical_expanded.get():
            self.clinical_panel.pack_forget()
            self.clinical_expanded.set(False)
            self.clinical_toggle.configure(text="▶  Datos clínicos")
        else:
            self.clinical_panel.pack(
                fill=tk.X, padx=10, pady=(0, 5)
            )
            self.clinical_expanded.set(True)
            self.clinical_toggle.configure(text="▼  Datos clínicos")

    def _set_guide(self, image, text):
        if image:
            self.guide_img_label.configure(image=image)
            self.guide_img_label.image = image
        else:
            self.guide_img_label.configure(
                image=self.img_blank_tk
            )
            self.guide_img_label.image = self.img_blank_tk
        self.guide_text_var.set(text)

    # ============================================================
    # UTILIDADES DE CALIDAD
    # ============================================================

    @staticmethod
    def _pct(score):
        if score is None:
            return "--"
        return str(
            int(round(
                max(0.0, min(1.0, float(score))) * 100
            ))
        )

    def _tech_label(self, score):
        if score is None:
            return "Confiabilidad técnica: no evaluable"

        pct = self._pct(score)

        if score >= self.HIGH_TECH_QUALITY:
            level = "alta"
        elif score >= self.GOOD_TECH_QUALITY:
            level = "buena"
        elif score >= self.MIN_TECH_QUALITY:
            level = "aceptable"
        else:
            level = "baja"

        return f"Confiabilidad técnica: {pct}% — {level}"

    @staticmethod
    def _normality_margin(score, threshold):
        """
        Índice de margen dentro del rango definido por el sistema.
        NO es probabilidad clínica.

        100% => score aproximadamente 0
        50%  => score aproximadamente 50% del threshold
        0%   => score >= threshold

        Es deliberadamente simple y transparente.
        """
        if score is None or threshold <= 0:
            return None

        margin = 1.0 - (
            float(score) /
            float(threshold)
        )

        return max(0.0, min(1.0, margin))

    def _register_quality(self, quality_data, modality):
        if not quality_data:
            return

        if modality == "FACE":
            score = quality_data.get("face", 0.0)
        elif modality == "ARMS":
            score = quality_data.get("arms", 0.0)
        else:
            score = quality_data.get("overall", 0.0)

        score = float(max(0.0, min(1.0, score)))

        self.quality_history.append(score)
        self.phase_quality_samples.append(score)

    def _finish_phase_quality(self, phase):
        if self.phase_quality_samples:
            score = float(
                np_mean(self.phase_quality_samples)
            )
        else:
            score = None

        self.phase_quality[phase] = score
        self.phase_quality_samples = []

        return score

    def _overall_quality(self):
        values = [
            v for v in self.phase_quality.values()
            if v is not None
        ]

        if not values:
            return None

        return float(np_mean(values))

    # ============================================================
    # FLOW
    # ============================================================

    def start_unified_flow(self):
        if not self.patient_id.get().strip():
            messagebox.showwarning(
                "DNI Requerido",
                "Ingrese el ID del paciente antes de empezar."
            )
            return

        if self.state not in ("IDLE", "RESULTS"):
            return

        self.state = "SETUP"
        self.flow_start_time = time.time()
        self.frame_counter = 0
        self.last_landmarks = {
            "face": None,
            "pose": None,
            "framing": {},
            "quality": {}
        }
        self.setup_retry_count = 0

        self.temp_scores = []
        self.alert_flags = []

        self.final_face = 0.0
        self.final_eyes = 0.0
        self.final_arm = 0.0
        self.final_speech = 0.0
        self.photo_saved_path = ""

        self.quality_history.clear()
        self.phase_quality_samples = []
        self.phase_quality = {}

        self.phase_normality = {
            "FACE": None,
            "EYES": None,
            "ARMS": None,
            "SPEECH": None
        }

        self.logic.baseline_face_asym = 0.0
        self.logic.is_calibrated = False
        self.logic.reset_face_windows()

        self.res_face_text.set("Cara: Pendiente")
        self.res_eyes_text.set("Ojos: Pendiente")
        self.res_arms_text.set("Brazos: Pendiente")
        self.res_speech_text.set("Habla: Pendiente")
        self.res_quality_text.set(
            "Confiabilidad técnica: preparando cámara..."
        )

        self.status_bar.configure(
            fg=WARNING_YELLOW
        )
        self.start_button.configure(
            state=tk.DISABLED
        )

    # ============================================================
    # DB
    # ============================================================

    def save_to_database(self):
        pid = self.patient_id.get().strip()

        if not pid:
            messagebox.showerror(
                "Error",
                "Falta DNI del paciente."
            )
            return

        if self.state != "RESULTS":
            messagebox.showwarning(
                "Examen incompleto",
                "Complete el examen antes de guardar."
            )
            return

        age = (
            int(self.patient_age.get())
            if self.patient_age.get().isdigit()
            else 0
        )

        nihss = (
            int(self.nihss_score.get())
            if self.nihss_score.get().isdigit()
            else 0
        )

        try:
            self.db.save_patient(
                pid,
                self.patient_name.get(),
                age,
                self.has_hta.get(),
                self.has_dm2.get(),
                self.has_anticoag.get(),
                self.otros_comorb.get(),
                nihss,
                self.photo_saved_path
            )

            is_alert = len(self.alert_flags) > 0

            self.db.save_evaluation(
                pid,
                self.final_face,
                self.final_arm,
                self.final_speech,
                is_alert
            )

            messagebox.showinfo(
                "Guardado",
                "Registro clínico e imágenes consolidadas con éxito."
            )

        except Exception as exc:
            messagebox.showerror(
                "Error de base de datos",
                f"No se pudo guardar el examen:\n{exc}"
            )

    # ============================================================
    # MENSAJES DE ENCUADRE
    # ============================================================

    def _setup_message(self, framing):
        if not framing.get("ready_for_face", False):
            return (
                "AJUSTE EL ENCUADRE: rostro centrado, "
                "mirando a la cámara y a una distancia aproximada "
                "de 50–80 cm."
            )

        return (
            "ENCUADRE CORRECTO ✓ — Mantenga la posición."
        )

    def _arms_setup_message(self, framing):
        visible = framing.get("arms_visible_count", 0)

        if visible < 6:
            return (
                f"AJUSTE LA POSICIÓN: se ven {visible}/6 "
                "puntos necesarios. Aléjese ligeramente y "
                "mantenga ambos hombros, codos y muñecas dentro "
                "del cuadro."
            )

        return (
            "POSICIÓN CORRECTA ✓ — Levante ambos brazos "
            "cuando aparezca la indicación."
        )

    # ============================================================
    # PANEL
    # ============================================================

    def _update_result_panel(self):
        face_q = self.phase_quality.get("FACE")
        eyes_q = self.phase_quality.get("FACE")
        arms_q = self.phase_quality.get("ARMS")
        speech_q = self.phase_quality.get("SPEECH")

        face_margin = self.phase_normality.get("FACE")
        eye_margin = self.phase_normality.get("EYES")
        arm_margin = self.phase_normality.get("ARMS")

        face_alert = "CARA" in self.alert_flags
        eyes_alert = "OJOS" in self.alert_flags
        arms_alert = "MOTOR" in self.alert_flags
        speech_alert = "HABLA" in self.alert_flags

        face_status = (
            "Asimetría detectada" if face_alert
            else "Sin asimetría significativa"
        )
        eyes_status = (
            "Alteración detectada" if eyes_alert
            else "Sin alteraciones detectadas"
        )

        if arms_q is None:
            arms_status = "Prueba no interpretable"
        elif arms_alert:
            arms_status = "Descenso de un brazo"
        else:
            arms_status = "Sin descenso significativo"

        speech_status = (
            "Respuesta no coincidente" if speech_alert
            else "Frase reconocida"
        )

        def margin_text(m):
            if m is None:
                return "Margen dentro del umbral: no interpretable"
            return f"Margen dentro del umbral: {self._pct(m)}%"

        self.card_vars["FACE"].set(face_status)
        self.card_detail_vars["FACE"].set(
            f"{self._tech_label(face_q)}\n{margin_text(face_margin)}"
        )

        self.card_vars["EYES"].set(eyes_status)
        self.card_detail_vars["EYES"].set(
            f"{self._tech_label(eyes_q)}\n{margin_text(eye_margin)}"
        )

        self.card_vars["ARMS"].set(arms_status)
        self.card_detail_vars["ARMS"].set(
            f"{self._tech_label(arms_q)}\n{margin_text(arm_margin)}"
        )

        self.card_vars["SPEECH"].set(speech_status)
        self.card_detail_vars["SPEECH"].set(
            f"Calidad de captura/reconocimiento: {self._pct(speech_q)}%\n"
            f"Coincidencia de frase: {self._pct(self.final_speech)}%"
        )

        # Colores de estado: verde = sin hallazgo, rojo = alerta,
        # amarillo = no interpretable.
        states = {
            "FACE": ("alert" if face_alert else "ok"),
            "EYES": ("alert" if eyes_alert else "ok"),
            "ARMS": (
                "alert" if arms_alert
                else "warning" if arms_q is None
                else "ok"
            ),
            "SPEECH": (
                "alert" if speech_alert else "ok"
            )
        }

        for key, state in states.items():
            card = self.card_widgets[key]
            if state == "alert":
                bg, fg = "#3a1720", ALERT_RED
            elif state == "warning":
                bg, fg = "#3a2f12", WARNING_YELLOW
            else:
                bg, fg = "#122b20", SUCCESS_GREEN

            card["frame"].configure(
                bg=bg, highlightbackground=fg
            )
            card["status"].configure(
                bg=bg, fg=TEXT_COLOR
            )
            card["detail"].configure(
                bg=bg, fg=MUTED_COLOR
            )
            # También cambia el fondo del header implícito:
            for child in card["frame"].winfo_children():
                try:
                    child.configure(bg=bg)
                except tk.TclError:
                    pass

        global_q = self._overall_quality()

        if global_q is None:
            self.res_quality_text.set(
                "Pendiente de completar"
            )
            self.global_quality_label.configure(
                fg=MUTED_COLOR
            )
        else:
            self.res_quality_text.set(
                f"{self._pct(global_q)}% — "
                f"{self._tech_label(global_q).split('—')[-1].strip()}\n"
                "No equivale a precisión diagnóstica."
            )
            self.global_quality_label.configure(
                fg=(
                    SUCCESS_GREEN
                    if global_q >= self.GOOD_TECH_QUALITY
                    else WARNING_YELLOW
                )
            )

    # ============================================================
    # LOOP
    # ============================================================

    def update_video(self):
        # Protección contra callbacks tempranos de Tkinter.
        if not hasattr(self, "frame_counter"):
            self.frame_counter = 0
        if not hasattr(self, "last_landmarks"):
            self.last_landmarks = {
                "face": None,
                "pose": None,
                "framing": {},
                "quality": {}
            }

        try:
            ret, frame = self.cap.read()

            if ret:
                frame = cv2.resize(
                    frame,
                    (self.PROCESS_WIDTH, self.PROCESS_HEIGHT)
                )
                self.frame_counter += 1

                # Overlay clínico dinámico: durante la cara mostramos
                # landmarks faciales clave; durante brazos, pose corporal.
                if self.state in ("SETUP", "CALIB", "FACE"):
                    self.vision.set_overlay_mode("FACE")
                elif self.state in ("ARMS_SETUP", "ARMS"):
                    self.vision.set_overlay_mode("ARMS")
                else:
                    self.vision.set_overlay_mode("MINIMAL")

                if (self.frame_counter % self.INFERENCE_EVERY_N_FRAMES == 0
                        or self.last_landmarks.get("face") is None and self.last_landmarks.get("pose") is None):
                    annotated, landmarks = self.vision.process_frame(frame)
                    self.last_landmarks = landmarks
                else:
                    # Muestra vídeo en cada frame, pero evita ejecutar FaceMesh + Pose en todos.
                    annotated = frame
                    landmarks = self.last_landmarks

                framing = landmarks.get("framing", {})
                quality = landmarks.get("quality", {})

                # ---------------- SETUP ----------------
                if self.state == "SETUP":

                    self.vision.draw_setup_overlay(
                        annotated,
                        framing,
                        arms=False
                    )

                    self.status_text.set(
                        self._setup_message(framing)
                    )

                    if framing.get(
                        "ready_for_face", False
                    ):
                        self.status_bar.configure(
                            fg=SUCCESS_GREEN
                        )
                        # Mantener una pequeña estabilidad antes de empezar.
                        if (
                            time.time() -
                            self.flow_start_time
                        ) > 1.0:

                            self.state = "CALIB"
                            self.flow_start_time = time.time()
                            self.temp_scores = []
                            self.phase_quality_samples = []

                    else:
                        self.status_bar.configure(
                            fg=WARNING_YELLOW
                        )

                # ---------------- CALIB ----------------
                elif self.state == "CALIB":

                    elapsed = time.time() - self.flow_start_time

                    self._register_quality(
                        quality,
                        "FACE"
                    )

                    remaining = max(
                        0,
                        3 - int(elapsed)
                    )

                    self.status_text.set(
                        f"1/4 CALIBRANDO ROSTRO NEUTRO... "
                        f"({remaining}s)"
                    )

                    if landmarks.get("face"):

                        res = self.logic.evaluate_face(
                            landmarks["face"]
                        )

                        self.temp_scores.append(
                            res["score"]
                        )

                    if elapsed > 3:

                        self.logic.calibrate_baseline(
                            self.temp_scores
                        )

                        phase_q = self._finish_phase_quality(
                            "FACE"
                        )

                        pid = self.patient_id.get().strip()

                        self.photo_saved_path = os.path.join(
                            self.photo_dir,
                            f"{pid}_{int(time.time())}.jpg"
                        )

                        cv2.imwrite(
                            self.photo_saved_path,
                            frame
                        )

                        self.state = "FACE"
                        self.flow_start_time = time.time()
                        self.phase_quality_samples = []

                        self._set_guide(
                            self.img_smile_tk,
                            "Pida al paciente que sonría ampliamente.\n"
                            "Mantenga la cabeza mirando al frente."
                        )

                # ---------------- FACE ----------------
                elif self.state == "FACE":

                    elapsed = time.time() - self.flow_start_time

                    self._register_quality(
                        quality,
                        "FACE"
                    )

                    self.status_bar.configure(
                        fg=ACCENT_BLUE
                    )

                    self.status_text.set(
                        f"2/4 SONRÍA AMPLIAMENTE "
                        f"({max(0, 6-int(elapsed))}s)"
                    )

                    if landmarks.get("face"):

                        res = self.logic.update_face_temporal(
                            landmarks["face"]
                        )

                        self.final_face = res["face_score_temporal"]
                        self.final_eyes = res["eye_score_temporal"]

                        if res.get("face_alert_temporal") and "CARA" not in self.alert_flags:
                            self.alert_flags.append("CARA")

                        if res.get("eyes_alert_temporal") and "OJOS" not in self.alert_flags:
                            self.alert_flags.append("OJOS")

                    if elapsed > 6:

                        face_q = self._finish_phase_quality(
                            "FACE"
                        )

                        self.phase_normality["FACE"] = (
                            self._normality_margin(
                                self.final_face,
                                self.logic.GLOBAL_ASYMMETRY_THRESHOLD
                            )
                            if face_q is not None
                            and face_q >= self.MIN_TECH_QUALITY
                            else None
                        )

                        self.phase_normality["EYES"] = (
                            self._normality_margin(
                                self.final_eyes,
                                self.logic.PTOSIS_THRESHOLD
                            )
                            if face_q is not None
                            and face_q >= self.MIN_TECH_QUALITY
                            else None
                        )

                        self._set_guide(
                            self.img_arms_tk,
                            "Pida al paciente que levante ambos brazos\n"
                            "y los mantenga elevados."
                        )

                        self.state = "ARMS_SETUP"
                        self.flow_start_time = time.time()
                        self.phase_quality_samples = []

                # ---------------- ARMS SETUP ----------------
                elif self.state == "ARMS_SETUP":

                    self.vision.draw_setup_overlay(
                        annotated,
                        framing,
                        arms=True
                    )

                    self.status_text.set(
                        self._arms_setup_message(
                            framing
                        )
                    )

                    self.status_bar.configure(
                        fg=(
                            SUCCESS_GREEN
                            if framing.get("ready_for_arms", False)
                            else WARNING_YELLOW
                        )
                    )

                    # No empezamos a contar los 5 s hasta que las
                    # seis referencias corporales estén visibles.
                    if framing.get(
                        "ready_for_arms", False
                    ):

                        if (
                            time.time() -
                            self.flow_start_time
                        ) > 1.0:

                            started, _ = self.logic.start_arm_test(landmarks.get("pose"))
                            if started:
                                self.state = "ARMS"
                            self.flow_start_time = time.time()
                            self.phase_quality_samples = []

                # ---------------- ARMS ----------------
                elif self.state == "ARMS":

                    elapsed = time.time() - self.flow_start_time

                    self._register_quality(
                        quality,
                        "ARMS"
                    )

                    self.status_bar.configure(
                        fg=ACCENT_BLUE
                    )

                    self.status_text.set(
                        f"3/4 LEVANTE AMBOS BRAZOS Y "
                        f"MANTÉNGALOS ELEVADOS "
                        f"({max(0, 5-int(elapsed))}s)"
                    )

                    if landmarks.get("pose"):

                        res = self.logic.update_arm_test(
                            landmarks["pose"],
                            landmarks.get("face")
                        )

                        self.final_arm = res.get("drift_score", 0.0)

                        if hasattr(self, "arm_live_label"):
                            self.arm_live_label.configure(
                                text=(
                                    f"Descenso izq: {res.get('left_drop', 0):.3f}\n"
                                    f"Descenso der: {res.get('right_drop', 0):.3f}\n"
                                    f"Asimetría: {res.get('asymmetry', 0):.3f}\n"
                                    f"Persistencia: {res.get('persistence', 0)*100:.0f}%"
                                )
                            )

                        if (
                            res.get("is_alert", False)
                            and
                            res.get("is_valid", False)
                            and
                            "MOTOR" not in self.alert_flags
                        ):
                            self.alert_flags.append(
                                "MOTOR"
                            )

                    if elapsed > 5:

                        arm_result = self.logic.finish_arm_test()

                        arms_q = self._finish_phase_quality(
                            "ARMS"
                        )

                        if (
                            not arm_result.get(
                                "has_raised",
                                False
                            )
                            or
                            arms_q is None
                            or
                            arms_q < self.MIN_TECH_QUALITY
                        ):
                            if "MOTOR" in self.alert_flags:
                                self.alert_flags.remove(
                                    "MOTOR"
                                )

                            self.res_arms_text.set(
                                "Brazos: Prueba no interpretable"
                            )

                            self.phase_normality["ARMS"] = None

                        else:
                            self.phase_normality["ARMS"] = (
                                self._normality_margin(
                                    self.final_arm,
                                    self.logic.ARM_DRIFT_THRESHOLD
                                )
                            )

                            if "MOTOR" in self.alert_flags:
                                self.res_arms_text.set(
                                    "Brazos: Descenso de un brazo"
                                )
                            else:
                                self.res_arms_text.set(
                                    "Brazos: Sin descenso significativo"
                                )

                        self.state = "SPEECH"
                        self.flow_start_time = time.time()
                        self.phase_quality_samples = []

                        self._set_guide(
                            self.img_blank_tk,
                            "Repita la frase indicada claramente."
                        )

                        self.speech.start_test()

                # ---------------- SPEECH ----------------
                elif self.state == "SPEECH":

                    self.status_bar.configure(
                        fg=WARNING_YELLOW
                    )

                    self.status_text.set(
                        f"4/4 REPITA: "
                        f"'{self.speech.target_phrase.upper()}'"
                    )

                    if self.speech.has_finished:

                        result = self.speech.get_result()

                        self.final_speech = float(
                            result.get("score", 0.0)
                        )

                        # Para speech no simulamos una probabilidad de
                        # reconocimiento que el backend no proporciona.
                        speech_q = (
                            1.0
                            if result.get(
                                "status"
                            ) == "COMPLETED"
                            else 0.0
                        )

                        self.phase_quality["SPEECH"] = speech_q

                        if result.get(
                            "status"
                        ) == "INVALID":

                            self.res_speech_text.set(
                                "Habla: Prueba no interpretable"
                            )

                        elif result.get(
                            "alert",
                            False
                        ):

                            if "HABLA" not in self.alert_flags:
                                self.alert_flags.append(
                                    "HABLA"
                                )

                            self.res_speech_text.set(
                                "Habla: Respuesta no coincidente"
                            )

                        else:

                            self.res_speech_text.set(
                                "Habla: Frase reconocida"
                            )

                        self.state = "RESULTS"

                        self._update_result_panel()

                # ---------------- RESULTS ----------------
                elif self.state == "RESULTS":

                    global_q = self._overall_quality()
                    self.final_quality = (
                        global_q if global_q is not None else 0.0
                    )

                    self._update_result_panel()

                    if (
                        global_q is None
                        or
                        global_q < self.MIN_TECH_QUALITY
                    ):

                        self.status_bar.configure(
                            fg=WARNING_YELLOW
                        )

                        self.status_text.set(
                            "⚠ EXAMEN COMPLETADO — "
                            "ALGUNA PRUEBA NO TIENE CALIDAD "
                            "TÉCNICA SUFICIENTE"
                        )

                    elif self.alert_flags:

                        self.status_bar.configure(
                            fg=ALERT_RED
                        )

                        self.status_text.set(
                            "⚠ RESULTADO SOSPECHOSO — "
                            "REQUIERE VALORACIÓN CLÍNICA"
                        )

                    else:

                        self.status_bar.configure(
                            fg=SUCCESS_GREEN
                        )

                        self.status_text.set(
                            "✅ EXAMEN COMPLETADO — "
                            "SIN SIGNOS SOSPECHOSOS DETECTADOS"
                        )

                    self.start_button.configure(
                        state=tk.NORMAL
                    )

                # Render
                cv_img = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                pil_img = Image.fromarray(cv_img)

                # Aprovechamos el ancho real disponible, manteniendo
                # aspecto 4:3.
                label_w = max(
                    640,
                    self.video_label.winfo_width()
                )

                label_h = max(
                    480,
                    self.video_label.winfo_height()
                )

                scale = min(
                    label_w / pil_img.width,
                    label_h / pil_img.height
                )

                if scale > 0:
                    new_size = (
                        max(1, int(pil_img.width * scale)),
                        max(1, int(pil_img.height * scale))
                    )
                    if new_size != pil_img.size:
                        pil_img = pil_img.resize(
                            new_size,
                            Image.Resampling.LANCZOS
                        )

                imgtk = ImageTk.PhotoImage(
                    image=pil_img
                )

                self.video_label.imgtk = imgtk
                self.video_label.configure(
                    image=imgtk
                )

        except Exception as exc:
            print(
                f"[NeuroScan] Error en update_video: {exc}"
            )

        self.root.after(
            15,
            self.update_video
        )

    # ============================================================
    # CIERRE
    # ============================================================

    def on_closing(self):
        try:
            self.cap.release()
        except Exception:
            pass

        try:
            self.vision.close()
        except Exception:
            pass

        try:
            self.db.close()
        except Exception:
            pass

        self.root.destroy()


def np_mean(values):
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroScanApp(root)
    root.protocol(
        "WM_DELETE_WINDOW",
        app.on_closing
    )
    root.mainloop()