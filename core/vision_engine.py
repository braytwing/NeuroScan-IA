import cv2
import mediapipe as mp
import numpy as np


class VisionEngine:
    """
    Motor de visión de NeuroScan IA.

    La 'confiabilidad técnica' NO es precisión diagnóstica.
    Es una estimación de qué tan adecuadas son las condiciones
    de captura y tracking para interpretar una prueba concreta.

    Scores:
        0.0 = datos no interpretables
        1.0 = condiciones técnicas excelentes

    La función clínica de los thresholds debe validarse posteriormente.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.C_BLU = (255, 100, 0)
        self.C_WHT = (255, 255, 255)
        self.C_CYA = (255, 200, 0)
        self.C_GRY = (100, 100, 100)
        self.C_GREEN = (50, 220, 50)
        self.C_RED = (50, 50, 220)
        self.C_YELLOW = (0, 220, 255)

        # Modo de overlay para no saturar la imagen.
        self.overlay_mode = "MINIMAL"

    def set_overlay_mode(self, mode):
        if mode not in ("MINIMAL", "FACE", "ARMS"):
            mode = "MINIMAL"
        self.overlay_mode = mode

    def process_frame(self, frame):
        if frame is None:
            return frame, {
                "face": None,
                "pose": None,
                "quality": self._empty_quality(),
                "framing": self._empty_framing()
            }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        res_face = self.face_mesh.process(rgb)
        res_pose = self.pose.process(rgb)

        face_lm = None
        pose_lm = None

        if res_face.multi_face_landmarks:
            face_lm = res_face.multi_face_landmarks[0].landmark
            if self.overlay_mode in ("MINIMAL", "FACE"):
                self._draw_face_overlay(frame, face_lm, w, h)

        if res_pose.pose_landmarks:
            pose_lm = res_pose.pose_landmarks.landmark
            if face_lm is not None and self.overlay_mode in ("ARMS", "FACE"):
                self._draw_pose_overlay(
                    frame, pose_lm, face_lm, w, h
                )

        quality = self.evaluate_frame_quality(
            frame,
            face_lm,
            pose_lm,
            w,
            h
        )

        framing = self.evaluate_framing(
            face_lm,
            pose_lm,
            w,
            h
        )

        return frame, {
            "face": face_lm,
            "pose": pose_lm,
            "quality": quality,
            "framing": framing
        }

    # ============================================================
    # QUALITY
    # ============================================================

    def evaluate_frame_quality(self, frame, face_lm, pose_lm, w, h):
        brightness = self._brightness_score(frame)
        sharpness = self._sharpness_score(frame)

        face_position = (
            self._face_position_score(face_lm, w, h)
            if face_lm is not None else 0.0
        )

        head_alignment = (
            self._head_alignment_score(face_lm)
            if face_lm is not None else 0.0
        )

        pose_visibility = self._pose_visibility_score(
            pose_lm
        )

        # Quality facial: NO depende de que los brazos estén visibles.
        if face_lm is not None:
            face_quality = (
                brightness * 0.25 +
                sharpness * 0.20 +
                face_position * 0.30 +
                head_alignment * 0.25
            )
        else:
            face_quality = 0.0

        # Quality motor: depende de las referencias necesarias
        # para brazos, no de la alineación facial.
        if pose_lm is not None:
            arms_quality = (
                brightness * 0.25 +
                sharpness * 0.15 +
                pose_visibility * 0.60
            )
        else:
            arms_quality = 0.0

        # Calidad general de la cámara, útil como referencia técnica.
        general_quality = (
            brightness * 0.35 +
            sharpness * 0.25 +
            (face_position if face_lm is not None else 0.0) * 0.20 +
            pose_visibility * 0.20
        )

        return {
            "overall": float(np.clip(general_quality, 0.0, 1.0)),
            "face": float(np.clip(face_quality, 0.0, 1.0)),
            "arms": float(np.clip(arms_quality, 0.0, 1.0)),
            "brightness": float(brightness),
            "sharpness": float(sharpness),
            "face_position": float(face_position),
            "head_alignment": float(head_alignment),
            "pose_visibility": float(pose_visibility),
            "status": self._quality_status(
                max(face_quality, arms_quality)
            )
        }

    def _empty_quality(self):
        return {
            "overall": 0.0,
            "face": 0.0,
            "arms": 0.0,
            "brightness": 0.0,
            "sharpness": 0.0,
            "face_position": 0.0,
            "head_alignment": 0.0,
            "pose_visibility": 0.0,
            "status": "INVALID"
        }

    @staticmethod
    def _quality_status(score):
        if score >= 0.90:
            return "EXCELLENT"
        if score >= 0.80:
            return "GOOD"
        if score >= 0.65:
            return "ACCEPTABLE"
        return "LOW"

    def _brightness_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))

        # No exigimos un brillo exacto. Usamos una zona amplia y
        # penalización progresiva para evitar scores artificialmente bajos.
        if mean < 45:
            base = mean / 45.0
        elif mean <= 190:
            base = 1.0
        else:
            base = max(0.0, 1.0 - (mean - 190.0) / 90.0)

        # Una imagen completamente plana puede ser poco útil.
        contrast_bonus = min(std / 55.0, 1.0)

        score = 0.80 * base + 0.20 * contrast_bonus
        return float(np.clip(score, 0.0, 1.0))

    def _sharpness_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

        # Curva suave. No se declara 'mala' una webcam simplemente
        # porque tenga menos varianza que una cámara profesional.
        score = variance / (variance + 120.0)
        return float(np.clip(score, 0.0, 1.0))

    def _face_position_score(self, lm, w, h):
        xs = np.array([p.x for p in lm], dtype=float)
        ys = np.array([p.y for p in lm], dtype=float)

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        # Distancia del centro facial al centro del encuadre.
        center_distance = np.sqrt(
            (cx - 0.5) ** 2 +
            (cy - 0.48) ** 2
        )

        center_score = max(
            0.0,
            1.0 - center_distance / 0.50
        )

        face_height = max_y - min_y

        # Mejor que un único threshold rígido.
        if 0.20 <= face_height <= 0.60:
            size_score = 1.0
        elif 0.15 <= face_height < 0.20:
            size_score = 0.75
        elif 0.60 < face_height <= 0.70:
            size_score = 0.80
        else:
            size_score = 0.35

        return float(np.clip(
            0.55 * center_score +
            0.45 * size_score,
            0.0,
            1.0
        ))

    def _head_alignment_score(self, lm):
        left_eye = lm[33]
        right_eye = lm[263]
        nose = lm[1]

        eye_mid_x = (
            left_eye.x + right_eye.x
        ) / 2.0
        eye_mid_y = (
            left_eye.y + right_eye.y
        ) / 2.0

        eye_span = abs(
            right_eye.x - left_eye.x
        )

        if eye_span < 1e-5:
            return 0.0

        # 1) Centrado horizontal de la nariz respecto a los ojos.
        horizontal_dev = abs(
            nose.x - eye_mid_x
        ) / eye_span

        # 2) Inclinación (roll) aproximada a partir de la línea ocular.
        roll = abs(
            right_eye.y - left_eye.y
        ) / eye_span

        horizontal_score = 1.0 / (
            1.0 + 7.0 * horizontal_dev
        )

        roll_score = 1.0 / (
            1.0 + 10.0 * roll
        )

        return float(np.clip(
            0.70 * horizontal_score +
            0.30 * roll_score,
            0.0,
            1.0
        ))

    def _pose_visibility_score(self, pose_lm):
        if pose_lm is None:
            return 0.0

        required = [11, 12, 13, 14, 15, 16]
        values = [
            float(np.clip(pose_lm[i].visibility, 0.0, 1.0))
            for i in required
        ]

        return float(np.mean(values))

    # ============================================================
    # FRAMING / POSITIONING
    # ============================================================

    def evaluate_framing(self, face_lm, pose_lm, w, h):
        """
        Comprueba si el sujeto está correctamente encuadrado.

        FACE:
            Rostro centrado y de tamaño útil.

        ARMS:
            Ambos hombros, codos y muñecas visibles y dentro del
            campo de imagen con margen de seguridad.
        """
        face_ok = False

        if face_lm is not None:
            xs = [p.x for p in face_lm]
            ys = [p.y for p in face_lm]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            face_height = max_y - min_y
            face_center_x = (min_x + max_x) / 2.0
            face_center_y = (min_y + max_y) / 2.0

            face_ok = (
                0.18 <= face_height <= 0.65
                and 0.28 <= face_center_x <= 0.72
                and 0.20 <= face_center_y <= 0.70
            )

        arm_indices = [11, 12, 13, 14, 15, 16]
        arm_visibility_values = []
        arm_inside_values = []

        if pose_lm is not None:
            for i in arm_indices:
                p = pose_lm[i]
                visible = float(p.visibility) >= 0.65
                inside = (
                    0.03 <= p.x <= 0.97
                    and
                    0.05 <= p.y <= 0.97
                )

                arm_visibility_values.append(visible)
                arm_inside_values.append(inside)

        arm_visible_count = sum(arm_visibility_values)
        arm_inside_count = sum(arm_inside_values)

        arms_ok = (
            len(arm_visibility_values) == 6
            and arm_visible_count == 6
            and arm_inside_count == 6
        )

        # Distancia al borde para explicar mejor el problema.
        min_margin = 0.0
        if pose_lm is not None:
            margins = []
            for i in arm_indices:
                p = pose_lm[i]
                margins.extend([
                    p.x,
                    1.0 - p.x,
                    p.y,
                    1.0 - p.y
                ])
            min_margin = float(max(0.0, min(margins)))

        return {
            "face_ready": face_ok,
            "arms_ready": arms_ok,
            "arms_visible_count": int(arm_visible_count),
            "arms_inside_count": int(arm_inside_count),
            "min_arm_margin": min_margin,
            "ready_for_face": face_ok,
            "ready_for_arms": arms_ok
        }

    # ============================================================
    # OVERLAYS
    # ============================================================

    def _draw_face_overlay(self, image, lm, w, h):
        """
        Overlay facial clínico de alta visibilidad.

        No dibujamos los 468 landmarks: eso genera ruido visual.
        Mostramos únicamente puntos anatómicos relevantes para
        asimetría facial, ojos, nariz y boca.
        """
        # Línea media
        cv2.line(
            image,
            (int(lm[168].x * w), int(lm[168].y * h)),
            (int(lm[152].x * w), int(lm[152].y * h)),
            self.C_GRY, 2, cv2.LINE_AA
        )

        # Puntos clínicos relevantes
        key_points = [
            1, 33, 133, 263, 362,       # nariz / ojos
            61, 291, 66, 296,            # comisuras / cejas
            0, 17, 152                    # boca / mentón
        ]

        for idx in key_points:
            p = lm[idx]
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(
                image, (x, y), 4,
                self.C_CYA, -1,
                cv2.LINE_AA
            )
            cv2.circle(
                image, (x, y), 6,
                (20, 30, 40), 1,
                cv2.LINE_AA
            )

        # Ojos: contorno simplificado y muy visible
        left_eye = np.array([
            (int(lm[i].x * w), int(lm[i].y * h))
            for i in [33, 159, 145, 133]
        ], np.int32)
        right_eye = np.array([
            (int(lm[i].x * w), int(lm[i].y * h))
            for i in [263, 386, 374, 362]
        ], np.int32)

        cv2.polylines(
            image, [left_eye], True,
            self.C_GREEN, 2, cv2.LINE_AA
        )
        cv2.polylines(
            image, [right_eye], True,
            self.C_GREEN, 2, cv2.LINE_AA
        )

        # Boca
        lips_idx = [
            61, 185, 40, 39, 37,
            0, 267, 269, 270, 409,
            291, 375, 321, 405,
            314, 17, 84, 181, 91, 146
        ]

        pts = np.array([
            (int(lm[i].x * w), int(lm[i].y * h))
            for i in lips_idx
        ], np.int32)

        cv2.polylines(
            image, [pts], True,
            self.C_WHT, 2, cv2.LINE_AA
        )

        # Vectores de las dos hemicaras
        cv2.line(
            image,
            (int(lm[296].x * w), int(lm[296].y * h)),
            (int(lm[291].x * w), int(lm[291].y * h)),
            self.C_CYA, 2, cv2.LINE_AA
        )
        cv2.line(
            image,
            (int(lm[66].x * w), int(lm[66].y * h)),
            (int(lm[61].x * w), int(lm[61].y * h)),
            self.C_CYA, 2, cv2.LINE_AA
        )

    def _draw_pose_overlay(self, image, pose_lm, face_lm, w, h):
        """
        Overlay motor: skeleton simplificado + landmarks de hombros,
        codos y muñecas. Más visible que unir solamente nariz-muñeca.
        """
        # Segmentos anatómicos relevantes
        segments = [
            (11, 13), (13, 15),  # izquierdo
            (12, 14), (14, 16),  # derecho
            (11, 12),             # cintura escapular
        ]

        for a, b in segments:
            pa, pb = pose_lm[a], pose_lm[b]

            if pa.visibility > 0.55 and pb.visibility > 0.55:
                p1 = (int(pa.x * w), int(pa.y * h))
                p2 = (int(pb.x * w), int(pb.y * h))

                cv2.line(
                    image, p1, p2,
                    self.C_BLU, 4,
                    cv2.LINE_AA
                )

        # Landmarks grandes y contrastados
        labels = {
            11: "H",
            12: "H",
            13: "C",
            14: "C",
            15: "M",
            16: "M"
        }

        for idx, label in labels.items():
            p = pose_lm[idx]

            if p.visibility > 0.45:
                x, y = int(p.x * w), int(p.y * h)

                cv2.circle(
                    image, (x, y), 8,
                    self.C_GREEN if p.visibility >= 0.65
                    else self.C_YELLOW,
                    -1, cv2.LINE_AA
                )
                cv2.circle(
                    image, (x, y), 11,
                    (20, 30, 40),
                    2, cv2.LINE_AA
                )

                cv2.putText(
                    image,
                    label,
                    (x + 10, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    self.C_WHT,
                    1,
                    cv2.LINE_AA
                )

        # Vector nariz -> muñecas para conservar tu referencia original
        if face_lm is not None:
            nose = face_lm[1]
            p_n = (
                int(nose.x * w),
                int(nose.y * h)
            )

            for idx in (15, 16):
                wrist = pose_lm[idx]
                if wrist.visibility > 0.60:
                    p_w = (
                        int(wrist.x * w),
                        int(wrist.y * h)
                    )
                    cv2.line(
                        image, p_n, p_w,
                        self.C_CYA, 2,
                        cv2.LINE_AA
                    )

    def draw_setup_overlay(self, image, status, arms=False):
        """
        Overlay simple para guiar al usuario durante el encuadre inicial.
        """
        h, w = image.shape[:2]

        if arms:
            color = self.C_GREEN if status["ready_for_arms"] else self.C_YELLOW
            cv2.rectangle(
                image,
                (int(w * 0.03), int(h * 0.04)),
                (int(w * 0.97), int(h * 0.97)),
                color,
                2
            )
            return

        color = self.C_GREEN if status["ready_for_face"] else self.C_YELLOW
        cv2.rectangle(
            image,
            (int(w * 0.18), int(h * 0.08)),
            (int(w * 0.82), int(h * 0.78)),
            color,
            2
        )

    def close(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass

        try:
            self.pose.close()
        except Exception:
            pass