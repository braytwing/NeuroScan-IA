import math
from collections import deque

import numpy as np


class ClinicalLogic:
    """
    Motor heurístico de NeuroScan IA.

    NOTA:
    Los thresholds y scores son parámetros experimentales del prototipo.
    No representan sensibilidad, especificidad, precisión diagnóstica
    ni probabilidad de ACV.
    """

    def __init__(self):

        # ============================================================
        # CARA
        # ============================================================

        self.GLOBAL_ASYMMETRY_THRESHOLD = 0.25

        self.THRESH_HEMI = 0.15
        self.THRESH_LIFT = 0.10
        self.THRESH_DIAG = 0.20

        self.SMILE_ACTIVATION_RATIO = 0.45

        # ============================================================
        # OJOS
        # ============================================================

        self.PTOSIS_THRESHOLD = 0.12

        # ============================================================
        # VENTANAS TEMPORALES
        # ============================================================

        self.FACE_WINDOW_SIZE = 45
        self.EYE_WINDOW_SIZE = 45

        self.face_scores = deque(
            maxlen=self.FACE_WINDOW_SIZE
        )

        self.eye_scores = deque(
            maxlen=self.EYE_WINDOW_SIZE
        )

        # ============================================================
        # BASELINE FACIAL
        # ============================================================

        self.baseline_face_asym = 0.0
        self.is_calibrated = False

        # ============================================================
        # BRAZOS
        # ============================================================

        # Estos son parámetros de ingeniería iniciales.
        self.ARM_DRIFT_THRESHOLD = 0.18

        self.ARM_ELBOW_THRESHOLD = 0.10

        self.ARM_ANGLE_THRESHOLD = 12.0

        self.ARM_ASYMMETRY_THRESHOLD = 0.10

        self.ARM_PERSISTENCE_THRESHOLD = 0.55

        self.ARM_MIN_VALID_FRAMES = 12

        self.ARM_START_MARGIN = 0.06

        self.ARM_VISIBILITY_THRESHOLD = 0.60

        # ============================================================
        # ESTADO DEL EXAMEN DE BRAZOS
        # ============================================================

        self.arm_test_active = False

        self.arm_baseline = None

        self.arm_samples = deque(
            maxlen=120
        )

        self.arm_valid_frames = 0

        self.arm_total_frames = 0

        self.last_arm_result = None

    # ================================================================
    # UTILIDADES
    # ================================================================

    @staticmethod
    def _distance(a, b):

        return math.dist(
            (a.x, a.y),
            (b.x, b.y)
        )

    @staticmethod
    def _point(point):

        return np.array(
            [point.x, point.y],
            dtype=np.float32
        )

    @staticmethod
    def _angle(a, b, c):

        ba = a - b
        bc = c - b

        denominator = (
            np.linalg.norm(ba)
            *
            np.linalg.norm(bc)
        )

        if denominator < 1e-6:
            return 0.0

        cosine = float(
            np.dot(ba, bc)
            /
            denominator
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        return math.degrees(
            math.acos(cosine)
        )

    # ================================================================
    # EAR
    # ================================================================

    def _calculate_ear(
        self,
        lm,
        top,
        bottom,
        left,
        right
    ):

        vertical = self._distance(
            lm[top],
            lm[bottom]
        )

        horizontal = self._distance(
            lm[left],
            lm[right]
        )

        if horizontal <= 1e-6:
            return 0.0

        return (
            vertical /
            horizontal
        )

    # ================================================================
    # EVALUACIÓN FACIAL
    # ================================================================

    def evaluate_face(
        self,
        face_lm
    ):

        if face_lm is None:

            return {
                "score": 0.0,
                "ptosis": 0.0,
                "is_smiling": False,
                "face_alert": False,
                "eyes_alert": False
            }

        eye_span = max(
            self._distance(
                face_lm[33],
                face_lm[263]
            ),
            1e-5
        )

        # ------------------------------------------------------------
        # OJOS
        # ------------------------------------------------------------

        ear_left = self._calculate_ear(
            face_lm,
            159,
            145,
            33,
            133
        )

        ear_right = self._calculate_ear(
            face_lm,
            386,
            374,
            362,
            263
        )

        ptosis_factor = abs(
            ear_left -
            ear_right
        )

        # ------------------------------------------------------------
        # HEMICARA
        # ------------------------------------------------------------

        hemi_left = self._distance(
            face_lm[296],
            face_lm[291]
        )

        hemi_right = self._distance(
            face_lm[66],
            face_lm[61]
        )

        factor_hemi = (
            abs(
                hemi_left -
                hemi_right
            )
            /
            eye_span
        )

        # ------------------------------------------------------------
        # COMISURAS
        # ------------------------------------------------------------

        lift_left = (
            face_lm[0].y -
            face_lm[291].y
        )

        lift_right = (
            face_lm[0].y -
            face_lm[61].y
        )

        factor_lift = (
            abs(
                lift_left -
                lift_right
            )
            /
            eye_span
        )

        # ------------------------------------------------------------
        # DIAGONAL
        # ------------------------------------------------------------

        diagonal_left = self._distance(
            face_lm[359],
            face_lm[291]
        )

        diagonal_right = self._distance(
            face_lm[130],
            face_lm[61]
        )

        factor_diag = (
            abs(
                diagonal_left -
                diagonal_right
            )
            /
            (
                (
                    diagonal_left +
                    diagonal_right
                )
                /
                2
                +
                1e-5
            )
        )

        # ------------------------------------------------------------
        # SONRISA
        # ------------------------------------------------------------

        mouth_width = self._distance(
            face_lm[291],
            face_lm[61]
        )

        smile_ratio = (
            mouth_width /
            eye_span
        )

        is_smiling = (
            smile_ratio >
            self.SMILE_ACTIVATION_RATIO
        )

        # ------------------------------------------------------------
        # SCORE
        # ------------------------------------------------------------

        raw_score = (
            factor_hemi * 0.40
            +
            factor_lift * 0.30
            +
            factor_diag * 0.30
        )

        if self.is_calibrated:

            dynamic_score = abs(
                raw_score -
                self.baseline_face_asym
            )

        else:

            dynamic_score = raw_score

        any_flag = (
            factor_hemi > self.THRESH_HEMI
            or
            factor_lift > self.THRESH_LIFT
            or
            factor_diag > self.THRESH_DIAG
        )

        face_alert = (
            (
                dynamic_score >
                self.GLOBAL_ASYMMETRY_THRESHOLD
                or
                any_flag
            )
            and
            is_smiling
        )

        eyes_alert = (
            ptosis_factor >
            self.PTOSIS_THRESHOLD
        )

        return {

            "score":
                float(dynamic_score),

            "ptosis":
                float(ptosis_factor),

            "is_smiling":
                bool(is_smiling),

            "face_alert":
                bool(face_alert),

            "eyes_alert":
                bool(eyes_alert),

            "factor_hemi":
                float(factor_hemi),

            "factor_lift":
                float(factor_lift),

            "factor_diag":
                float(factor_diag),

            "ear_left":
                float(ear_left),

            "ear_right":
                float(ear_right)
        }

    # ================================================================
    # FACE TEMPORAL
    # ================================================================

    def update_face_temporal(
        self,
        face_lm
    ):

        result = self.evaluate_face(
            face_lm
        )

        if face_lm is not None:

            self.face_scores.append(
                result["score"]
            )

            self.eye_scores.append(
                result["ptosis"]
            )

        if not self.face_scores:

            return {
                **result,
                "face_score_temporal": 0.0,
                "eye_score_temporal": 0.0,
                "face_alert_temporal": False,
                "eyes_alert_temporal": False,
                "samples": 0
            }

        face_values = list(
            self.face_scores
        )

        eye_values = list(
            self.eye_scores
        )

        face_p75 = float(
            np.percentile(
                face_values,
                75
            )
        )

        eye_p75 = float(
            np.percentile(
                eye_values,
                75
            )
        )

        result.update({

            "face_score_temporal":
                face_p75,

            "eye_score_temporal":
                eye_p75,

            "face_alert_temporal":
                (
                    len(face_values) >= 12
                    and
                    face_p75 >
                    self.GLOBAL_ASYMMETRY_THRESHOLD
                    and
                    result["is_smiling"]
                ),

            "eyes_alert_temporal":
                (
                    len(eye_values) >= 12
                    and
                    eye_p75 >
                    self.PTOSIS_THRESHOLD
                ),

            "samples":
                len(face_values)
        })

        return result

    def reset_face_windows(self):

        self.face_scores.clear()
        self.eye_scores.clear()

    # ================================================================
    # BASELINE
    # ================================================================

    def calibrate_baseline(
        self,
        scores_list
    ):

        if scores_list:

            self.baseline_face_asym = float(
                np.median(scores_list)
            )

            self.is_calibrated = True

        return self.baseline_face_asym

    # ================================================================
    # BRAZOS: GEOMETRÍA
    # ================================================================

    def _arm_geometry(
        self,
        pose_lm,
        side
    ):

        if side == "left":

            shoulder_idx = 11
            elbow_idx = 13
            wrist_idx = 15

        else:

            shoulder_idx = 12
            elbow_idx = 14
            wrist_idx = 16

        shoulder_lm = pose_lm[
            shoulder_idx
        ]

        elbow_lm = pose_lm[
            elbow_idx
        ]

        wrist_lm = pose_lm[
            wrist_idx
        ]

        visibility = min(

            float(
                getattr(
                    shoulder_lm,
                    "visibility",
                    0.0
                )
            ),

            float(
                getattr(
                    elbow_lm,
                    "visibility",
                    0.0
                )
            ),

            float(
                getattr(
                    wrist_lm,
                    "visibility",
                    0.0
                )
            )
        )

        shoulder = self._point(
            shoulder_lm
        )

        elbow = self._point(
            elbow_lm
        )

        wrist = self._point(
            wrist_lm
        )

        return {

            "visibility":
                visibility,

            "wrist_elevation":
                float(
                    shoulder[1] -
                    wrist[1]
                ),

            "elbow_elevation":
                float(
                    shoulder[1] -
                    elbow[1]
                ),

            "angle":
                float(
                    self._angle(
                        shoulder,
                        elbow,
                        wrist
                    )
                )
        }

    # ================================================================
    # BRAZOS: POSICIÓN INICIAL
    # ================================================================

    def check_arm_position(
        self,
        pose_lm
    ):

        if pose_lm is None:

            return (
                False,
                "No se detecta el cuerpo.",
                0
            )

        left = self._arm_geometry(
            pose_lm,
            "left"
        )

        right = self._arm_geometry(
            pose_lm,
            "right"
        )

        left_visible = (
            left["visibility"]
            >= 0.65
        )

        right_visible = (
            right["visibility"]
            >= 0.65
        )

        visible_count = (
            (3 if left_visible else 0)
            +
            (3 if right_visible else 0)
        )

        if not (
            left_visible
            and
            right_visible
        ):

            return (
                False,
                (
                    f"Landmarks visibles "
                    f"{visible_count}/6. "
                    "Aléjese ligeramente."
                ),
                visible_count
            )

        left_up = (
            left["wrist_elevation"]
            >= self.ARM_START_MARGIN
        )

        right_up = (
            right["wrist_elevation"]
            >= self.ARM_START_MARGIN
        )

        if not (
            left_up
            and
            right_up
        ):

            return (
                False,
                (
                    "Levante ambos brazos "
                    "y manténgalos elevados."
                ),
                6
            )

        return (
            True,
            "Posición correcta.",
            6
        )

    # ================================================================
    # BRAZOS: INICIO
    # ================================================================

    def start_arm_test(
        self,
        pose_lm
    ):

        ready, message, _ = (
            self.check_arm_position(
                pose_lm
            )
        )

        if not ready:

            return (
                False,
                message
            )

        left = self._arm_geometry(
            pose_lm,
            "left"
        )

        right = self._arm_geometry(
            pose_lm,
            "right"
        )

        self.arm_baseline = {

            "left_wrist":
                left["wrist_elevation"],

            "right_wrist":
                right["wrist_elevation"],

            "left_elbow":
                left["elbow_elevation"],

            "right_elbow":
                right["elbow_elevation"],

            "left_angle":
                left["angle"],

            "right_angle":
                right["angle"]
        }

        self.arm_samples.clear()

        self.arm_valid_frames = 0

        self.arm_total_frames = 0

        self.arm_test_active = True

        self.last_arm_result = None

        return (
            True,
            "Prueba iniciada."
        )

    # ================================================================
    # BRAZOS: ACTUALIZACIÓN
    # ================================================================

    def update_arm_test(
        self,
        pose_lm,
        face_lm=None
    ):

        if not self.arm_test_active:

            return self._empty_arm_result()

        self.arm_total_frames += 1

        if pose_lm is None:

            return self._arm_result()

        left = self._arm_geometry(
            pose_lm,
            "left"
        )

        right = self._arm_geometry(
            pose_lm,
            "right"
        )

        if min(
            left["visibility"],
            right["visibility"]
        ) < self.ARM_VISIBILITY_THRESHOLD:

            return self._arm_result()

        self.arm_valid_frames += 1

        # ------------------------------------------------------------
        # Descenso de muñeca
        # ------------------------------------------------------------

        left_drop = max(
            0.0,
            self.arm_baseline[
                "left_wrist"
            ]
            -
            left["wrist_elevation"]
        )

        right_drop = max(
            0.0,
            self.arm_baseline[
                "right_wrist"
            ]
            -
            right["wrist_elevation"]
        )

        # ------------------------------------------------------------
        # Descenso de codo
        # ------------------------------------------------------------

        left_elbow_drop = max(
            0.0,
            self.arm_baseline[
                "left_elbow"
            ]
            -
            left["elbow_elevation"]
        )

        right_elbow_drop = max(
            0.0,
            self.arm_baseline[
                "right_elbow"
            ]
            -
            right["elbow_elevation"]
        )

        # ------------------------------------------------------------
        # Cambio angular
        # ------------------------------------------------------------

        left_angle_change = abs(
            left["angle"]
            -
            self.arm_baseline[
                "left_angle"
            ]
        )

        right_angle_change = abs(
            right["angle"]
            -
            self.arm_baseline[
                "right_angle"
            ]
        )

        # ------------------------------------------------------------
        # Asimetría
        # ------------------------------------------------------------

        wrist_asymmetry = abs(
            left_drop -
            right_drop
        )

        elbow_asymmetry = abs(
            left_elbow_drop -
            right_elbow_drop
        )

        # ------------------------------------------------------------
        # Evidencia individual
        # ------------------------------------------------------------

        left_evidence = (

            left_drop >
            self.ARM_DRIFT_THRESHOLD

            and

            (
                left_elbow_drop >
                self.ARM_ELBOW_THRESHOLD

                or

                left_angle_change >
                self.ARM_ANGLE_THRESHOLD
            )
        )

        right_evidence = (

            right_drop >
            self.ARM_DRIFT_THRESHOLD

            and

            (
                right_elbow_drop >
                self.ARM_ELBOW_THRESHOLD

                or

                right_angle_change >
                self.ARM_ANGLE_THRESHOLD
            )
        )

        unilateral = (
            left_evidence
            !=
            right_evidence
        )

        self.arm_samples.append({

            "left_drop":
                left_drop,

            "right_drop":
                right_drop,

            "left_elbow_drop":
                left_elbow_drop,

            "right_elbow_drop":
                right_elbow_drop,

            "left_angle_change":
                left_angle_change,

            "right_angle_change":
                right_angle_change,

            "wrist_asymmetry":
                wrist_asymmetry,

            "elbow_asymmetry":
                elbow_asymmetry,

            "unilateral":
                unilateral
        })

        result = self._arm_result()

        self.last_arm_result = result

        return result

    # ================================================================
    # RESULTADO BRAZOS
    # ================================================================

    def _empty_arm_result(self):

        return {

            "drift_score": 0.0,

            "is_alert": False,

            "valid": False,

            "has_raised":
                self.arm_baseline is not None,

            "persistence": 0.0,

            "left_drop": 0.0,

            "right_drop": 0.0,

            "left_elbow_drop": 0.0,

            "right_elbow_drop": 0.0,

            "left_angle_change": 0.0,

            "right_angle_change": 0.0,

            "asymmetry": 0.0,

            "valid_frames":
                self.arm_valid_frames,

            "samples":
                len(self.arm_samples)
        }

    def _arm_result(self):

        if not self.arm_samples:

            return self._empty_arm_result()

        samples = list(
            self.arm_samples
        )

        def median(key):

            return float(
                np.median(
                    [
                        sample[key]
                        for sample in samples
                    ]
                )
            )

        left_drop = median(
            "left_drop"
        )

        right_drop = median(
            "right_drop"
        )

        left_elbow_drop = median(
            "left_elbow_drop"
        )

        right_elbow_drop = median(
            "right_elbow_drop"
        )

        left_angle = median(
            "left_angle_change"
        )

        right_angle = median(
            "right_angle_change"
        )

        asymmetry = abs(
            left_drop -
            right_drop
        )

        persistence = float(
            np.mean(
                [
                    bool(
                        s["unilateral"]
                    )
                    for s in samples
                ]
            )
        )

        # ------------------------------------------------------------
        # COMPONENTES DEL SCORE
        # ------------------------------------------------------------

        wrist_component = min(
            max(
                left_drop,
                right_drop
            )
            /
            max(
                self.ARM_DRIFT_THRESHOLD,
                1e-6
            ),
            2.0
        ) / 2.0

        elbow_component = min(
            max(
                left_elbow_drop,
                right_elbow_drop
            )
            /
            max(
                self.ARM_ELBOW_THRESHOLD,
                1e-6
            ),
            2.0
        ) / 2.0

        angle_component = min(
            max(
                left_angle,
                right_angle
            )
            /
            max(
                self.ARM_ANGLE_THRESHOLD,
                1e-6
            ),
            2.0
        ) / 2.0

        asymmetry_component = min(
            asymmetry
            /
            max(
                self.ARM_ASYMMETRY_THRESHOLD,
                1e-6
            ),
            2.0
        ) / 2.0

        drift_score = (

            wrist_component * 0.40

            +

            elbow_component * 0.20

            +

            angle_component * 0.15

            +

            asymmetry_component * 0.15

            +

            persistence * 0.10
        )

        # ------------------------------------------------------------
        # DECISIÓN FINAL
        # ------------------------------------------------------------

        enough_frames = (
            self.arm_valid_frames
            >=
            self.ARM_MIN_VALID_FRAMES
        )

        persistent = (
            persistence
            >=
            self.ARM_PERSISTENCE_THRESHOLD
        )

        significant_asymmetry = (
            asymmetry
            >=
            self.ARM_ASYMMETRY_THRESHOLD
        )

        strong_drop = (
            max(
                left_drop,
                right_drop
            )
            >=
            self.ARM_DRIFT_THRESHOLD * 1.35
        )

        is_alert = (

            enough_frames

            and

            persistent

            and

            (
                significant_asymmetry
                or
                strong_drop
            )
        )

        return {

            "drift_score":
                float(drift_score),

            "is_alert":
                bool(is_alert),

            "valid":
                self.arm_valid_frames > 0,

            "has_raised":
                self.arm_baseline is not None,

            "persistence":
                float(persistence),

            "left_drop":
                float(left_drop),

            "right_drop":
                float(right_drop),

            "left_elbow_drop":
                float(left_elbow_drop),

            "right_elbow_drop":
                float(right_elbow_drop),

            "left_angle_change":
                float(left_angle),

            "right_angle_change":
                float(right_angle),

            "asymmetry":
                float(asymmetry),

            "valid_frames":
                int(self.arm_valid_frames),

            "samples":
                int(len(samples))
        }

    # ================================================================
    # FIN DE PRUEBA
    # ================================================================

    def finish_arm_test(self):

        result = self._arm_result()

        self.arm_test_active = False

        self.last_arm_result = result

        return result

    # ================================================================
    # COMPATIBILIDAD CON EL PROTOTIPO ORIGINAL
    # ================================================================

    def evaluate_arms(
        self,
        pose_lm,
        face_lm=None
    ):

        if not self.arm_test_active:

            if pose_lm is None:

                return self._empty_arm_result()

            ready, _, _ = (
                self.check_arm_position(
                    pose_lm
                )
            )

            if not ready:

                return {
                    **self._empty_arm_result(),
                    "has_raised": False
                }

            started, _ = (
                self.start_arm_test(
                    pose_lm
                )
            )

            if not started:

                return self._empty_arm_result()

        return self.update_arm_test(
            pose_lm,
            face_lm
        )
