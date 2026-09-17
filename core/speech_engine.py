import speech_recognition as sr
import threading
import re
import difflib
import time


class SpeechEngine:

    def __init__(self):

        self.recognizer = sr.Recognizer()

        self.target_phrase = (
            "la casa tiene una ventana grande"
        )

        self.result_text = ""

        self.is_listening = False
        self.has_finished = False

        # Esto representa similitud textual, NO una probabilidad clínica.
        self.similarity_score = 0.0

        self.word_coverage = 0.0
        self.sequence_similarity = 0.0

        self.recording_time = 0.0

        self.is_valid = False
        self.speech_alert = False

        self.error_message = ""

    # ============================================================
    # NORMALIZACIÓN
    # ============================================================

    @staticmethod
    def _normalize_text(text):

        text = text.lower().strip()

        text = re.sub(
            r"[^\w\sáéíóúüñ]",
            "",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text

    # ============================================================
    # INICIO
    # ============================================================

    def start_test(self):

        if self.is_listening:
            return

        self.is_listening = True
        self.has_finished = False

        self.result_text = ""

        self.similarity_score = 0.0
        self.word_coverage = 0.0
        self.sequence_similarity = 0.0

        self.recording_time = 0.0

        self.is_valid = False
        self.speech_alert = False

        self.error_message = ""

        thread = threading.Thread(
            target=self._listen_thread,
            daemon=True
        )

        thread.start()

    # ============================================================
    # HILO DE AUDIO
    # ============================================================

    def _listen_thread(self):

        start_time = time.time()

        try:

            with sr.Microphone() as source:

                # Ajuste de ruido ambiental
                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=0.5
                )

                audio = self.recognizer.listen(
                    source,
                    timeout=4,
                    phrase_time_limit=5
                )

            self.recording_time = (
                time.time() -
                start_time
            )

            # ----------------------------------------------------
            # RECONOCIMIENTO
            # ----------------------------------------------------

            recognized = (
                self.recognizer
                .recognize_google(
                    audio,
                    language="es-ES"
                )
            )

            self.result_text = (
                self._normalize_text(
                    recognized
                )
            )

            # ----------------------------------------------------
            # EVALUACIÓN
            # ----------------------------------------------------

            self._evaluate_text()

            self.is_valid = True

        except sr.WaitTimeoutError:

            self.result_text = ""

            self.error_message = (
                "No se detectó respuesta."
            )

            self.is_valid = False

        except sr.UnknownValueError:

            self.result_text = ""

            self.error_message = (
                "No se pudo interpretar el audio."
            )

            self.is_valid = False

        except sr.RequestError:

            self.result_text = ""

            self.error_message = (
                "No fue posible conectar con "
                "el servicio de reconocimiento."
            )

            self.is_valid = False

        except Exception as e:

            self.result_text = ""

            self.error_message = (
                f"Error de audio: {str(e)}"
            )

            self.is_valid = False

        finally:

            self.is_listening = False
            self.has_finished = True

    # ============================================================
    # EVALUACIÓN TEXTUAL
    # ============================================================

    def _evaluate_text(self):

        target = self._normalize_text(
            self.target_phrase
        )

        result = self._normalize_text(
            self.result_text
        )

        target_words = target.split()
        result_words = result.split()

        if not target_words or not result_words:

            self.similarity_score = 0.0

            self.word_coverage = 0.0
            self.sequence_similarity = 0.0

            self.speech_alert = True

            return

        # --------------------------------------------------------
        # COBERTURA DE PALABRAS
        # --------------------------------------------------------

        target_counts = {}

        for word in target_words:

            target_counts[word] = (
                target_counts.get(word, 0) + 1
            )

        result_counts = {}

        for word in result_words:

            result_counts[word] = (
                result_counts.get(word, 0) + 1
            )

        matched_words = 0

        for word, count in target_counts.items():

            matched_words += min(
                count,
                result_counts.get(
                    word,
                    0
                )
            )

        self.word_coverage = (
            matched_words /
            len(target_words)
        )

        # --------------------------------------------------------
        # SIMILITUD DE SECUENCIA
        # --------------------------------------------------------

        self.sequence_similarity = (
            difflib.SequenceMatcher(
                None,
                target,
                result
            ).ratio()
        )

        # --------------------------------------------------------
        # SCORE FINAL
        # --------------------------------------------------------

        self.similarity_score = (
            self.word_coverage * 0.55 +
            self.sequence_similarity * 0.45
        )

        # --------------------------------------------------------
        # REGLA PROVISIONAL
        #
        # Estos thresholds NO están clínicamente validados.
        # Sólo son un punto de partida técnico.
        # --------------------------------------------------------

        self.speech_alert = (
            self.similarity_score < 0.72
        )

    # ============================================================
    # RESULTADO
    # ============================================================

    def get_result(self):

        if not self.has_finished:

            return {
                "status": "RUNNING",
                "text": "",
                "score": 0.0,
                "is_valid": False,
                "alert": False
            }

        if not self.is_valid:

            return {
                "status": "INVALID",
                "text": self.result_text,
                "score": 0.0,
                "is_valid": False,
                "alert": False,
                "error": self.error_message
            }

        return {

            "status": "COMPLETED",

            "text": self.result_text,

            "score": float(
                self.similarity_score
            ),

            "word_coverage": float(
                self.word_coverage
            ),

            "sequence_similarity": float(
                self.sequence_similarity
            ),

            "is_valid": True,

            "alert": bool(
                self.speech_alert
            )
        }