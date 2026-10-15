"""
BabyCry AI - Analyseur audio de demonstration
Analyse deterministe d'un WAV PCM sans dependances lourdes.
"""
from __future__ import annotations

import io
import math
import struct
import wave

from app.domain import CryPrediction, CryType


class DemoCryAnalyzer:
    """Analyse heuristique pour le mode demo."""

    def analyze(self, audio_data: bytes) -> CryPrediction:
        samples, sample_rate = self._read_wav(audio_data)
        if not samples:
            return self._build_prediction(CryType.INCONFORT, 0.35, "audio trop court")

        duration = len(samples) / sample_rate if sample_rate else 0.0
        abs_samples = [abs(sample) for sample in samples]
        peak = max(abs_samples)
        rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
        zero_crossings = sum(
            1
            for index in range(1, len(samples))
            if (samples[index - 1] <= 0 < samples[index]) or (samples[index - 1] >= 0 > samples[index])
        )
        zcr = zero_crossings / max(len(samples) - 1, 1)

        chunk_size = max(int(sample_rate * 0.25), 1)
        chunk_rms = []
        for start in range(0, len(samples), chunk_size):
            chunk = samples[start : start + chunk_size]
            if not chunk:
                continue
            chunk_rms.append(math.sqrt(sum(sample * sample for sample in chunk) / len(chunk)))

        active_chunks = sum(1 for value in chunk_rms if value > 0.08)
        activity_ratio = active_chunks / max(len(chunk_rms), 1)
        dynamic_range = peak - rms

        scores = {
            CryType.DOULEUR: (rms * 1.4) + (peak * 0.9) + (zcr * 5.5),
            CryType.FAIM: (activity_ratio * 1.1) + (0.7 - abs(rms - 0.16)) + (0.6 - abs(zcr - 0.08) * 2),
            CryType.FATIGUE: ((1 - activity_ratio) * 1.3) + (0.22 - min(rms, 0.22)) + (0.05 - min(zcr, 0.05)),
            CryType.PEUR: (peak * 0.8) + (dynamic_range * 1.6) + (zcr * 3.2),
            CryType.INCONFORT: (activity_ratio * 0.8) + (dynamic_range * 0.9) + (0.35 - abs(rms - 0.12)),
        }

        best_type = max(scores, key=scores.get)
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - (sorted_scores[1] if len(sorted_scores) > 1 else 0.0)
        confidence = min(0.98, max(0.45, 0.55 + margin / 2.5))

        reason = f"rms={rms:.3f}, zcr={zcr:.3f}, activite={activity_ratio:.2f}, duree={duration:.1f}s"
        return self._build_prediction(best_type, confidence, reason, scores)

    def _build_prediction(
        self,
        cry_type: CryType,
        confidence: float,
        reason: str,
        scores: dict | None = None,
    ) -> CryPrediction:
        normalized_scores = scores or {cry_type: confidence}
        total = sum(max(value, 0.0) for value in normalized_scores.values()) or 1.0
        probabilities = {
            key.value: round(max(value, 0.0) / total, 4)
            for key, value in normalized_scores.items()
        }
        return CryPrediction(
            cry_type=cry_type,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            advice=reason,
        )

    def _read_wav(self, audio_data: bytes) -> tuple[list[float], int]:
        with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)

        if sample_width != 2:
            raise ValueError("Le mode demo attend un WAV PCM 16-bit.")

        frame_format = f"<{frame_count * channels}h"
        ints = struct.unpack(frame_format, raw_frames)

        if channels > 1:
            mono = []
            for index in range(0, len(ints), channels):
                mono.append(sum(ints[index : index + channels]) / channels)
            ints = mono

        samples = [max(-1.0, min(1.0, value / 32768.0)) for value in ints]
        return samples, sample_rate
