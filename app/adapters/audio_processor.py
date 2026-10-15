"""
BabyCry AI - Adaptateur Audio (Librosa)
Implémentation du port AudioProcessorPort.
"""
import io
import numpy as np
import librosa
import soundfile as sf
from app.ports import AudioProcessorPort
from app.domain import MFCCFeatures
from config import SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH, AUDIO_DURATION


class LibrosaAudioProcessor(AudioProcessorPort):
    """Traitement audio via Librosa : prétraitement + extraction MFCC."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mfcc: int = N_MFCC,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        duration: float = AUDIO_DURATION,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration

    def preprocess(self, audio_data: bytes) -> bytes:
        """
        Prétraitement audio :
        1. Chargement depuis bytes
        2. Rééchantillonnage à SAMPLE_RATE
        3. Normalisation d'amplitude
        4. Padding/truncation à AUDIO_DURATION
        """
        # Charger l'audio depuis les bytes
        audio_buffer = io.BytesIO(audio_data)
        y, sr = sf.read(audio_buffer)

        # Convertir en mono si stéréo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Rééchantillonner
        if sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)

        # Normaliser l'amplitude
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # Padding ou truncation à la durée cible
        target_length = int(self.sample_rate * self.duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]

        # Re-sérialiser en bytes
        output_buffer = io.BytesIO()
        sf.write(output_buffer, y, self.sample_rate, format='WAV')
        return output_buffer.getvalue()

    def extract_mfcc(self, audio_data: bytes) -> MFCCFeatures:
        """
        Extraction des caractéristiques MFCC :
        - N_MFCC coefficients cepstraux
        - Delta et delta-delta MFCC
        - Moyennage temporel → vecteur fixe
        """
        # Charger l'audio prétraité
        audio_buffer = io.BytesIO(audio_data)
        y, sr = sf.read(audio_buffer)
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        # Extraire les MFCC
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Delta MFCC (vitesse)
        mfcc_delta = librosa.feature.delta(mfcc)

        # Delta-Delta MFCC (accélération)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Concaténer et moyenner sur l'axe temporel
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.mean(mfcc_delta2, axis=1),
            np.std(mfcc, axis=1),      # écart-type pour plus de robustesse
        ])

        return MFCCFeatures(
            coefficients=tuple(features.tolist()),
            n_mfcc=self.n_mfcc,
            sample_rate=sr,
        )


class AudioFeatureExtractor:
    """
    Extracteur de features audio avancé pour le training.
    Utilisé dans le pipeline d'apprentissage.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.processor = LibrosaAudioProcessor(sample_rate=sample_rate, n_mfcc=n_mfcc)

    def extract_features_from_file(self, file_path: str) -> np.ndarray:
        """Extrait les features MFCC depuis un fichier audio."""
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        processed = self.processor.preprocess(audio_data)
        features = self.processor.extract_mfcc(processed)
        return np.array(features.coefficients)

    def extract_batch(self, file_paths: list) -> np.ndarray:
        """Extrait les features pour un batch de fichiers."""
        all_features = []
        for path in file_paths:
            try:
                feat = self.extract_features_from_file(path)
                all_features.append(feat)
            except Exception as e:
                print(f"Erreur sur {path}: {e}")
                continue
        return np.array(all_features) if all_features else np.array([])
