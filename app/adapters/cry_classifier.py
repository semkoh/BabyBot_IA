"""
BabyCry AI - Adaptateur Classificateur de Pleurs
Implémentation du port CryClassifierPort.
Modèle : SVM + Random Forest ensemble, entraîné sur Donate-a-Cry.
"""
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.ports import CryClassifierPort
from app.domain import CryPrediction, CryType, MFCCFeatures, TrainingMetrics
from config import MODEL_DIR, CRY_CLASSES


class CryClassifierAdapter(CryClassifierPort):
    """
    Classificateur de pleurs bébé par ensemble SVM + Random Forest.
    Features : MFCC + Delta + Delta-Delta + statistiques.
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(MODEL_DIR / "cry_classifier.pkl")
        self.scaler_path = str(MODEL_DIR / "cry_scaler.pkl")
        self.model = None
        self.scaler = StandardScaler()
        self._load_or_init()

    def _load_or_init(self):
        """Charge le modèle existant ou initialise un nouveau."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            # Initialiser l'ensemble de classificateurs
            svm = SVC(
                kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=42
            )
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=20,
                min_samples_split=5, random_state=42
            )
            self.model = VotingClassifier(
                estimators=[('svm', svm), ('rf', rf)],
                voting='soft',
                weights=[0.4, 0.6],
            )

    def predict(self, features: MFCCFeatures) -> CryPrediction:
        """Prédit le type de pleur à partir des MFCC."""
        X = np.array(features.coefficients).reshape(1, -1)

        # Normaliser les features
        X_scaled = self.scaler.transform(X)

        # Prédiction avec probabilités
        proba = self.model.predict_proba(X_scaled)[0]
        predicted_class = int(np.argmax(proba))
        confidence = float(proba[predicted_class])

        # Mapper vers le type de pleur
        cry_type = CryType(CRY_CLASSES[predicted_class])

        # Probabilités par classe
        probabilities = {
            CRY_CLASSES[i]: float(p) for i, p in enumerate(proba)
        }

        return CryPrediction(
            cry_type=cry_type,
            confidence=confidence,
            probabilities=probabilities,
        )

    def train(self, features_list: list, labels: list) -> TrainingMetrics:
        """
        Entraîne le classificateur sur un dataset de features MFCC.
        
        Args:
            features_list: Liste de vecteurs MFCC (np.ndarray)
            labels: Liste d'entiers (indices de CRY_CLASSES)
        """
        X = np.array(features_list)
        y = np.array(labels)

        # Normalisation
        X_scaled = self.scaler.fit_transform(X)

        # Validation croisée
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')

        # Entraînement final sur tout le dataset
        self.model.fit(X_scaled, y)

        # Métriques
        y_pred = self.model.predict(X_scaled)
        metrics = TrainingMetrics(
            accuracy=float(accuracy_score(y, y_pred)),
            precision=float(precision_score(y, y_pred, average='weighted')),
            recall=float(recall_score(y, y_pred, average='weighted')),
            f1_score=float(f1_score(y, y_pred, average='weighted')),
            loss=float(1.0 - np.mean(cv_scores)),
        )

        # Sauvegarder le modèle
        self._save_model()

        return metrics

    def _save_model(self):
        """Persiste le modèle et le scaler."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
