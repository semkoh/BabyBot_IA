"""
BabyCry AI - Ports (Interfaces Hexagonales)
Ports d'entrée (driving) et ports de sortie (driven).
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from app.domain import (
    AudioSample, CryPrediction, ChatMessage, Feedback,
    Parent, KnowledgeDocument, MFCCFeatures, TrainingMetrics
)


# ═══════════════════════════════════════════════════════════════
#  PORTS D'ENTRÉE (Driving Ports / Use Cases)
# ═══════════════════════════════════════════════════════════════

class CryAnalysisPort(ABC):
    """Port d'entrée : Analyse des pleurs de bébé."""

    @abstractmethod
    async def analyze_cry(self, audio_data: bytes, parent_id: str) -> CryPrediction:
        """Analyse un enregistrement audio et retourne la prédiction."""
        ...

    @abstractmethod
    async def get_prediction_history(
        self, parent_id: str, limit: int = 20
    ) -> List[CryPrediction]:
        """Historique des prédictions pour un parent."""
        ...


class ChatbotPort(ABC):
    """Port d'entrée : Chatbot conseil parental."""

    @abstractmethod
    async def ask_question(
        self, question: str, parent_id: str, context: Optional[str] = None
    ) -> ChatMessage:
        """Pose une question au chatbot."""
        ...

    @abstractmethod
    async def get_chat_history(
        self, parent_id: str, limit: int = 50
    ) -> List[ChatMessage]:
        """Récupère l'historique de conversation."""
        ...


class FeedbackPort(ABC):
    """Port d'entrée : Système de feedback parent."""

    @abstractmethod
    async def submit_feedback(
        self, message_id: str, parent_id: str, is_positive: bool, comment: str = ""
    ) -> Feedback:
        """Soumet un feedback (pouce haut/bas)."""
        ...

    @abstractmethod
    async def get_feedback_stats(self) -> dict:
        """Statistiques globales de feedback."""
        ...


class TrainingPort(ABC):
    """Port d'entrée : Pipeline d'entraînement."""

    @abstractmethod
    async def trigger_finetuning(self) -> TrainingMetrics:
        """Lance le finetuning basé sur les feedbacks accumulés."""
        ...

    @abstractmethod
    async def get_model_metrics(self) -> TrainingMetrics:
        """Métriques du modèle actuel."""
        ...


# ═══════════════════════════════════════════════════════════════
#  PORTS DE SORTIE (Driven Ports / Infrastructure)
# ═══════════════════════════════════════════════════════════════

class AudioProcessorPort(ABC):
    """Port de sortie : Traitement audio."""

    @abstractmethod
    def extract_mfcc(self, audio_data: bytes) -> MFCCFeatures:
        """Extrait les caractéristiques MFCC d'un audio."""
        ...

    @abstractmethod
    def preprocess(self, audio_data: bytes) -> bytes:
        """Prétraitement audio (normalisation, rééchantillonnage)."""
        ...


class CryClassifierPort(ABC):
    """Port de sortie : Classificateur de pleurs."""

    @abstractmethod
    def predict(self, features: MFCCFeatures) -> CryPrediction:
        """Prédit le type de pleur à partir des features MFCC."""
        ...

    @abstractmethod
    def train(self, features_list: list, labels: list) -> TrainingMetrics:
        """Entraîne le classificateur."""
        ...


class LLMPort(ABC):
    """Port de sortie : Modèle de langage (LLM)."""

    @abstractmethod
    async def generate_response(
        self, prompt: str, context: str = "", history: List[dict] = None
    ) -> str:
        """Génère une réponse à partir du prompt."""
        ...

    @abstractmethod
    async def generate_advice(self, cry_type: str) -> str:
        """Génère un conseil basé sur le type de pleur détecté."""
        ...


class KnowledgeBasePort(ABC):
    """Port de sortie : Base de connaissances vectorielle (RAG)."""

    @abstractmethod
    def add_document(self, document: KnowledgeDocument) -> str:
        """Ajoute un document à la base."""
        ...

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[KnowledgeDocument]:
        """Recherche sémantique dans la base."""
        ...


class DatabasePort(ABC):
    """Port de sortie : Persistance des données."""

    @abstractmethod
    async def save_prediction(self, prediction: CryPrediction) -> str:
        ...

    @abstractmethod
    async def save_message(self, message: ChatMessage) -> str:
        ...

    @abstractmethod
    async def save_feedback(self, feedback: Feedback) -> str:
        ...

    @abstractmethod
    async def get_feedbacks_for_training(self) -> List[Feedback]:
        """Récupère les feedbacks non encore utilisés pour le training."""
        ...

    @abstractmethod
    async def save_parent(self, parent: Parent) -> str:
        ...

    @abstractmethod
    async def get_parent(self, parent_id: str) -> Optional[Parent]:
        ...


class NotificationPort(ABC):
    """Port de sortie : Notifications aux parents."""

    @abstractmethod
    async def send_notification(
        self, parent_id: str, title: str, message: str
    ) -> bool:
        ...
