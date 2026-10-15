"""
BabyCry AI - Couche Domaine (Noyau Hexagonal)
Entités métier pures, sans dépendance externe.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


# ═══════════════════════════════════════════════════════════════
#  ENTITÉS MÉTIER
# ═══════════════════════════════════════════════════════════════

class CryType(str, Enum):
    FAIM = "faim"
    DOULEUR = "douleur"
    FATIGUE = "fatigue"
    PEUR = "peur"
    INCONFORT = "inconfort"


class FeedbackType(str, Enum):
    THUMBS_UP = "up"
    THUMBS_DOWN = "down"


@dataclass
class AudioSample:
    """Échantillon audio d'un pleur de bébé."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    duration: float = 0.0
    sample_rate: int = 22050
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CryPrediction:
    """Résultat de l'analyse d'un pleur."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_id: str = ""
    cry_type: CryType = CryType.FAIM
    confidence: float = 0.0
    probabilities: dict = field(default_factory=dict)
    advice: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ChatMessage:
    """Message dans la conversation parent-IA."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str = ""
    role: str = "user"  # "user" ou "assistant"
    content: str = ""
    context: Optional[str] = None  # contexte RAG utilisé
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Feedback:
    """Feedback parent sur une réponse IA (pouce haut/bas)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_id: str = ""
    parent_id: str = ""
    feedback_type: FeedbackType = FeedbackType.THUMBS_UP
    comment: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Parent:
    """Profil parent utilisateur."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    username: str = ""
    email: str = ""
    baby_name: Optional[str] = None
    baby_age_months: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeDocument:
    """Document de la base de connaissances parentalité."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    category: str = ""  # santé, éducation, sommeil, alimentation...
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════════
#  OBJETS VALEUR
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MFCCFeatures:
    """Caractéristiques MFCC extraites d'un audio."""
    coefficients: tuple = ()  # vecteur MFCC aplati
    n_mfcc: int = 40
    sample_rate: int = 22050


@dataclass(frozen=True)
class TrainingMetrics:
    """Métriques d'entraînement du modèle."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = 0.0
    epoch: int = 0
