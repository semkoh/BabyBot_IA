"""
BabyCry AI - Configuration
Architecture hexagonale : configuration centralisée
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Chemins ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"

for d in [DATA_DIR, MODEL_DIR, UPLOAD_DIR]:
    d.mkdir(exist_ok=True)

# ─── Audio ──────────────────────────────────────────────────
SAMPLE_RATE = 22050
AUDIO_DURATION = 5  # secondes
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

# ─── Classes de pleurs ─────────────────────────────────────
CRY_CLASSES = {
    0: "faim",
    1: "douleur",
    2: "fatigue",
    3: "peur",
    4: "inconfort",
}

CRY_LABELS_FR = {
    "faim": "Votre bébé a probablement faim. Essayez de le nourrir.",
    "douleur": "Votre bébé semble avoir mal. Vérifiez s'il y a une source d'inconfort.",
    "fatigue": "Votre bébé est fatigué. Essayez de le bercer doucement.",
    "peur": "Votre bébé a peur. Prenez-le dans vos bras pour le rassurer.",
    "inconfort": "Votre bébé est inconfortable. Vérifiez la couche ou la température.",
}

# ─── LLM ────────────────────────────────────────────────────
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = str(DATA_DIR / "chroma_db")

# ─── Finetuning ─────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
BATCH_SIZE = 4
MAX_SEQ_LENGTH = 512

# ─── Base de données ────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/babycry.db")

# ─── API ────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
SECRET_KEY = os.getenv("SECRET_KEY", "babycry-secret-key-change-me")
