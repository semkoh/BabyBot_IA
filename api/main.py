"""
BabyCry AI - API FastAPI
Point d'entrée de l'application.
Routes REST + WebSocket pour le chatbot et l'analyse audio.

Modes :
  - demo  : fonctionne sans GPU, LLM simulé (par défaut)
  - full  : utilise Mistral-7B + tous les adaptateurs
"""
import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger
import uvicorn

from config import API_HOST, API_PORT
from app.adapters.demo_analyzer import DemoCryAnalyzer

# Déterminer le mode d'exécution
RUN_MODE = os.getenv("BABYCRY_MODE", "demo")
logger.info(f"🚀 Mode de lancement : {RUN_MODE}")

# ─── Modèles de requête/réponse ────────────────────────────

class ChatRequest(BaseModel):
    question: str
    parent_id: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    message_id: str
    response: str
    sources: List[str] = []

class FeedbackRequest(BaseModel):
    message_id: str
    parent_id: str
    is_positive: bool
    comment: Optional[str] = ""

class CryAnalysisResponse(BaseModel):
    prediction_id: str
    cry_type: str
    confidence: float
    probabilities: dict
    advice: str

class ParentCreate(BaseModel):
    username: str
    email: str
    baby_name: Optional[str] = None
    baby_age_months: Optional[int] = None


# ─── Application FastAPI ───────────────────────────────────

app = FastAPI(
    title="BabyCry AI",
    description="IA d'analyse des pleurs de bébé & assistant parental",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Injection de dépendances (Architecture Hexagonale) ────

def get_dependencies():
    """
    Assemble les adaptateurs et use cases.
    Mode demo : utilise DemoLLM (sans GPU)
    Mode full : utilise HuggingFace Mistral-7B
    """
    from app.adapters.database import SQLiteDatabaseAdapter
    from app.adapters.notification import WebSocketNotifier
    from app.use_cases import (
        AnalyzeCryUseCase, AskChatbotUseCase,
        SubmitFeedbackUseCase, FinetuneModelUseCase,
    )

    # Adaptateurs communs
    database = SQLiteDatabaseAdapter()
    notifier = WebSocketNotifier()

    if RUN_MODE == "full":
        # Mode complet avec GPU
        from app.adapters.audio_processor import LibrosaAudioProcessor
        from app.adapters.cry_classifier import CryClassifierAdapter
        from app.adapters.llm_adapter import HuggingFaceLLMAdapter
        from app.adapters.knowledge_base import ChromaKnowledgeBase

        audio_processor = LibrosaAudioProcessor()
        classifier = CryClassifierAdapter()
        llm = HuggingFaceLLMAdapter()
        knowledge_base = ChromaKnowledgeBase()
        knowledge_base.seed_knowledge_base()
        logger.info("✅ Mode COMPLET activé (Mistral-7B + tous adaptateurs)")
    else:
        # Mode démo sans GPU
        from app.adapters.demo_llm import DemoLLMAdapter

        audio_processor = None
        classifier = None
        llm = DemoLLMAdapter()
        knowledge_base = None
        logger.info("✅ Mode DEMO activé (LLM simulé, pas de GPU requis)")

    # Use Cases
    analyze_cry = AnalyzeCryUseCase(
        audio_processor=audio_processor,
        classifier=classifier,
        llm=llm,
        database=database,
        notifier=notifier,
    ) if RUN_MODE == "full" else None

    ask_chatbot = AskChatbotUseCase(
        llm=llm,
        knowledge_base=knowledge_base,
        database=database,
    ) if RUN_MODE == "full" else None

    submit_feedback = SubmitFeedbackUseCase(database=database)
    finetune = FinetuneModelUseCase(database=database, llm=llm)

    return {
        "analyze_cry": analyze_cry,
        "ask_chatbot": ask_chatbot,
        "submit_feedback": submit_feedback,
        "finetune": finetune,
        "notifier": notifier,
        "database": database,
        "llm": llm,
    }


# Initialisation paresseuse
_deps = None
_demo_analyzer = DemoCryAnalyzer()

def deps():
    global _deps
    if _deps is None:
        _deps = get_dependencies()
    return _deps


# ═══════════════════════════════════════════════════════════
#  ROUTES API
# ═══════════════════════════════════════════════════════════

# ─── Analyse des pleurs ────────────────────────────────────

@app.post("/api/analyze", response_model=CryAnalysisResponse)
async def analyze_cry(
    audio: UploadFile = File(...),
    parent_id: str = "default",
):
    """
    🎤 Analyse un enregistrement audio de pleur de bébé.
    
    - Accepte : WAV, MP3, OGG
    - Retourne : type de pleur, confiance, conseil
    """
    try:
        audio_data = await audio.read()
        if RUN_MODE == "demo":
            result = _demo_analyzer.analyze(audio_data)
            result.advice = await deps()["llm"].generate_advice(result.cry_type.value)
            await deps()["database"].save_prediction(result)
        else:
            result = await deps()["analyze_cry"].execute(audio_data, parent_id)
        return CryAnalysisResponse(
            prediction_id=result.id,
            cry_type=result.cry_type.value,
            confidence=result.confidence,
            probabilities=result.probabilities,
            advice=result.advice,
        )
    except Exception as e:
        logger.error(f"Erreur analyse : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Chatbot parental ─────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    💬 Pose une question au chatbot parental.
    
    Mode demo : utilise DemoLLM (matching mots-clés)
    Mode full : utilise RAG + Mistral-7B
    """
    try:
        if RUN_MODE == "demo":
            # Mode démo : appel direct au DemoLLM
            llm = deps()["llm"]
            response_text = await llm.generate_response(request.question)
            import uuid
            msg_id = str(uuid.uuid4())

            # Sauvegarder en base
            from app.domain import ChatMessage
            user_msg = ChatMessage(parent_id=request.parent_id, role="user", content=request.question)
            await deps()["database"].save_message(user_msg)
            bot_msg = ChatMessage(id=msg_id, parent_id=request.parent_id, role="assistant", content=response_text)
            await deps()["database"].save_message(bot_msg)

            return ChatResponse(message_id=msg_id, response=response_text)
        else:
            # Mode complet avec RAG
            history = await deps()["database"].get_chat_history(request.parent_id, limit=10)
            history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
            result = await deps()["ask_chatbot"].execute(
                question=request.question,
                parent_id=request.parent_id,
                history=history_dicts,
            )
            return ChatResponse(message_id=result.id, response=result.content)
    except Exception as e:
        logger.error(f"Erreur chat : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Feedback ──────────────────────────────────────────────

@app.post("/api/feedback")
async def feedback(request: FeedbackRequest):
    """
    👍👎 Soumet un feedback sur une réponse du chatbot.
    
    Les feedbacks sont utilisés pour le finetuning continu du modèle.
    Seuil : 100 feedbacks → déclenchement automatique du finetuning.
    """
    try:
        result = await deps()["submit_feedback"].execute(
            message_id=request.message_id,
            parent_id=request.parent_id,
            is_positive=request.is_positive,
            comment=request.comment or "",
        )
        return {"feedback_id": result.id, "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Finetuning ───────────────────────────────────────────

@app.post("/api/finetune")
async def trigger_finetuning():
    """
    🚀 Déclenche manuellement le pipeline de finetuning.
    
    Collecte les feedbacks, prépare le dataset DPO,
    lance le finetuning LoRA sur le LLM.
    """
    try:
        metrics = await deps()["finetune"].execute()
        return {"status": "completed", "metrics": vars(metrics)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── WebSocket (notifications temps réel) ──────────────────

@app.websocket("/ws/{parent_id}")
async def websocket_endpoint(websocket: WebSocket, parent_id: str):
    """
    🔌 Connexion WebSocket pour les notifications en temps réel.
    
    Reçoit les alertes quand un pleur est détecté par le micro.
    """
    notifier = deps()["notifier"]
    await notifier.connect(parent_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Le client peut envoyer des commandes via WebSocket
            logger.info(f"Message WS de {parent_id}: {data}")
    except Exception:
        await notifier.disconnect(parent_id, websocket)


# ─── Santé ─────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "BabyCry AI", "mode": RUN_MODE}


@app.get("/api/mode")
async def get_mode():
    return {"mode": RUN_MODE}


# ─── Frontend statique ─────────────────────────────────────

STATIC_DIR = Path(__file__).parent.parent / "static"

@app.get("/app")
@app.get("/")
async def serve_frontend():
    """Sert l'interface web BabyBot."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "BabyCry AI API", "docs": "/docs", "mode": RUN_MODE}

# Monter les fichiers statiques (CSS, JS, images)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Point d'entrée ───────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
