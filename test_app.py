#!/usr/bin/env python3
"""
BabyCry AI - Script de test rapide
Verifie que tous les composants fonctionnent.

Usage :
    python test_app.py          (test hors-ligne, pas besoin du serveur)
    python test_app.py --api    (test avec le serveur lance)
"""
import asyncio
import sys
from pathlib import Path

# Ajouter le repertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
NC = "\033[0m"

SUPPORTS_UNICODE = (sys.stdout.encoding or "").lower().startswith("utf")
OK_SYMBOL = "✓" if SUPPORTS_UNICODE else "[OK]"
FAIL_SYMBOL = "✗" if SUPPORTS_UNICODE else "[FAIL]"
INFO_SYMBOL = "→" if SUPPORTS_UNICODE else "->"
BOX_TOP = "╔═══════════════════════════════════════╗" if SUPPORTS_UNICODE else "======================================="
BOX_MID = "║   👶 BabyCry AI - Tests automatiques  ║" if SUPPORTS_UNICODE else "BabyCry AI - Tests automatiques"
BOX_BOTTOM = "╚═══════════════════════════════════════╝" if SUPPORTS_UNICODE else "======================================="
SEP = "═══" if SUPPORTS_UNICODE else "---"


def ok(msg):
    print(f"  {GREEN}{OK_SYMBOL}{NC} {msg}")


def fail(msg):
    print(f"  {RED}{FAIL_SYMBOL}{NC} {msg}")


def info(msg):
    print(f"  {BLUE}{INFO_SYMBOL}{NC} {msg}")


def title(text):
    print(f"\n{YELLOW}{SEP} {text} {SEP}{NC}")


async def test_offline():
    """Tests sans serveur lance."""

    title("Test 1 : Domaine Metier")
    try:
        from app.domain import (
            AudioSample,
            ChatMessage,
            CryPrediction,
            CryType,
            Feedback,
            FeedbackType,
            MFCCFeatures,
            Parent,
            TrainingMetrics,
        )

        sample = AudioSample(file_path="test.wav", duration=5.0)
        prediction = CryPrediction(cry_type=CryType.FAIM, confidence=0.95)
        message = ChatMessage(parent_id="test", role="user", content="Bonjour")
        feedback = Feedback(
            message_id="123",
            parent_id="test",
            feedback_type=FeedbackType.THUMBS_UP,
        )
        _ = (prediction, message, feedback, Parent, MFCCFeatures, TrainingMetrics)
        ok(f"Toutes les entites creees (AudioSample.id={sample.id[:8]}...)")
    except Exception as e:
        fail(f"Domaine : {e}")
        return

    title("Test 2 : Base de Donnees SQLite")
    try:
        import os

        from app.adapters.database import SQLiteDatabaseAdapter

        db = SQLiteDatabaseAdapter("sqlite:///test_babycry.db")

        msg = ChatMessage(parent_id="test-parent", role="user", content="Mon bebe pleure beaucoup")
        msg_id = await db.save_message(msg)
        ok(f"Message sauvegarde (id={msg_id[:8]}...)")

        pred = CryPrediction(
            cry_type=CryType.FATIGUE,
            confidence=0.88,
            probabilities={"fatigue": 0.88},
        )
        pred_id = await db.save_prediction(pred)
        ok(f"Prediction sauvegardee (id={pred_id[:8]}...)")

        fb = Feedback(
            message_id=msg_id,
            parent_id="test-parent",
            feedback_type=FeedbackType.THUMBS_UP,
        )
        fb_id = await db.save_feedback(fb)
        ok(f"Feedback sauvegarde (id={fb_id[:8]}...)")

        history = await db.get_chat_history("test-parent")
        ok(f"Historique recupere ({len(history)} messages)")

        feedbacks = await db.get_feedbacks_for_training()
        ok(f"Feedbacks pour training : {len(feedbacks)} disponibles")

        db.engine.dispose()
        if Path("test_babycry.db").exists():
            os.remove("test_babycry.db")
            ok("Base de test nettoyee")
    except Exception as e:
        fail(f"Base de donnees : {e}")

    title("Test 3 : DemoLLM")
    try:
        from app.adapters.demo_llm import DemoLLMAdapter

        llm = DemoLLMAdapter()

        response = await llm.generate_response("Mon bebe pleure beaucoup la nuit")
        assert "pleur" in response.lower() or "faim" in response.lower() or "douleur" in response.lower()
        ok(f"Reponse pleurs OK ({len(response)} chars)")

        response = await llm.generate_response("Comment faire dormir mon bebe ?")
        assert "sommeil" in response.lower() or "nuit" in response.lower() or "dormir" in response.lower()
        ok(f"Reponse sommeil OK ({len(response)} chars)")

        response = await llm.generate_response("Quand commencer la diversification alimentaire ?")
        assert "aliment" in response.lower() or "diversi" in response.lower()
        ok(f"Reponse alimentation OK ({len(response)} chars)")

        advice = await llm.generate_advice("faim")
        assert "faim" in advice.lower() or "tetee" in advice.lower() or "biberon" in advice.lower()
        ok("Conseil 'faim' OK")

        advice = await llm.generate_advice("douleur")
        assert "douleur" in advice.lower() or "mal" in advice.lower()
        ok("Conseil 'douleur' OK")

        response = await llm.generate_response("Raconte-moi une blague")
        assert "babybot" in response.lower() or "assistant" in response.lower()
        ok("Reponse par defaut OK")
    except Exception as e:
        fail(f"DemoLLM : {e}")

    title("Test 4 : Configuration")
    try:
        from config import CRY_CLASSES, LLM_MODEL_NAME, LORA_R, N_MFCC, SAMPLE_RATE

        ok(f"5 classes de pleurs : {list(CRY_CLASSES.values())}")
        ok(f"Sample rate : {SAMPLE_RATE} Hz")
        ok(f"MFCC : {N_MFCC} coefficients")
        ok(f"LLM : {LLM_MODEL_NAME}")
        ok(f"LoRA rank : {LORA_R}")
    except Exception as e:
        fail(f"Config : {e}")

    title("Test 5 : Ports & Use Cases")
    try:
        from app.ports import (
            AudioProcessorPort,
            ChatbotPort,
            CryAnalysisPort,
            CryClassifierPort,
            DatabasePort,
            FeedbackPort,
            KnowledgeBasePort,
            LLMPort,
            NotificationPort,
            TrainingPort,
        )
        from app.use_cases import (
            AnalyzeCryUseCase,
            AskChatbotUseCase,
            FinetuneModelUseCase,
            SubmitFeedbackUseCase,
        )

        _ = (
            AudioProcessorPort,
            ChatbotPort,
            CryAnalysisPort,
            CryClassifierPort,
            DatabasePort,
            FeedbackPort,
            KnowledgeBasePort,
            LLMPort,
            NotificationPort,
            TrainingPort,
            AnalyzeCryUseCase,
            AskChatbotUseCase,
            FinetuneModelUseCase,
            SubmitFeedbackUseCase,
        )
        ok("10 ports definis (5 driving, 5 driven)")
        ok("4 use cases definis")
    except Exception as e:
        fail(f"Ports/Use Cases : {e}")

    title("Test 6 : Structure du projet")
    files = [
        "config.py",
        "requirements.txt",
        "app/domain.py",
        "app/ports.py",
        "app/use_cases.py",
        "app/adapters/audio_processor.py",
        "app/adapters/cry_classifier.py",
        "app/adapters/llm_adapter.py",
        "app/adapters/knowledge_base.py",
        "app/adapters/database.py",
        "app/adapters/notification.py",
        "app/adapters/demo_llm.py",
        "training/finetune.py",
        "api/main.py",
        "static/index.html",
    ]
    for file_path in files:
        if Path(file_path).exists():
            ok(file_path)
        else:
            fail(f"{file_path} MANQUANT")


async def test_api():
    """Tests avec le serveur lance."""
    import httpx

    base_url = "http://localhost:8000"

    title("Test API : Sante")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/health")
            data = response.json()
            ok(f"Health OK : {data}")
    except Exception:
        fail(f"Le serveur ne repond pas sur {base_url}")
        info("Lancez d'abord : .\\venv\\Scripts\\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000")
        return

    title("Test API : Chat")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                json={"question": "Mon bebe pleure beaucoup la nuit", "parent_id": "test-user"},
            )
            data = response.json()
            ok(f"Chat OK : {data['response'][:80]}...")

            response = await client.post(
                f"{base_url}/api/feedback",
                json={
                    "message_id": data["message_id"],
                    "parent_id": "test-user",
                    "is_positive": True,
                    "comment": "Tres utile !",
                },
            )
            ok(f"Feedback OK : {response.json()}")
    except Exception as e:
        fail(f"Chat API : {e}")

    title("Test API : Frontend")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/app")
            assert response.status_code == 200
            assert "BabyBot" in response.text
            ok(f"Frontend servi OK ({len(response.text)} chars)")
    except Exception as e:
        fail(f"Frontend : {e}")


async def main():
    print(f"\n{BLUE}{BOX_TOP}{NC}")
    print(f"{BLUE}{BOX_MID}{NC}")
    print(f"{BLUE}{BOX_BOTTOM}{NC}")

    await test_offline()

    if "--api" in sys.argv:
        await test_api()
    else:
        print(f"\n{BLUE}{INFO_SYMBOL} Pour tester l'API aussi : python test_app.py --api{NC}")

    print(f"\n{GREEN}{SEP} Tests termines {SEP}{NC}\n")


if __name__ == "__main__":
    asyncio.run(main())
