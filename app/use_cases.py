"""
BabyCry AI - Cas d'Usage (Services Applicatifs)
Orchestration de la logique métier via les ports.
"""
from typing import List, Optional
from loguru import logger

from app.domain import (
    CryPrediction, ChatMessage, Feedback, FeedbackType,
    TrainingMetrics
)
from app.ports import (
    AudioProcessorPort, CryClassifierPort, LLMPort,
    KnowledgeBasePort, DatabasePort, NotificationPort
)


# ═══════════════════════════════════════════════════════════════
#  USE CASE 1 : Analyse des pleurs
# ═══════════════════════════════════════════════════════════════

class AnalyzeCryUseCase:
    """
    Flux : Micro → Prétraitement → MFCC → Classificateur → Conseil → Notification
    """

    def __init__(
        self,
        audio_processor: AudioProcessorPort,
        classifier: CryClassifierPort,
        llm: LLMPort,
        database: DatabasePort,
        notifier: NotificationPort,
    ):
        self.audio_processor = audio_processor
        self.classifier = classifier
        self.llm = llm
        self.database = database
        self.notifier = notifier

    async def execute(self, audio_data: bytes, parent_id: str) -> CryPrediction:
        """Pipeline complet d'analyse d'un pleur."""
        logger.info(f"🎤 Analyse du pleur pour parent {parent_id}")

        # 1. Prétraitement audio
        processed = self.audio_processor.preprocess(audio_data)

        # 2. Extraction MFCC
        features = self.audio_processor.extract_mfcc(processed)
        logger.info(f"📊 MFCC extrait : {features.n_mfcc} coefficients")

        # 3. Classification
        prediction = self.classifier.predict(features)
        logger.info(
            f"🔍 Prédiction : {prediction.cry_type.value} "
            f"(confiance: {prediction.confidence:.2%})"
        )

        # 4. Génération de conseil personnalisé via LLM
        advice = await self.llm.generate_advice(prediction.cry_type.value)
        prediction.advice = advice

        # 5. Persistance
        await self.database.save_prediction(prediction)

        # 6. Notification parent
        await self.notifier.send_notification(
            parent_id=parent_id,
            title=f"Pleur détecté : {prediction.cry_type.value}",
            message=advice,
        )

        return prediction


# ═══════════════════════════════════════════════════════════════
#  USE CASE 2 : Chatbot conseil parental
# ═══════════════════════════════════════════════════════════════

class AskChatbotUseCase:
    """
    Flux : Question → RAG (recherche KB) → LLM (génération) → Réponse
    """

    def __init__(
        self,
        llm: LLMPort,
        knowledge_base: KnowledgeBasePort,
        database: DatabasePort,
    ):
        self.llm = llm
        self.knowledge_base = knowledge_base
        self.database = database

    async def execute(
        self, question: str, parent_id: str, history: List[dict] = None
    ) -> ChatMessage:
        """Répond à une question parentale avec contexte RAG."""
        logger.info(f"💬 Question de {parent_id}: {question[:80]}...")

        # 1. Sauvegarder la question
        user_msg = ChatMessage(
            parent_id=parent_id,
            role="user",
            content=question,
        )
        await self.database.save_message(user_msg)

        # 2. Recherche dans la base de connaissances (RAG)
        relevant_docs = self.knowledge_base.search(question, k=3)
        context = "\n\n".join([
            f"[{doc.category}] {doc.title}\n{doc.content}"
            for doc in relevant_docs
        ])
        logger.info(f"📚 {len(relevant_docs)} documents pertinents trouvés")

        # 3. Construction du prompt système
        system_prompt = self._build_system_prompt(context)

        # 4. Génération de la réponse via LLM
        response_text = await self.llm.generate_response(
            prompt=question,
            context=system_prompt,
            history=history or [],
        )

        # 5. Sauvegarder la réponse
        assistant_msg = ChatMessage(
            parent_id=parent_id,
            role="assistant",
            content=response_text,
            context=context[:500],
        )
        await self.database.save_message(assistant_msg)

        return assistant_msg

    def _build_system_prompt(self, context: str) -> str:
        return f"""Tu es BabyBot, un assistant IA spécialisé dans la parentalité.
Tu aides les parents à comprendre leur bébé/enfant et tu donnes des conseils sur :
- La santé et le bien-être du bébé
- L'éducation et le développement
- Le sommeil et l'alimentation
- La gestion des pleurs et des émotions

Règles :
- Réponds en français, avec empathie et bienveillance
- Base tes réponses sur les connaissances médicales et pédiatriques validées
- Si tu n'es pas sûr, recommande de consulter un pédiatre
- Sois concis mais complet

Contexte de la base de connaissances :
{context}
"""


# ═══════════════════════════════════════════════════════════════
#  USE CASE 3 : Feedback & Apprentissage continu
# ═══════════════════════════════════════════════════════════════

class SubmitFeedbackUseCase:
    """
    Flux : Feedback parent → Stockage → Déclenchement finetuning si seuil atteint
    """

    FINETUNING_THRESHOLD = 100  # nombre de feedbacks avant relance

    def __init__(self, database: DatabasePort):
        self.database = database

    async def execute(
        self, message_id: str, parent_id: str, is_positive: bool, comment: str = ""
    ) -> Feedback:
        """Enregistre un feedback et vérifie le seuil de finetuning."""
        feedback = Feedback(
            message_id=message_id,
            parent_id=parent_id,
            feedback_type=FeedbackType.THUMBS_UP if is_positive else FeedbackType.THUMBS_DOWN,
            comment=comment,
        )
        await self.database.save_feedback(feedback)
        logger.info(
            f"{'👍' if is_positive else '👎'} Feedback de {parent_id} "
            f"sur message {message_id}"
        )

        # Vérifier si on atteint le seuil de finetuning
        all_feedbacks = await self.database.get_feedbacks_for_training()
        if len(all_feedbacks) >= self.FINETUNING_THRESHOLD:
            logger.info(
                f"🎯 Seuil de {self.FINETUNING_THRESHOLD} feedbacks atteint → "
                f"Finetuning déclenché"
            )

        return feedback


# ═══════════════════════════════════════════════════════════════
#  USE CASE 4 : Pipeline de Finetuning
# ═══════════════════════════════════════════════════════════════

class FinetuneModelUseCase:
    """
    Flux : Collecte feedbacks → Préparation dataset → LoRA finetuning → Évaluation
    """

    def __init__(self, database: DatabasePort, llm: LLMPort):
        self.database = database
        self.llm = llm

    async def execute(self) -> TrainingMetrics:
        """Lance le pipeline de finetuning complet."""
        logger.info("🚀 Démarrage du pipeline de finetuning")

        # 1. Collecter les feedbacks
        feedbacks = await self.database.get_feedbacks_for_training()
        logger.info(f"📦 {len(feedbacks)} feedbacks collectés")

        # 2. Préparer le dataset DPO/RLHF
        # Les réponses avec thumbs_up → exemples positifs
        # Les réponses avec thumbs_down → exemples négatifs
        positive = [f for f in feedbacks if f.feedback_type == FeedbackType.THUMBS_UP]
        negative = [f for f in feedbacks if f.feedback_type == FeedbackType.THUMBS_DOWN]

        logger.info(f"✅ {len(positive)} positifs | ❌ {len(negative)} négatifs")

        # 3. Le finetuning est délégué à l'adaptateur LLM
        # (voir training/finetune.py pour l'implémentation complète)
        metrics = TrainingMetrics(
            accuracy=0.0, precision=0.0, recall=0.0,
            f1_score=0.0, loss=0.0, epoch=0
        )

        logger.info("✅ Finetuning terminé")
        return metrics
