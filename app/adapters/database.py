"""
BabyCry AI - Adaptateur Base de Données (SQLAlchemy)
Implémentation du port DatabasePort.
"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    create_engine, Column, String, Float, DateTime,
    Integer, Text, JSON, Enum as SQLEnum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from loguru import logger

from app.ports import DatabasePort
from app.domain import (
    CryPrediction, ChatMessage, Feedback, Parent,
    CryType, FeedbackType
)
from config import DATABASE_URL

Base = declarative_base()


# ─── Modèles ORM ──────────────────────────────────────────

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True)
    audio_id = Column(String, nullable=True)
    cry_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON, nullable=True)
    advice = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class MessageRecord(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True)
    parent_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class FeedbackRecord(Base):
    __tablename__ = "feedbacks"
    id = Column(String, primary_key=True)
    message_id = Column(String, nullable=False, index=True)
    parent_id = Column(String, nullable=False, index=True)
    feedback_type = Column(String, nullable=False)
    comment = Column(Text, nullable=True)
    used_for_training = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)


class ParentRecord(Base):
    __tablename__ = "parents"
    id = Column(String, primary_key=True)
    username = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False)
    baby_name = Column(String, nullable=True)
    baby_age_months = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)


# ─── Adaptateur ────────────────────────────────────────────

class SQLiteDatabaseAdapter(DatabasePort):
    """Adaptateur SQLite via SQLAlchemy."""

    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("🗄️ Base de données initialisée")

    async def save_prediction(self, prediction: CryPrediction) -> str:
        with self.Session() as session:
            record = PredictionRecord(
                id=prediction.id,
                audio_id=prediction.audio_id,
                cry_type=prediction.cry_type.value,
                confidence=prediction.confidence,
                probabilities=prediction.probabilities,
                advice=prediction.advice,
            )
            session.add(record)
            session.commit()
        return prediction.id

    async def save_message(self, message: ChatMessage) -> str:
        with self.Session() as session:
            record = MessageRecord(
                id=message.id,
                parent_id=message.parent_id,
                role=message.role,
                content=message.content,
                context=message.context,
            )
            session.add(record)
            session.commit()
        return message.id

    async def save_feedback(self, feedback: Feedback) -> str:
        with self.Session() as session:
            record = FeedbackRecord(
                id=feedback.id,
                message_id=feedback.message_id,
                parent_id=feedback.parent_id,
                feedback_type=feedback.feedback_type.value,
                comment=feedback.comment,
            )
            session.add(record)
            session.commit()
        return feedback.id

    async def get_feedbacks_for_training(self) -> List[Feedback]:
        with self.Session() as session:
            records = (
                session.query(FeedbackRecord)
                .filter(FeedbackRecord.used_for_training == 0)
                .all()
            )
            return [
                Feedback(
                    id=r.id,
                    message_id=r.message_id,
                    parent_id=r.parent_id,
                    feedback_type=FeedbackType(r.feedback_type),
                    comment=r.comment,
                )
                for r in records
            ]

    async def mark_feedbacks_used(self, feedback_ids: List[str]):
        with self.Session() as session:
            session.query(FeedbackRecord).filter(
                FeedbackRecord.id.in_(feedback_ids)
            ).update({FeedbackRecord.used_for_training: 1})
            session.commit()

    async def save_parent(self, parent: Parent) -> str:
        with self.Session() as session:
            record = ParentRecord(
                id=parent.id,
                username=parent.username,
                email=parent.email,
                baby_name=parent.baby_name,
                baby_age_months=parent.baby_age_months,
            )
            session.merge(record)
            session.commit()
        return parent.id

    async def get_parent(self, parent_id: str) -> Optional[Parent]:
        with self.Session() as session:
            record = session.query(ParentRecord).filter_by(id=parent_id).first()
            if not record:
                return None
            return Parent(
                id=record.id,
                username=record.username,
                email=record.email,
                baby_name=record.baby_name,
                baby_age_months=record.baby_age_months,
            )

    async def get_chat_history(
        self, parent_id: str, limit: int = 50
    ) -> List[ChatMessage]:
        with self.Session() as session:
            records = (
                session.query(MessageRecord)
                .filter_by(parent_id=parent_id)
                .order_by(MessageRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                ChatMessage(
                    id=r.id,
                    parent_id=r.parent_id,
                    role=r.role,
                    content=r.content,
                    context=r.context,
                )
                for r in reversed(records)
            ]
