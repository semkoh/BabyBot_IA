"""
BabyCry AI - Adaptateur Notification (WebSocket + Push)
Implémentation du port NotificationPort.
"""
import json
from typing import Dict, Set
from loguru import logger
from fastapi import WebSocket
from app.ports import NotificationPort


class WebSocketNotifier(NotificationPort):
    """
    Notifications en temps réel via WebSocket.
    Chaque parent connecté reçoit les alertes de pleurs.
    """

    def __init__(self):
        # parent_id → set de connexions WebSocket actives
        self.connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, parent_id: str, websocket: WebSocket):
        """Enregistre une connexion WebSocket pour un parent."""
        await websocket.accept()
        if parent_id not in self.connections:
            self.connections[parent_id] = set()
        self.connections[parent_id].add(websocket)
        logger.info(f"🔌 Parent {parent_id} connecté (WebSocket)")

    async def disconnect(self, parent_id: str, websocket: WebSocket):
        """Déconnecte un WebSocket."""
        if parent_id in self.connections:
            self.connections[parent_id].discard(websocket)
            if not self.connections[parent_id]:
                del self.connections[parent_id]

    async def send_notification(
        self, parent_id: str, title: str, message: str
    ) -> bool:
        """Envoie une notification push au parent via WebSocket."""
        if parent_id not in self.connections:
            logger.warning(f"⚠️ Parent {parent_id} non connecté")
            return False

        payload = json.dumps({
            "type": "notification",
            "title": title,
            "message": message,
        })

        disconnected = set()
        for ws in self.connections[parent_id]:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)

        # Nettoyer les connexions mortes
        self.connections[parent_id] -= disconnected
        return True
