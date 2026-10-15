"""
BabyCry AI - Adaptateur LLM
Implémentation du port LLMPort.
Supporte : Mistral (HuggingFace), OpenAI API, ou local.
"""
import os
from typing import List, Optional
from loguru import logger

from app.ports import LLMPort
from config import LLM_MODEL_NAME, CRY_LABELS_FR


class HuggingFaceLLMAdapter(LLMPort):
    """
    Adaptateur LLM utilisant HuggingFace Transformers.
    Modèle par défaut : Mistral-7B-Instruct-v0.2
    Peut être finetuné avec LoRA via le pipeline de training.
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _lazy_load(self):
        """Chargement paresseux du modèle (lourd en mémoire)."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"⏳ Chargement du LLM : {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )

        # Charger l'adaptateur LoRA si disponible
        lora_path = os.path.join("models", "lora_adapter")
        if os.path.exists(lora_path):
            from peft import PeftModel
            logger.info(f"🔧 Chargement de l'adaptateur LoRA : {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self._loaded = True
        logger.info("✅ LLM chargé avec succès")

    async def generate_response(
        self, prompt: str, context: str = "", history: List[dict] = None
    ) -> str:
        """Génère une réponse conversationnelle."""
        self._lazy_load()

        # Construire le prompt Mistral format
        messages = []

        # Prompt système
        if context:
            messages.append({"role": "system", "content": context})

        # Historique de conversation
        if history:
            for msg in history[-6:]:  # Garder les 6 derniers messages
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

        # Question actuelle
        messages.append({"role": "user", "content": prompt})

        # Tokeniser et générer
        input_text = self._format_messages(messages)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    async def generate_advice(self, cry_type: str) -> str:
        """Génère un conseil personnalisé basé sur le type de pleur."""
        # Conseil de base rapide (sans LLM pour la latence)
        base_advice = CRY_LABELS_FR.get(cry_type, "")

        # Enrichir via LLM si chargé
        if self._loaded:
            prompt = (
                f"En tant que spécialiste en pédiatrie, donne un conseil court "
                f"et rassurant à un parent dont le bébé pleure de {cry_type}. "
                f"Limite-toi à 2-3 phrases pratiques."
            )
            try:
                advice = await self.generate_response(prompt)
                return advice
            except Exception:
                pass

        return base_advice

    def _format_messages(self, messages: list) -> str:
        """Formate les messages au format Mistral Instruct."""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"<s>[INST] {content}\n"
            elif role == "user":
                if formatted and not formatted.endswith("\n"):
                    formatted += "\n"
                formatted += f"[INST] {content} [/INST]"
            elif role == "assistant":
                formatted += f" {content}</s>"
        return formatted


class OpenAILLMAdapter(LLMPort):
    """
    Adaptateur alternatif utilisant l'API OpenAI (ou compatible).
    Utile pour le développement ou quand les ressources GPU sont limitées.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def generate_response(
        self, prompt: str, context: str = "", history: List[dict] = None
    ) -> str:
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        if history:
            messages.extend(history[-6:])
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        return response.choices[0].message.content

    async def generate_advice(self, cry_type: str) -> str:
        prompt = (
            f"Conseil court et rassurant pour un parent dont le bébé pleure "
            f"de {cry_type}. 2-3 phrases pratiques en français."
        )
        return await self.generate_response(prompt)
