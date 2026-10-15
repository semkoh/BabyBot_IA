"""
BabyCry AI - Pipeline de Finetuning
LoRA (Low-Rank Adaptation) + DPO (Direct Preference Optimization)

Ce module implémente le pipeline d'apprentissage continu :
1. Collecte des feedbacks parents (👍/👎)
2. Construction du dataset de préférences
3. Finetuning LoRA du LLM
4. Évaluation et déploiement
"""
import os
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from loguru import logger

from config import (
    LLM_MODEL_NAME, MODEL_DIR,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    LEARNING_RATE, NUM_EPOCHS, BATCH_SIZE, MAX_SEQ_LENGTH,
)


@dataclass
class PreferenceExample:
    """Exemple de préférence pour DPO."""
    prompt: str
    chosen: str       # réponse validée (👍)
    rejected: str     # réponse rejetée (👎) ou version alternative


class FinetuningPipeline:
    """
    Pipeline complet de finetuning LoRA.

    Architecture :
    ┌──────────────────────────────────────────────────────┐
    │                PIPELINE D'APPRENTISSAGE               │
    │                                                       │
    │  Feedbacks   →  Dataset DPO  →  LoRA Training  →  Eval│
    │  (👍/👎)       (paires pref)    (Mistral+LoRA)    (F1) │
    │                                                       │
    │  Seuil atteint → Trigger auto → Merge adapter → Deploy│
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.model_name = model_name
        self.output_dir = str(MODEL_DIR / "lora_adapter")
        self.dataset_path = str(MODEL_DIR / "training_data.json")

    def prepare_preference_dataset(
        self, feedbacks_with_messages: List[Dict]
    ) -> List[PreferenceExample]:
        """
        Transforme les feedbacks en dataset de préférences DPO.

        Structure attendue de feedbacks_with_messages :
        [
            {
                "question": "...",
                "response": "...",
                "feedback": "up" | "down",
                "comment": "..."  (optionnel)
            }
        ]
        """
        # Grouper par question
        question_groups: Dict[str, Dict] = {}
        for item in feedbacks_with_messages:
            q = item["question"]
            if q not in question_groups:
                question_groups[q] = {"positive": [], "negative": []}
            if item["feedback"] == "up":
                question_groups[q]["positive"].append(item["response"])
            else:
                question_groups[q]["negative"].append(item["response"])

        # Créer les paires de préférences
        examples = []
        for question, responses in question_groups.items():
            for chosen in responses["positive"]:
                for rejected in responses["negative"]:
                    examples.append(PreferenceExample(
                        prompt=question,
                        chosen=chosen,
                        rejected=rejected,
                    ))

        # Sauvegarder le dataset
        dataset_json = [
            {"prompt": e.prompt, "chosen": e.chosen, "rejected": e.rejected}
            for e in examples
        ]
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=2)

        logger.info(f"📊 Dataset DPO préparé : {len(examples)} paires")
        return examples

    def run_lora_finetuning(self, examples: List[PreferenceExample]) -> dict:
        """
        Lance le finetuning LoRA sur le LLM.
        Utilise PEFT (Parameter-Efficient Fine-Tuning).

        Config LoRA :
        - r = 16 (rang de la décomposition)
        - alpha = 32 (facteur de mise à l'échelle)
        - dropout = 0.05
        - Modules cibles : q_proj, k_proj, v_proj, o_proj
        """
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            TrainingArguments, Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset

        logger.info("🚀 Démarrage du finetuning LoRA")

        # 1. Charger le modèle de base
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # 2. Configurer LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # 3. Préparer le dataset
        def format_example(example: PreferenceExample) -> str:
            return (
                f"[INST] {example.prompt} [/INST] {example.chosen}</s>"
            )

        texts = [format_example(e) for e in examples]
        encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        )
        dataset = Dataset.from_dict({
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"].clone(),
        })

        # 4. Configuration d'entraînement
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            gradient_accumulation_steps=4,
            report_to="none",
        )

        # 5. Entraîner
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        result = trainer.train()

        # 6. Sauvegarder l'adaptateur LoRA
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        logger.info(f"✅ LoRA finetuning terminé → {self.output_dir}")
        return {
            "train_loss": result.training_loss,
            "epochs": NUM_EPOCHS,
            "lora_r": LORA_R,
            "output_dir": self.output_dir,
        }


class CryClassifierTrainer:
    """
    Pipeline d'entraînement du classificateur de pleurs.
    Dataset : Donate-a-Cry Corpus.

    Architecture :
    ┌──────────────────────────────────────────────────┐
    │           PIPELINE CLASSIFICATEUR                 │
    │                                                   │
    │  Audio WAV → Preprocess → MFCC → Augmentation    │
    │          → Train (SVM+RF) → Évaluation → Save    │
    └──────────────────────────────────────────────────┘
    """

    def __init__(self, data_dir: str = "data/donate_a_cry"):
        self.data_dir = data_dir

    def load_dataset(self) -> Tuple[List, List]:
        """
        Charge le dataset Donate-a-Cry.
        Structure attendue :
        data/donate_a_cry/
        ├── faim/
        │   ├── cry_001.wav
        │   └── ...
        ├── douleur/
        ├── fatigue/
        ├── peur/
        └── inconfort/
        """
        from app.adapters.audio_processor import AudioFeatureExtractor

        extractor = AudioFeatureExtractor()
        features, labels = [], []
        label_map = {"faim": 0, "douleur": 1, "fatigue": 2, "peur": 3, "inconfort": 4}

        for label_name, label_idx in label_map.items():
            label_dir = os.path.join(self.data_dir, label_name)
            if not os.path.exists(label_dir):
                logger.warning(f"⚠️ Répertoire manquant : {label_dir}")
                continue

            audio_files = [
                os.path.join(label_dir, f)
                for f in os.listdir(label_dir)
                if f.endswith(('.wav', '.mp3', '.ogg'))
            ]

            for fpath in audio_files:
                try:
                    feat = extractor.extract_features_from_file(fpath)
                    features.append(feat)
                    labels.append(label_idx)
                except Exception as e:
                    logger.error(f"Erreur {fpath}: {e}")

        logger.info(f"📦 Dataset chargé : {len(features)} échantillons")
        return features, labels

    def augment_data(self, features, labels) -> Tuple[List, List]:
        """
        Augmentation de données audio :
        - Ajout de bruit gaussien
        - Décalage temporel
        - Modification de pitch (simulé via features)
        """
        import numpy as np

        aug_features, aug_labels = list(features), list(labels)

        for feat, label in zip(features, labels):
            feat = np.array(feat)

            # Bruit gaussien
            noise = np.random.normal(0, 0.01, feat.shape)
            aug_features.append(feat + noise)
            aug_labels.append(label)

            # Scaling aléatoire
            scale = np.random.uniform(0.95, 1.05)
            aug_features.append(feat * scale)
            aug_labels.append(label)

        logger.info(
            f"🔀 Augmentation : {len(features)} → {len(aug_features)} échantillons"
        )
        return aug_features, aug_labels

    def train(self):
        """Lance le pipeline complet d'entraînement du classificateur."""
        from app.adapters.cry_classifier import CryClassifierAdapter

        # 1. Charger les données
        features, labels = self.load_dataset()
        if not features:
            logger.error("❌ Aucune donnée trouvée")
            return None

        # 2. Augmenter
        features, labels = self.augment_data(features, labels)

        # 3. Entraîner
        classifier = CryClassifierAdapter()
        metrics = classifier.train(features, labels)

        logger.info(
            f"📈 Résultats : Accuracy={metrics.accuracy:.4f} "
            f"F1={metrics.f1_score:.4f}"
        )
        return metrics
