# 👶 BabyBot IA — Projet Complet

## Démarrage Rapide

### Interface seule (sans serveur)
Le fichier `BabyBot_Final.jsx` fonctionne directement dans **Claude.ai** comme artifact React.
Chat IA (Claude), multi-profils enfants, analyse audio micro/fichier, feedbacks persistants cross-device.

### Projet complet (avec serveur backend)
```bash
cd BabyBot_IA
chmod +x setup.sh run.sh
./setup.sh    # Installe les dépendances dans un venv
./run.sh      # Lance http://localhost:8000
```
Dans BabyBot, renseignez l'URL serveur `http://localhost:8000` pour activer l'analyse ML complète.

### Installation manuelle
```bash
python3 -m venv venv && source venv/bin/activate
pip install fastapi uvicorn python-multipart pydantic jinja2 python-dotenv loguru sqlalchemy
pip install librosa soundfile numpy scipy scikit-learn chromadb sentence-transformers
mkdir -p data models uploads
python -m uvicorn api.main:app --reload --port 8000
```

## Fonctionnalités

**Multi-profils enfants** — Un profil par enfant (prénom + âge), conseils adaptés. Modifier, supprimer, ou discussion rapide sans profil.

**Chat IA contextuel** — Connecté à Claude. Mémoire de conversation, approfondissement progressif, jamais de réponse générique.

**Analyse audio** — Micro navigateur (son ambiant réel) + upload fichier. Mode serveur : MFCC+SVM/RF → 5 classes. Mode local : heuristiques RMS/ZCR.

**Feedbacks persistants** — 👍/👎 sauvegardés avec horodatage et nom de l'enfant. Persistent cross-session et cross-device. Compteur finetuning LoRA (seuil 100).

**Réinitialisation** — Bouton sur l'écran d'accueil pour supprimer toutes les données.

## Architecture Hexagonale
```
BabyBot_IA/
├── BabyBot_Final.jsx             ← Interface React (artifact Claude.ai)
├── BabyBot_IA_Soutenance.pptx   ← Présentation 9 slides
├── architecture_hexagonale.svg   ← Diagramme d'architecture
├── config.py                     ← Configuration
├── requirements.txt
├── setup.sh / run.sh             ← Scripts installation & lancement
├── test_app.py                   ← Tests automatiques
├── app/
│   ├── domain.py                 ← Entités métier
│   ├── ports.py                  ← Interfaces hexagonales (10 ports)
│   ├── use_cases.py              ← 4 cas d'usage
│   └── adapters/                 ← 7 adaptateurs (audio, ML, LLM, DB...)
├── training/finetune.py          ← Pipeline LoRA + DPO
├── api/main.py                   ← FastAPI REST + WebSocket
└── static/index.html             ← Frontend HTML (version serveur)
```

## API
```
POST /api/chat       — Chatbot parental
POST /api/analyze    — Analyse audio (WAV)
POST /api/feedback   — Feedback 👍/👎
POST /api/finetune   — Déclencher le finetuning
GET  /api/health     — État du serveur
WS   /ws/{parent_id} — Notifications temps réel
```

## Tests
```bash
python test_app.py          # Tests offline (domaine, DB, LLM, config, structure)
python test_app.py --api    # Tests avec serveur lancé
```
