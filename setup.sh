#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  BabyCry AI — Script d'installation et de lancement
# ═══════════════════════════════════════════════════════════
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║         👶 BabyCry AI Setup          ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ─── Vérifications ─────────────────────────────────────────
echo -e "${BLUE}[1/5] Vérification des prérequis...${NC}"

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} $1 trouvé"
        return 0
    else
        echo -e "  ${RED}✗${NC} $1 manquant"
        return 1
    fi
}

check_command python3 || { echo -e "${RED}Installez Python 3.10+ : https://python.org${NC}"; exit 1; }
check_command pip3 || check_command pip || { echo -e "${RED}pip manquant${NC}"; exit 1; }

PYTHON=$(command -v python3 || command -v python)
PIP=$(command -v pip3 || command -v pip)

echo -e "  Python: $($PYTHON --version)"

# ─── Environnement virtuel ─────────────────────────────────
echo ""
echo -e "${BLUE}[2/5] Création de l'environnement virtuel...${NC}"

if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo -e "  ${GREEN}✓${NC} venv créé"
else
    echo -e "  ${YELLOW}→${NC} venv existe déjà"
fi

# Activation
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo -e "  ${GREEN}✓${NC} venv activé"

# ─── Installation des dépendances ──────────────────────────
echo ""
echo -e "${BLUE}[3/5] Installation des dépendances...${NC}"
echo -e "  ${YELLOW}⏳ Cela peut prendre quelques minutes...${NC}"

pip install --upgrade pip -q

# Installation par groupes (pour éviter les conflits)
echo -e "  Installing core..."
pip install fastapi uvicorn python-multipart pydantic jinja2 python-dotenv loguru sqlalchemy alembic -q

echo -e "  Installing audio..."
pip install librosa soundfile numpy scipy -q

echo -e "  Installing ML..."
pip install scikit-learn -q

echo -e "  Installing LLM & RAG..."
pip install chromadb sentence-transformers -q

# Les paquets lourds (torch, transformers) sont optionnels pour le mode démo
echo -e "  ${YELLOW}→${NC} PyTorch & Transformers sont optionnels (mode démo sans GPU)"
echo -e "    Pour les installer : pip install torch transformers peft"

echo -e "  ${GREEN}✓${NC} Dépendances installées"

# ─── Création des dossiers ─────────────────────────────────
echo ""
echo -e "${BLUE}[4/5] Création des dossiers...${NC}"
mkdir -p data models uploads data/chroma_db data/donate_a_cry
echo -e "  ${GREEN}✓${NC} Structure créée"

# ─── Initialisation de la base ─────────────────────────────
echo ""
echo -e "${BLUE}[5/5] Initialisation...${NC}"
$PYTHON -c "
from app.adapters.database import SQLiteDatabaseAdapter
db = SQLiteDatabaseAdapter()
print('  Base de données SQLite initialisée')
" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Base de données prête" || echo -e "  ${YELLOW}→${NC} DB sera créée au premier lancement"

# ─── Terminé ───────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Installation terminée !${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo -e "  Pour lancer l'application :"
echo -e "  ${PURPLE}  ./run.sh${NC}"
echo -e ""
echo -e "  Ou manuellement :"
echo -e "  ${BLUE}  source venv/bin/activate${NC}"
echo -e "  ${BLUE}  python -m uvicorn api.main:app --reload --port 8000${NC}"
echo ""
echo -e "  Puis ouvrir : ${GREEN}http://localhost:8000${NC}"
echo ""
