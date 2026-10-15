#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  BabyCry AI — Lancement rapide
# ═══════════════════════════════════════════════════════════

PURPLE='\033[0;35m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${PURPLE}"
echo "  👶 BabyCry AI — Démarrage..."
echo -e "${NC}"

# Activer le venv
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate 2>/dev/null
else
    source venv/bin/activate 2>/dev/null
fi

# Mode de lancement
MODE=${1:-demo}

if [ "$MODE" == "demo" ]; then
    echo -e "${YELLOW}  Mode : DEMO (sans GPU, LLM simulé)${NC}"
    export BABYCRY_MODE=demo
elif [ "$MODE" == "full" ]; then
    echo -e "${GREEN}  Mode : COMPLET (avec LLM Mistral-7B)${NC}"
    export BABYCRY_MODE=full
fi

echo ""
echo -e "${BLUE}  🌐 API :       http://localhost:8000${NC}"
echo -e "${BLUE}  📖 Swagger :   http://localhost:8000/docs${NC}"
echo -e "${BLUE}  🎨 Interface : http://localhost:8000/app${NC}"
echo ""

python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
