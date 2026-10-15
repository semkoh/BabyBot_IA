"""
BabyCry AI — Adaptateur LLM Mode Démo
Fonctionne sans GPU ni modèle Mistral téléchargé.
Utilise des réponses pré-construites + logique de matching.
"""
import re
import unicodedata
from typing import List, Optional
from app.ports import LLMPort
from config import CRY_LABELS_FR


# Base de connaissances intégrée pour le mode démo
KNOWLEDGE = {
    "pleur": (
        "Les pleurs sont le principal moyen de communication de votre bébé. "
        "On distingue généralement 5 types :\n\n"
        "🍼 **Faim** — Pleurs rythmiques et répétitifs, bébé porte les mains à la bouche.\n"
        "😢 **Douleur** — Pleurs soudains et aigus, bébé se raidit.\n"
        "😴 **Fatigue** — Pleurs geignards et continus, bébé se frotte les yeux.\n"
        "😨 **Peur** — Pleurs soudains après un stimulus externe.\n"
        "😣 **Inconfort** — Pleurs intermittents, bébé gigote.\n\n"
        "En moyenne, un nouveau-né pleure 2 à 3 heures par jour. "
        "Si les pleurs persistent plus de 3h/jour pendant plus de 3 jours/semaine, "
        "consultez votre pédiatre (règle des 3 pour les coliques)."
    ),
    "sommeil": (
        "Le sommeil évolue rapidement chez le nourrisson :\n\n"
        "• **0-3 mois** : 16-17h/jour par cycles de 2-4h\n"
        "• **3-6 mois** : les cycles s'allongent à 5-6h la nuit\n"
        "• **6-12 mois** : la majorité dorment 6-8h d'affilée\n\n"
        "**Conseils pour un bon sommeil :**\n"
        "1. Instaurez une routine du coucher (bain, histoire, berceuse)\n"
        "2. Couchez bébé sur le dos (prévention mort subite)\n"
        "3. Température idéale : 18-20°C\n"
        "4. Pas d'écran 1h avant le coucher\n"
        "5. Distinguez jour/nuit dès les premières semaines"
    ),
    "alimentation": (
        "L'alimentation du bébé suit des étapes clés :\n\n"
        "**0-6 mois** : Allaitement maternel exclusif recommandé par l'OMS. "
        "Le lait maternel contient tous les nutriments nécessaires.\n\n"
        "**4-6 mois** : Début de la diversification alimentaire :\n"
        "• Commencez par les légumes (carottes, courgettes, patates douces)\n"
        "• Puis les fruits (pomme, poire, banane)\n"
        "• Un nouvel aliment à la fois, attendre 3 jours entre chaque\n\n"
        "**À éviter :**\n"
        "• Miel avant 1 an (risque botulisme)\n"
        "• Lait de vache avant 1 an\n"
        "• Sel et sucre ajoutés"
    ),
    "colique": (
        "Les coliques touchent 20-25% des nourrissons entre 2 semaines et 4 mois.\n\n"
        "**Signes :** Pleurs intenses, visage rouge, poings serrés, "
        "jambes repliées sur le ventre, souvent en fin de journée.\n\n"
        "**Solutions à essayer :**\n"
        "1. Portage en écharpe (contact peau à peau)\n"
        "2. Massage abdominal doux (sens des aiguilles d'une montre)\n"
        "3. Position sur le ventre sur votre avant-bras (position du tigre)\n"
        "4. Bruit blanc ou musique douce\n"
        "5. Bain tiède\n\n"
        "Les coliques disparaissent généralement vers 3-4 mois. "
        "Si les pleurs s'accompagnent de fièvre ou vomissements, consultez."
    ),
    "fièvre": (
        "⚠️ **La fièvre chez le nourrisson nécessite une attention particulière :**\n\n"
        "• **< 3 mois + fièvre > 38°C** → Urgence, consultez immédiatement\n"
        "• **3-6 mois + fièvre > 39°C** → Consultez dans la journée\n"
        "• **> 6 mois + fièvre > 38.5°C** → Surveillez, consultez si > 48h\n\n"
        "**Que faire :**\n"
        "1. Découvrez légèrement bébé\n"
        "2. Hydratez régulièrement (tétées fréquentes ou eau)\n"
        "3. Paracétamol adapté au poids (pas d'aspirine)\n"
        "4. Bain tiède (37°C, pas froid !)\n\n"
        "**SAMU : 15** | **Urgences : 112**"
    ),
    "développement": (
        "Le développement de votre bébé mois par mois :\n\n"
        "**0-3 mois** : Tient sa tête, suit des yeux, premiers sourires sociaux\n"
        "**4-6 mois** : Se retourne, attrape les objets, gazouille, rit\n"
        "**7-9 mois** : S'assied seul, rampe, anxiété de séparation, 'mama/papa'\n"
        "**10-12 mois** : Se met debout, premiers pas, pointe du doigt, quelques mots\n"
        "**13-18 mois** : Marche seul, empile des cubes, 10-20 mots\n\n"
        "⚠️ Chaque bébé a son propre rythme ! Ces repères sont des moyennes. "
        "En cas de doute, parlez-en à votre pédiatre lors des visites de routine."
    ),
    "urgence": (
        "🚨 **Consultez en URGENCE si :**\n\n"
        "• Fièvre > 38°C avant 3 mois\n"
        "• Bébé ne réagit plus aux stimuli\n"
        "• Pleurs inconsolables > 3 heures\n"
        "• Vomissements répétés ou diarrhée avec signes de déshydratation\n"
        "• Difficultés respiratoires ou teint bleuté\n"
        "• Chute ou traumatisme crânien\n"
        "• Éruption cutanée qui ne s'efface pas à la pression\n"
        "• Convulsions\n\n"
        "**Numéros d'urgence :**\n"
        "• SAMU : **15**\n"
        "• Urgences européennes : **112**\n"
        "• SOS Médecins : **3624**\n"
        "• Centres antipoison : **01 40 05 48 48** (Paris)"
    ),
    "bain": (
        "Le bain de bébé, étape par étape :\n\n"
        "**Température :** 37°C (vérifiez avec un thermomètre de bain)\n"
        "**Pièce :** 22-24°C\n"
        "**Durée :** 5-10 minutes maximum pour les premiers mois\n\n"
        "**Fréquence :** 2-3 fois par semaine suffit (la peau de bébé est fragile)\n\n"
        "**Conseils :**\n"
        "• Ne laissez JAMAIS bébé seul dans le bain, même 1 seconde\n"
        "• Préparez tout avant : serviette, vêtements, couche\n"
        "• Savon surgras sans parfum\n"
        "• Séchez bien dans les plis (cou, aisselles, cuisses)"
    ),
}

# Mots-clés pour le matching
KEYWORD_MAP = {
    "pleur": ["pleur", "cri", "pleure", "crie", "larme", "hurle"],
    "sommeil": ["sommeil", "dort", "dormir", "nuit", "dodo", "sieste", "réveil", "coucher", "endorm"],
    "alimentation": ["mang", "aliment", "lait", "tét", "biber", "diversif", "nourr", "solide", "purée"],
    "colique": ["colique", "ventre", "gaz", "mal au ventre", "ballonne", "crampe"],
    "fièvre": ["fièvre", "température", "chaud", "malade", "rhume", "toux", "nez"],
    "développement": ["développ", "marche", "parle", "rampe", "assis", "moteur", "mot", "langage"],
    "urgence": ["urgence", "danger", "médecin", "hôpital", "samu", "grave", "inqui"],
    "bain": ["bain", "laver", "douche", "savon", "eau"],
}

INTENT_PATTERNS = {
    "reveil_nuit": [
        "se reveille la nuit", "reveil la nuit", "reveille la nuit",
        "pourquoi mon enfant se reveille", "pourquoi bebe se reveille",
        "pourquoi il se reveille", "pourquoi elle se reveille",
        "reveille souvent la nuit", "se reveille souvent",
    ],
    "sommeil_developpement": [
        "sommeil influence", "sommeil influence-t-il", "sommeil et developpement",
        "impact du sommeil sur le developpement", "le sommeil joue-t-il",
        "le sommeil est-il important pour le developpement",
    ],
    "refus_manger": [
        "refuse de manger", "refuse le biberon", "refuse le sein",
        "ne veut pas manger", "ne mange pas", "mange tres peu",
        "mange peu", "refuse de boire", "refuse de tetee",
    ],
    "signes_faim": [
        "signes de faim", "a faim", "a-t-il faim", "a-t-elle faim",
        "reconnaitre la faim", "comment savoir si", "comment voir si",
        "mains a la bouche", "cherche le sein", "cherche le biberon",
    ],
    "fievre_urgence": [
        "fievre 38", "temperature 38", "temperature elevee",
        "quand consulter", "dois-je consulter", "urgence fievre",
    ],
}


class DemoLLMAdapter(LLMPort):
    """
    Adaptateur LLM de démonstration.
    Fonctionne sans GPU ni modèle téléchargé.
    Utilise du matching de mots-clés + réponses pré-construites.
    """

    def __init__(self):
        self.name = "DemoLLM (mode hors-ligne)"

    def _normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", (text or "").lower())
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.replace("œ", "oe")

    def _extract_child_age_months(self, text: str) -> Optional[int]:
        normalized = self._normalize(text)

        match = re.search(r"(\d+)\s*mois", normalized)
        if match:
            return int(match.group(1))

        match = re.search(r"(\d+)\s*an", normalized)
        if match:
            return int(match.group(1)) * 12

        return None

    def _extract_user_question(self, text: str) -> str:
        if not text:
            return ""
        parts = re.split(r"\n\s*\ncontexte enfant\s*:", text, flags=re.IGNORECASE)
        return parts[0].strip()

    def _detect_intent(self, text: str) -> str:
        normalized = self._normalize(text)
        for intent, patterns in INTENT_PATTERNS.items():
            if any(pattern in normalized for pattern in patterns):
                return intent
        if "nuit" in normalized and any(token in normalized for token in ["reve", "eveil", "veil"]):
            return "reveil_nuit"
        if "sommeil" in normalized and "develop" in normalized:
            return "sommeil_developpement"
        return ""

    def _match_topic(self, text: str) -> str:
        """Trouve le sujet le plus pertinent via mots-clés."""
        text_lower = self._normalize(text)
        scores = {}
        for topic, keywords in KEYWORD_MAP.items():
            score = sum(1 for kw in keywords if self._normalize(kw) in text_lower)
            if score > 0:
                scores[topic] = score
        if scores:
            return max(scores, key=scores.get)
        return ""

    async def generate_response(
        self, prompt: str, context: str = "", history: List[dict] = None
    ) -> str:
        """Génère une réponse basée sur le matching de mots-clés."""
        user_question = self._extract_user_question(prompt)
        age_months = self._extract_child_age_months(prompt)
        intent = self._detect_intent(user_question)
        topic = self._match_topic(user_question)

        if intent == "reveil_nuit":
            if age_months is not None and age_months <= 6:
                response = (
                    f"À {age_months} mois, se réveiller la nuit est encore fréquent. "
                    "Les causes les plus courantes sont la faim, des cycles de sommeil encore courts, l'inconfort, un besoin de réassurance ou parfois une poussée de développement.\n\n"
                    "Si votre bébé se rendort vite après avoir mangé ou été apaisé, ce n'est pas forcément anormal. "
                    "Essayez une routine régulière, une chambre calme, et observez si les réveils surviennent toujours aux mêmes heures.\n\n"
                    "Consultez si les réveils s'accompagnent de pleurs inhabituels, fièvre, difficultés respiratoires, vomissements ou baisse d'alimentation."
                )
            else:
                response = (
                    "Les réveils nocturnes peuvent venir de plusieurs causes : faim, dentition, cauchemars, maladie, changement de routine ou besoin d'être rassuré.\n\n"
                    "Le plus utile est d'observer le contexte : âge de l'enfant, heure du réveil, durée, signes de douleur ou de faim. "
                    "Une routine stable et des réponses calmes aident souvent.\n\n"
                    "Si tu me donnes son âge et la fréquence des réveils, je peux te répondre plus précisément."
                )
            return self._add_demo_note(response)

        if intent == "sommeil_developpement":
            response = (
                "Oui, le sommeil joue un rôle important dans le développement de l'enfant. "
                "Il participe à la croissance, à la maturation du cerveau, à la mémoire, à l'humeur et à l'apprentissage.\n\n"
                "Un sommeil perturbé de temps en temps n'est pas forcément inquiétant. "
                "En revanche, un manque de sommeil durable peut se voir sur l'irritabilité, l'attention, l'appétit et le rythme global de l'enfant.\n\n"
                "Si tu veux, je peux aussi te dire combien d'heures de sommeil sont habituelles à 5 mois."
            )
            return self._add_demo_note(response)

        if intent == "refus_manger":
            if age_months is not None and age_months < 6:
                response = (
                    f"Si votre bébé a {age_months} mois et refuse de manger, "
                    "il faut d'abord préciser s'il refuse le sein, le biberon ou la diversification.\n\n"
                    "À cet âge, le plus important est de surveiller les signes suivants : moins de couches mouillées, grande somnolence, fièvre, vomissements, difficulté à respirer ou refus répété de boire.\n\n"
                    "S'il refuse seulement un repas mais reste éveillé, hydrate bien, propose à nouveau calmement un peu plus tard, et évite de forcer. "
                    "S'il refuse plusieurs prises de suite ou semble abattu, consulte rapidement un pédiatre."
                )
            else:
                response = (
                    "Quand un enfant refuse de manger, il faut d'abord chercher la cause la plus probable : fatigue, poussée dentaire, début d'infection, nez bouché, stress ou simple manque d'appétit passager.\n\n"
                    "Commencez par proposer de petites quantités, sans forcer, dans un moment calme. "
                    "S'il boit normalement, reste éveillé et urine bien, on peut souvent surveiller sur quelques heures.\n\n"
                    "Consultez si le refus dure, s'accompagne de fièvre, vomissements, douleur, signes de déshydratation ou baisse nette d'énergie."
                )
            return self._add_demo_note(response)

        if intent == "signes_faim":
            if age_months is not None and age_months < 6:
                response = (
                    f"À {age_months} mois, les signes de faim les plus fréquents sont : mains à la bouche, mouvements de succion, bébé qui cherche le sein ou le biberon, agitation croissante, petits bruits ou pleurs qui montent progressivement.\n\n"
                    "Il vaut mieux proposer à manger dès les premiers signes, avant les gros pleurs. "
                    "Un bébé qui pleure fort peut être plus difficile à calmer et à nourrir.\n\n"
                    "Si tu veux, je peux aussi te dire comment distinguer faim, fatigue et coliques."
                )
            else:
                response = (
                    "Les signes de faim les plus courants sont : l'enfant cherche à téter, met les mains ou des objets à la bouche, devient agité, réclame plus tôt que d'habitude ou pleure de façon progressive.\n\n"
                    "Le contexte compte beaucoup : heure du dernier repas, âge, fatigue, poussée dentaire et habitudes de l'enfant. "
                    "Si tu me donnes son âge, je peux te répondre de façon plus précise."
                )
            return self._add_demo_note(response)

        if topic == "fièvre" and "quand consulter" in self._normalize(user_question):
            response = (
                "Pour une fièvre, on consulte rapidement si l'enfant a moins de 3 mois avec 38°C ou plus, s'il respire mal, boit très peu, est inhabituellement mou, convulse, ou présente des signes de déshydratation.\n\n"
                "Chez un bébé plus grand, on surveille aussi la durée de la fièvre, l'état général et les autres symptômes. "
                "Si tu veux, donne-moi son âge exact et sa température, et je te dirai le niveau d'urgence."
            )
            return self._add_demo_note(response)

        if topic and topic in KNOWLEDGE:
            return self._add_demo_note(KNOWLEDGE[topic])

        return (
            "Je suis BabyBot, votre assistant parental. 👶\n\n"
            "Je peux vous aider sur ces sujets :\n"
            "• **Les pleurs** — comprendre pourquoi bébé pleure\n"
            "• **Le sommeil** — conseils pour mieux dormir\n"
            "• **L'alimentation** — allaitement, diversification\n"
            "• **Les coliques** — solutions et remèdes\n"
            "• **La fièvre** — quand s'inquiéter\n"
            "• **Le développement** — étapes mois par mois\n"
            "• **Les urgences** — quand consulter\n"
            "• **Le bain** — comment baigner bébé\n\n"
            "Posez-moi votre question ! 😊"
        )

    def _add_demo_note(self, response: str) -> str:
        return (
            f"{response}\n\n---\n"
            "*💡 Cette réponse est générée en mode démo. "
            "En production, le LLM Mistral-7B fournit des réponses "
            "plus personnalisées et contextuelles.*"
        )

    async def generate_advice(self, cry_type: str) -> str:
        """Génère un conseil basé sur le type de pleur détecté."""
        advice_map = {
            "faim": (
                "🍼 Votre bébé a probablement faim.\n\n"
                "Proposez-lui une tétée ou un biberon. "
                "Les signes de faim incluent : mains à la bouche, "
                "mouvements de succion, agitation croissante. "
                "En général, un nouveau-né mange toutes les 2-3 heures."
            ),
            "douleur": (
                "😢 Votre bébé semble avoir mal.\n\n"
                "Vérifiez : couche sale, vêtement trop serré, "
                "position inconfortable, poussée dentaire. "
                "Prenez-le contre vous (peau à peau si possible). "
                "Si la douleur persiste, consultez votre pédiatre."
            ),
            "fatigue": (
                "😴 Votre bébé est fatigué.\n\n"
                "Signes : se frotte les yeux, bâille, regard fixe. "
                "Emmenez-le dans un endroit calme et sombre. "
                "Bercez-le doucement ou mettez un bruit blanc. "
                "Essayez de respecter ses horaires de sieste."
            ),
            "peur": (
                "😨 Votre bébé a peur.\n\n"
                "Prenez-le dans vos bras pour le rassurer. "
                "Parlez-lui doucement, votre voix le calme. "
                "Éloignez-le de la source de stress si possible. "
                "Le contact physique est le meilleur remède contre la peur."
            ),
            "inconfort": (
                "😣 Votre bébé est inconfortable.\n\n"
                "Vérifiez : couche à changer, température de la pièce "
                "(18-20°C idéal), vêtements adaptés à la saison. "
                "Changez sa position. Un massage doux peut aussi aider."
            ),
        }
        return advice_map.get(cry_type, CRY_LABELS_FR.get(cry_type, ""))
