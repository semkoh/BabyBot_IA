"""
BabyCry AI - Adaptateur Base de Connaissances (RAG)
Implémentation du port KnowledgeBasePort via ChromaDB + Sentence-Transformers.
"""
from typing import List
from loguru import logger
from app.ports import KnowledgeBasePort
from app.domain import KnowledgeDocument
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL


class ChromaKnowledgeBase(KnowledgeBasePort):
    """
    Base de connaissances vectorielle pour le RAG.
    Stocke des documents sur la parentalité, santé bébé, éducation.
    Recherche sémantique via embeddings Sentence-Transformers.
    """

    COLLECTION_NAME = "parenting_knowledge"

    def __init__(self):
        import chromadb
        from chromadb.utils import embedding_functions

        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"📚 Knowledge Base initialisée : "
            f"{self.collection.count()} documents"
        )

    def add_document(self, document: KnowledgeDocument) -> str:
        """Ajoute un document à la base vectorielle."""
        self.collection.add(
            ids=[document.id],
            documents=[document.content],
            metadatas=[{
                "title": document.title,
                "category": document.category,
                "source": document.source,
            }],
        )
        logger.info(f"📄 Document ajouté : {document.title}")
        return document.id

    def add_documents_batch(self, documents: List[KnowledgeDocument]):
        """Ajoute un batch de documents."""
        if not documents:
            return
        self.collection.add(
            ids=[d.id for d in documents],
            documents=[d.content for d in documents],
            metadatas=[{
                "title": d.title,
                "category": d.category,
                "source": d.source,
            } for d in documents],
        )
        logger.info(f"📄 {len(documents)} documents ajoutés en batch")

    def search(self, query: str, k: int = 5) -> List[KnowledgeDocument]:
        """Recherche sémantique dans la base."""
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count() or 1),
        )

        documents = []
        if results and results['documents']:
            for i, doc_text in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                documents.append(KnowledgeDocument(
                    id=results['ids'][0][i],
                    title=meta.get('title', ''),
                    content=doc_text,
                    category=meta.get('category', ''),
                    source=meta.get('source', ''),
                ))
        return documents

    def seed_knowledge_base(self):
        """Peuple la base avec des connaissances parentalité de référence."""
        if self.collection.count() > 0:
            logger.info("📚 Base déjà peuplée, skip seed")
            return

        seed_docs = [
            KnowledgeDocument(
                title="Les pleurs du nouveau-né",
                content=(
                    "Les bébés pleurent en moyenne 2 à 3 heures par jour. "
                    "Les pleurs de faim sont rythmiques et répétitifs. "
                    "Les pleurs de douleur sont soudains et aigus. "
                    "Les pleurs de fatigue sont geignards et continus. "
                    "Les pleurs de peur sont soudains après un stimulus."
                ),
                category="santé",
                source="OMS - Guide du nouveau-né",
            ),
            KnowledgeDocument(
                title="Le sommeil du nourrisson 0-6 mois",
                content=(
                    "Un nouveau-né dort 16 à 17 heures par jour par cycles de 2-4h. "
                    "À 3 mois, les cycles s'allongent à 5-6h la nuit. "
                    "À 6 mois, la majorité des bébés peuvent dormir 6-8h d'affilée. "
                    "Coucher bébé sur le dos réduit le risque de mort subite. "
                    "Une routine du coucher (bain, histoire, berceuse) aide à l'endormissement."
                ),
                category="sommeil",
                source="HAS - Recommandations sommeil nourrisson",
            ),
            KnowledgeDocument(
                title="L'alimentation du bébé 0-12 mois",
                content=(
                    "Allaitement maternel exclusif recommandé jusqu'à 6 mois. "
                    "La diversification alimentaire débute entre 4 et 6 mois. "
                    "Introduire un aliment à la fois pour détecter les allergies. "
                    "Éviter miel avant 1 an (risque botulisme). "
                    "Les signes de faim : bébé tourne la tête, porte les mains à la bouche."
                ),
                category="alimentation",
                source="ANSES - Alimentation du nourrisson",
            ),
            KnowledgeDocument(
                title="Le développement moteur 0-18 mois",
                content=(
                    "0-3 mois : tient sa tête, suit des yeux, sourit. "
                    "4-6 mois : se retourne, attrape les objets, gazouille. "
                    "7-9 mois : s'assied seul, rampe, dit 'mama/papa'. "
                    "10-12 mois : se met debout, premiers pas, dit quelques mots. "
                    "13-18 mois : marche seul, empile des cubes, vocabulaire de 10-20 mots."
                ),
                category="développement",
                source="Carnet de santé - Ministère de la Santé",
            ),
            KnowledgeDocument(
                title="Quand consulter en urgence",
                content=(
                    "Fièvre > 38°C avant 3 mois : urgence. "
                    "Bébé qui ne réagit pas aux stimuli. "
                    "Pleurs inconsolables pendant plus de 3 heures. "
                    "Vomissements répétés ou diarrhée avec déshydratation. "
                    "Difficultés respiratoires ou teint bleuté. "
                    "Chute ou traumatisme crânien."
                ),
                category="santé",
                source="SAMU - Guide urgences pédiatriques",
            ),
            KnowledgeDocument(
                title="Les coliques du nourrisson",
                content=(
                    "Les coliques touchent 20-25% des nourrissons entre 2 semaines et 4 mois. "
                    "Règle des 3 : pleurs > 3h/jour, > 3 jours/semaine, > 3 semaines. "
                    "Causes possibles : immaturité digestive, aérophagie, stress. "
                    "Solutions : portage, massage abdominal, position sur le ventre sur l'avant-bras."
                ),
                category="santé",
                source="SFP - Société Française de Pédiatrie",
            ),
            KnowledgeDocument(
                title="La communication avec bébé",
                content=(
                    "Parler à bébé stimule le développement du langage dès la naissance. "
                    "Le 'parentais' (voix aiguë, mots simples) capte l'attention de bébé. "
                    "La langue des signes bébé permet de communiquer dès 6-8 mois. "
                    "Lire des histoires dès le plus jeune âge développe le vocabulaire."
                ),
                category="éducation",
                source="Guide parentalité positive - CAF",
            ),
            KnowledgeDocument(
                title="Gérer le stress parental",
                content=(
                    "Le baby blues touche 50-80% des mères dans les jours suivant l'accouchement. "
                    "La dépression post-partum nécessite un suivi médical. "
                    "Signes d'alerte : tristesse persistante, perte d'intérêt, troubles du sommeil. "
                    "Ne pas hésiter à demander de l'aide : famille, professionnels, associations."
                ),
                category="bien-être parental",
                source="Santé Publique France",
            ),
        ]

        self.add_documents_batch(seed_docs)
        logger.info(f"🌱 Base de connaissances peuplée : {len(seed_docs)} documents")
