"""
Recuperador híbrido: un RAGSystem por colegio (por namespace), combinando
búsqueda semántica (Pinecone) con búsqueda léxica (BM25).

Por qué uno por colegio y no uno global cruzando ambos RICE: ver
DOCUMENTS_MANIFEST en config.py — el namespace existe para separar la
búsqueda por establecimiento, y BM25 indexa en memoria, no en Pinecone, así
que mezclar los chunks de los dos colegios en un mismo índice BM25 rompería
esa separación.

Uso:
    from retriever.rag_system import build_rag_systems
    systems = build_rag_systems()
    systems["escuela-bernardo-leighton-guzman"].query("¿Cuál es el conducto
    regular para reclamos?")
"""

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

import config
from ingest.ingest import chunk_pages, load_and_tag_pages


class RAGSystem:
    """Encapsula el EnsembleRetriever de un colegio. No genera respuesta en
    lenguaje natural — solo recupera y rankea los chunks más relevantes.
    Eso es lo que pide esta entrega; el LLM que redacta una respuesta a
    partir de estos chunks queda para una capa aparte si hace falta."""

    def __init__(self, namespace: str, chunks: list, pinecone_index, embedder: GoogleGenerativeAIEmbeddings):
        self.namespace = namespace

        semantic_retriever = PineconeVectorStore(
            index=pinecone_index,
            embedding=embedder,
            # el texto original se guardó en metadata["texto"] durante la
            # ingesta (no en la clave default "text") — hay que decírselo
            # explícitamente o PineconeVectorStore no sabe reconstruir el
            # page_content de cada Document.
            text_key="texto",
            namespace=namespace,
        ).as_retriever(search_kwargs={"k": config.TOP_K})

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = config.TOP_K

        # pesos parejos: ninguna de las dos señales (semántica/léxica) vale
        # más que la otra a priori para este corpus. LangChain fusiona los
        # rankings con Reciprocal Rank Fusion — no hace falta implementarlo.
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

    def query(self, pregunta: str) -> list:
        """Devuelve el top-5 (config.TOP_K) combinando ambas señales.
        EnsembleRetriever puede fusionar más resultados que TOP_K (hasta la
        unión de lo que devuelve cada sub-retriever) — se recorta acá para
        garantizar exactamente el top-5 que pide la rúbrica."""
        resultados = self.ensemble_retriever.invoke(pregunta)
        return resultados[: config.TOP_K]


def build_rag_systems() -> dict:
    """Construye un RAGSystem por cada documento del manifiesto, indexado
    por namespace. El corpus de BM25 se reconstruye localmente con las
    mismas funciones de ingest.py (no se trae de Pinecone) — así queda
    garantizado que es idéntico al que se usó para generar los embeddings,
    porque corre exactamente el mismo código de carga y chunking."""
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.INDEX_NAME)
    embedder = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)

    systems = {}
    for doc_config in config.DOCUMENTS_MANIFEST:
        pages = load_and_tag_pages(doc_config)
        chunks = chunk_pages(pages)
        systems[doc_config["namespace"]] = RAGSystem(
            namespace=doc_config["namespace"],
            chunks=chunks,
            pinecone_index=index,
            embedder=embedder,
        )
    return systems


if __name__ == "__main__":
    # Prueba rápida y manual, no reemplaza evaluate.py (Etapa 4).
    systems = build_rag_systems()
    for namespace, system in systems.items():
        print(f"\n[rag_system] Namespace '{namespace}' listo.")