"""
Pipeline de ingesta para RAG_RICE.

Flujo por cada documento del manifiesto (config.DOCUMENTS_MANIFEST):
    1. Cargar el PDF página por página (PyMuPDFLoader).
    2. Detectar en qué sección del documento cae cada página, buscando el
       encabezado directamente en el texto de esa página (no confiamos en
       los números de página que declara el índice del documento — ver
       nota en config.py).
    3. Fragmentar cada página en chunks de ~700 tokens con 50 de overlap.
       Se fragmenta página por página (no el documento completo de un
       tirón) para que cada chunk herede sin ambigüedad el número de
       página real de donde salió. El costo de esto es que un chunk no
       puede combinar texto de dos páginas distintas — se pierde algo de
       continuidad exactamente en el borde de página, a cambio de que el
       metadato "pagina" sea siempre exacto.
    4. Generar embeddings con Gemini y subir a Pinecone en lotes seguros (20 chunks),
       con el texto original guardado en la metadata y control de Rate Limit (429).

Idempotencia: si el namespace de un documento ya tiene vectores cargados,
se omite ese documento (usar force_reindex=True para forzar la recarga).

Uso:
    python -m ingest.ingest
"""

import re
import time

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import config

# Reintento con backoff exponencial ante límites de tasa (429) o errores temporales de la API.
_EMBED_RETRY = retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    reraise=True,
)


def _detect_section(page_text: str, pattern: str, current_section: str) -> str:
    """Busca el encabezado de sección en el texto de la página. Si no
    encuentra uno nuevo, la página hereda la sección de la página
    anterior (asume que sigue dentro de la misma sección)."""
    match = re.search(pattern, page_text, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return current_section


def load_and_tag_pages(doc_config: dict) -> list:
    """Carga el PDF y devuelve una lista de Document (uno por página) con
    la metadata de negocio ya asignada a cada uno."""
    loader = PyMuPDFLoader(doc_config["file_path"])
    pages = loader.load()  # PyMuPDFLoader numera page.metadata['page'] desde 0

    current_section = "Preliminares"
    for page in pages:
        current_section = _detect_section(
            page.page_content, doc_config["section_pattern"], current_section
        )
        page.metadata.update(
            {
                "document_id": doc_config["document_id"],
                "nombre_documento": doc_config["nombre_documento"],
                "nombre_establecimiento": doc_config["nombre_establecimiento"],
                "seccion": current_section,
                # +1 para que coincida con el número de página que ve un
                # humano al abrir el PDF (PyMuPDFLoader es 0-indexado).
                "pagina": page.metadata["page"] + 1,
            }
        )

    return pages


def chunk_pages(pages: list) -> list:
    """Fragmenta cada página por separado. split_documents() procesa cada
    Document de la lista de forma independiente y le copia su metadata
    (incluida 'pagina') a cada chunk resultante — por eso partimos de
    Documents por página y no del texto del PDF entero concatenado."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
    )
    chunks = splitter.split_documents(pages)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks


def get_embedding_function() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)


def namespace_already_loaded(index, namespace: str) -> bool:
    stats = index.describe_index_stats()
    namespace_stats = stats.namespaces.get(namespace)
    return bool(namespace_stats and namespace_stats.get("vector_count", 0) > 0)


@_EMBED_RETRY
def _embed_with_retry(embedder: GoogleGenerativeAIEmbeddings, texts: list) -> list:
    return embedder.embed_documents(texts)


def upsert_chunks(index, embedder: GoogleGenerativeAIEmbeddings, chunks: list, namespace: str) -> None:
    # Usamos lotes seguros de máximo 20 chunks para no saturar las cuotas por minuto de Gemini
    batch_size = min(getattr(config, "BATCH_UPSERT_SIZE", 20), 20)
    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.page_content for c in batch]
        
        # Generar embeddings con reintento automático
        embeddings = _embed_with_retry(embedder, texts)

        vectors = []
        for chunk, values in zip(batch, embeddings):
            vector_id = f"{chunk.metadata['document_id']}-chunk-{chunk.metadata['chunk_index']}"
            metadata = dict(chunk.metadata)
            # El texto original vive en la metadata para no tener que
            # consultar otra base al momento de responder.
            metadata["texto"] = chunk.page_content
            vectors.append({"id": vector_id, "values": values, "metadata": metadata})

        index.upsert(vectors=vectors, namespace=namespace)
        print(f"[ingest]   lote {start}-{min(start + batch_size, total)} de {total} subido")
        
        # Pausa de 2 segundos entre lotes para respetar el Rate Limit de la API de Google Gemini
        time.sleep(2)


def run_ingestion(force_reindex: bool = False) -> None:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.INDEX_NAME)
    embedder = get_embedding_function()

    for doc_config in config.DOCUMENTS_MANIFEST:
        namespace = doc_config["namespace"]

        if not force_reindex and namespace_already_loaded(index, namespace):
            print(
                f"[ingest] Namespace '{namespace}' ya tiene datos — se omite "
                f"(force_reindex=True para forzar recarga)."
            )
            continue

        print(f"[ingest] Procesando: {doc_config['nombre_documento']} ({doc_config['nombre_establecimiento']})")
        pages = load_and_tag_pages(doc_config)
        chunks = chunk_pages(pages)
        print(f"[ingest]   {len(pages)} páginas -> {len(chunks)} chunks")
        upsert_chunks(index, embedder, chunks, namespace)

    print("[ingest] Ingesta completa.")


if __name__ == "__main__":
    run_ingestion()