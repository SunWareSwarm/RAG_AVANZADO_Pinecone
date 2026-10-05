"""
Inicializa el índice de Pinecone si no existe todavía. Correrlo dos veces
no crea un índice duplicado ni rompe nada — es seguro incluirlo al
principio del pipeline de ingesta.

Uso:
    python init_pinecone.py
"""

from pinecone import Pinecone, ServerlessSpec

import config


def ensure_index_exists() -> None:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()

    if config.INDEX_NAME in existing_indexes:
        print(f"[init_pinecone] El índice '{config.INDEX_NAME}' ya existe. Nada que hacer.")
        return

    print(
        f"[init_pinecone] Creando índice '{config.INDEX_NAME}' "
        f"(dimensión={config.EMBEDDING_DIMENSION}, métrica={config.PINECONE_METRIC})..."
    )
    pc.create_index(
        name=config.INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        metric=config.PINECONE_METRIC,
        spec=ServerlessSpec(
            cloud=config.PINECONE_CLOUD,
            region=config.PINECONE_REGION,
        ),
    )
    print(f"[init_pinecone] Índice '{config.INDEX_NAME}' creado.")


if __name__ == "__main__":
    ensure_index_exists()