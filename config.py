"""
Configuración central del proyecto. Un solo lugar para las constantes que
tienen que mantenerse coherentes entre el script de ingesta y el de
recuperación — sobre todo EMBEDDING_MODEL y EMBEDDING_DIMENSION, que si
divergen entre ambos rompen la consulta con un error de mismatch de
dimensiones contra el índice de Pinecone.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise EnvironmentError(
            f"Falta la variable de entorno {var_name}. "
            f"Copiá .env.example a .env y completala."
        )
    return value


# --- Credenciales ---
GOOGLE_API_KEY = _require_env("GOOGLE_API_KEY")
PINECONE_API_KEY = _require_env("PINECONE_API_KEY")

# --- Modelos de Gemini ---
# gemini-3.8-flash es el Flash estable vigente en el free tier a sept. 2026.
# Si esto vuelve a fallar con 404 o un error de acceso, el modelo fue dado
# de baja o movido de tier — revisá https://ai.google.dev/gemini-api/docs/deprecations
# antes de asumir que es un bug propio.
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"

# Dimensión de salida del embedder. gemini-embedding-001 devuelve 3072 por
# defecto. Este número TIENE que coincidir exactamente con la dimensión
# configurada en el índice de Pinecone (ver init_pinecone.py) — es el error
# de "Mismatch de Dimensiones" que la clase marca como el más común.
EMBEDDING_DIMENSION = 3072

# --- Infraestructura Pinecone ---
INDEX_NAME = os.getenv("INDEX_NAME", "rag-rice")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
# coseno: la métrica correcta para embeddings de Gemini, que están
# optimizados para similitud coseno, no euclidiana.
PINECONE_METRIC = "cosine"

# --- Recuperación ---
CHUNK_SIZE_TOKENS = 700
CHUNK_OVERLAP_TOKENS = 50
TOP_K = 5
BATCH_UPSERT_SIZE = 150

# --- Manifiesto de documentos ---
# Un namespace por establecimiento: con dos colegios reales, esta es la
# razón de ser del namespace ("nombre del colegio" que pediste), no un
# valor fijo global. Cada documento se ingesta de forma independiente al
# namespace que le corresponde, y la idempotencia se chequea por namespace.
#
# document_id: se usa el Rol Base de Datos (RBD) real de cada colegio en
# vez de inventar un ID artificial — ya es un identificador único y
# oficial que aparece en la portada de cada RICE.
#
# section_pattern: regex para detectar el encabezado de sección en el
# TEXTO de cada página. No usamos los números de página que declara el
# índice de cada documento porque no coinciden con la paginación real
# del PDF (verificado: el índice del RICE Sto Tomás dice que "TÍTULO I"
# está en la página 5; el encabezado real está en la página física 10).
# Detectar el encabezado en el propio texto es inmune a ese desfase.
DOCUMENTS_MANIFEST = [
    {
        "file_path": "data/rice_liceo_sara_blinder_dargoltz.pdf",
        "document_id": "8617-7",
        "nombre_documento": "Reglamento Interno de Convivencia Escolar — RICE 2026",
        "nombre_establecimiento": "Liceo Sara Blinder Dargoltz",
        "namespace": "liceo-sara-blinder-dargoltz",
        "section_pattern": r"^TÍTULO\s+[IVXLCDM]+:\s*(.+)$",
    },
    {
        "file_path": "data/rice_escuela_bernardo_leighton_guzman.pdf",
        "document_id": "10559-7",
        "nombre_documento": "Reglamento Interno de Convivencia Escolar",
        "nombre_establecimiento": "Escuela Bernardo Leighton Guzmán",
        "namespace": "escuela-bernardo-leighton-guzman",
        "section_pattern": r"^CAP[IÍ]TULO\s*N?[°º]?\s*\d+\s*:\s*(.+)$",
    },
]

# Generar un resumen de cada chunk con el LLM es posible pero deliberadamente
# lo dejamos apagado: duplica el costo de la ingesta (una llamada extra al
# LLM por chunk) por un campo cuyo valor es marginal cuando ya guardamos el
# texto completo del chunk en la metadata. Se puede activar más adelante
# si en la evaluación se detecta que hace falta.
GENERATE_CHUNK_SUMMARIES = False