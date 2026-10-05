# Sistema RAG Avanzado con Pinecone y Búsqueda Híbrida (RAG_RICE)

Este repositorio contiene la implementación de un pipeline de **Retrieval-Augmented Generation (RAG)** diseñado para la ingesta, indexación y consulta precisa de Reglamentos Internos de Convivencia Escolar (RICE). El sistema integra vectorización semántica en **Pinecone Serverless** con filtros por metadatos y búsqueda léxica (BM25) mediante un `EnsembleRetriever` de LangChain.

---

## 📌 Características Principales

- **Carga y Etiquetado Preciso por Página**: Utiliza `PyMuPDFLoader` procesando cada PDF página por página para garantizar que cada chunk de texto herede de forma exacta su número de página real.
- **Detección Dinámica de Secciones**: Aplica expresiones regulares sobre el texto de cada página para asignar y propagar la sección reglamentaria correspondiente a la que pertenece.
- **Ingesta Robusta e Idempotente**:
  - Verifica la existencia de datos en el *namespace* de Pinecone antes de procesar para evitar re-procesamientos duplicados (permite forzar la recarga con `force_reindex=True`).
  - Control de Rate Limits (Error `429 RESOURCE_EXHAUSTED` de Google Gemini Embeddings) mediante subidas en lotes reducidos (20 chunks) y reintentos automáticos con *exponential backoff* vía `tenacity`.
- **Evaluación Automatizada**: Incluye un modulo de benchmarking para medir la exactitud de recuperación sobre un *Golden Set* de preguntas de negocio (**Recall@5** y **Precision@5**).

---

## 🛠️ Requisitos Previos e Instalación

### 1. Clonar el repositorio
```bash
git clone <URL_DE_TU_REPOSITORIO>
cd RAG_AVANZADO_Pinecone

# Crear entorno virtual
python -m venv venv

# Activar en Windows (PowerShell)
.\venv\Scripts\activate

# Activar en Linux/macOS
source venv/bin/activate

pip install -r requirements.txt

PINECONE_API_KEY=tu_pinecone_api_key
GOOGLE_API_KEY=tu_google_api_key
INDEX_NAME=nombre-de-tu-indice

python -m ingest.ingest
python -m evaluation.evaluate



## 📊 Resultados de la Evaluación (Top-5)

Resultados obtenidos al ejecutar `python -m evaluation.evaluate` sobre el Golden Set:

| Métrica | Resultado |
| :--- | :--- |
| **Preguntas evaluadas** | 5 / 5 |
| **Recall@5 Promedio** | 100.00% |
| **Precision@5 Promedio** | 100.00% |

### Detalle por Consulta

1. **¿En qué plazo deben los adultos del establecimiento denunciar un delito del que tomen conocimiento?**
   - Esperado: `8617-7` | Recall@5: `1.00` | Precision@5: `1.00`
2. **¿Qué consecuencia disciplinaria tienen los atrasos reiterados sin justificación?**
   - Esperado: `8617-7` | Recall@5: `1.00` | Precision@5: `1.00`
3. **¿Existe un protocolo específico sobre el uso de celulares y dispositivos móviles dentro del colegio?**
   - Esperado: `8617-7` | Recall@5: `1.00` | Precision@5: `1.00`
4. **¿Cuál es el conducto regular para presentar un reclamo o una sugerencia?**
   - Esperado: `10559-7` | Recall@5: `1.00` | Precision@5: `1.00`
5. **¿En qué plazo puede un estudiante apelar una medida de suspensión?**
   - Esperado: `10559-7` | Recall@5: `1.00` | Precision@5: `1.00`