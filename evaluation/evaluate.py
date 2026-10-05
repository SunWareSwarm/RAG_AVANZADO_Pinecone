"""
Evalúa el RAGSystem de cada colegio contra un Golden Set de preguntas con
documento_id_esperado conocido de antemano.

Definiciones (a nivel de documento, no de chunk — el Golden Set conoce el
documento correcto, no el chunk exacto):
    Recall@5    = 1 si el documento esperado aparece entre los 5 chunks
                  recuperados, 0 si no. Se promedia entre preguntas.
    Precision@5 = (# de los 5 chunks recuperados que pertenecen al
                  documento esperado) / 5. Se promedia entre preguntas.

Uso:
    python -m evaluation.evaluate
"""

import json
from pathlib import Path

import config
from retriever.rag_system import build_rag_systems

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def load_golden_set() -> list:
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def evaluate_question(rag_system, pregunta: str, documento_id_esperado: str) -> dict:
    resultados = rag_system.query(pregunta)
    doc_ids_recuperados = [r.metadata.get("document_id") for r in resultados]

    aciertos = sum(1 for doc_id in doc_ids_recuperados if doc_id == documento_id_esperado)
    recall = 1.0 if aciertos > 0 else 0.0
    precision = aciertos / config.TOP_K

    return {
        "pregunta": pregunta,
        "documento_id_esperado": documento_id_esperado,
        "doc_ids_recuperados": doc_ids_recuperados,
        "recall_at_5": recall,
        "precision_at_5": precision,
    }


def run_evaluation() -> None:
    golden_set = load_golden_set()
    systems = build_rag_systems()

    resultados_por_pregunta = []
    for item in golden_set:
        namespace = item["namespace"]
        if namespace not in systems:
            raise ValueError(
                f"El Golden Set referencia el namespace '{namespace}', que no "
                f"existe en config.DOCUMENTS_MANIFEST."
            )
        resultado = evaluate_question(
            systems[namespace], item["pregunta"], item["documento_id_esperado"]
        )
        resultados_por_pregunta.append(resultado)

    print("\n=== Detalle por pregunta ===")
    for r in resultados_por_pregunta:
        estado = "✓" if r["recall_at_5"] == 1.0 else "✗"
        print(f"\n[{estado}] {r['pregunta']}")
        print(f"    Esperado:   {r['documento_id_esperado']}")
        print(f"    Recuperado: {r['doc_ids_recuperados']}")
        print(f"    Recall@5: {r['recall_at_5']:.2f}  |  Precision@5: {r['precision_at_5']:.2f}")

    n = len(resultados_por_pregunta)
    recall_promedio = sum(r["recall_at_5"] for r in resultados_por_pregunta) / n
    precision_promedio = sum(r["precision_at_5"] for r in resultados_por_pregunta) / n

    print("\n=== Resumen ===")
    print(f"Preguntas evaluadas: {n}")
    print(f"Recall@5 promedio:    {recall_promedio:.2%}")
    print(f"Precision@5 promedio: {precision_promedio:.2%}")


if __name__ == "__main__":
    run_evaluation()