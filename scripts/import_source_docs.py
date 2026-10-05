"""
Utilidad local y opcional: copia los PDF originales desde donde los tengas
en tu máquina (ej. la carpeta de Descargas) hacia data/, con el nombre
exacto que espera config.DOCUMENTS_MANIFEST.

Por qué las rutas de origen NO están escritas acá ni en ningún archivo
versionado: una ruta como C:\\Users\\simon\\Downloads\\... solo existe en tu
PC. Si quedara hardcodeada en un .py del repo, el pipeline se rompe apenas
alguien (el profesor, un compañero, vos mismo en otra máquina) clone el
repositorio. Por eso viven en tu .env local, que .gitignore ya excluye de
git — el repo entregado nunca depende de tu carpeta de Descargas, solo de
que los PDFs ya estén copiados dentro de data/.

Agregá a tu .env (no se versiona):
    SOURCE_PDF_8617_7=C:\\Users\\simon\\Downloads\\RICE_STO_TOMAS_2026.pdf
    SOURCE_PDF_10559_7=C:\\Users\\simon\\Downloads\\ReglamentodeConvivencia10559 (1).pdf

Uso:
    python -m scripts.import_source_docs
"""

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

import config

load_dotenv()

# Mapea document_id -> nombre de la variable de entorno con la ruta de origen.
# Si en el futuro se suma un tercer colegio, solo hay que agregar una línea acá.
SOURCE_ENV_VAR_BY_DOCUMENT_ID = {
    "8617-7": "SOURCE_PDF_8617_7",
    "10559-7": "SOURCE_PDF_10559_7",
}


def import_source_docs(force: bool = False) -> None:
    for doc_config in config.DOCUMENTS_MANIFEST:
        document_id = doc_config["document_id"]
        destino = Path(doc_config["file_path"])

        if destino.exists() and not force:
            print(f"[import] {destino} ya existe — se omite (force=True para sobreescribir).")
            continue

        env_var = SOURCE_ENV_VAR_BY_DOCUMENT_ID.get(document_id)
        origen = os.getenv(env_var) if env_var else None

        if not origen:
            raise EnvironmentError(
                f"Falta la variable {env_var} en tu .env, con la ruta local del "
                f"PDF de '{doc_config['nombre_establecimiento']}'."
            )
        origen_path = Path(origen)
        if not origen_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de origen: {origen}")

        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origen_path, destino)
        print(f"[import] Copiado: {origen_path.name} -> {destino}")

    print("[import] Importación completa.")


if __name__ == "__main__":
    import_source_docs()