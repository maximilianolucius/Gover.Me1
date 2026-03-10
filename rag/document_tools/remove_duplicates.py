#!/usr/bin/env python3
"""Deduplication utility that removes duplicate scraped JSON files.

Identifies duplicates by comparing the ``url_original`` field across all JSON
files in a directory and interactively offers to delete the extras.
"""

import json
import os
import glob
from pathlib import Path


def remove_duplicates_by_url(directory_path="./rag_document_data/diarios/noticias/"):
    """Elimina archivos JSON duplicados usando la URL como clave única"""

    json_files = glob.glob(os.path.join(directory_path, "*.json"))

    if not json_files:
        print(f"No se encontraron archivos JSON en {directory_path}")
        return

    url_to_file = {}  # {url: filepath}
    duplicates = []
    errors = []

    # Procesar archivos y detectar duplicados
    for filepath in json_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            url = data.get('url_original')
            if not url:
                errors.append(f"Sin URL: {filepath}")
                continue

            if url in url_to_file:
                # Duplicado encontrado
                duplicates.append(filepath)
                print(f"Duplicado: {Path(filepath).name} -> {url}")
            else:
                # Primera ocurrencia de esta URL
                url_to_file[url] = filepath

        except (json.JSONDecodeError, IOError) as e:
            errors.append(f"Error leyendo {filepath}: {e}")

    # Eliminar duplicados
    if duplicates:
        confirm = input(f"\n¿Eliminar {len(duplicates)} archivos duplicados? (s/N): ")
        if confirm.lower() in ['s', 'si', 'y', 'yes']:
            for dup_file in duplicates:
                try:
                    # os.remove(dup_file)
                    print(f"Eliminado: {Path(dup_file).name}")
                except OSError as e:
                    errors.append(f"Error eliminando {dup_file}: {e}")

    # Reporte final
    print(f"\n📊 Resumen:")
    print(f"   Archivos procesados: {len(json_files)}")
    print(f"   URLs únicas: {len(url_to_file)}")
    print(
        f"   Duplicados eliminados: {len(duplicates) if duplicates and confirm.lower() in ['s', 'si', 'y', 'yes'] else 0}")
    print(f"   Errores: {len(errors)}")

    if errors:
        print("\n❌ Errores:")
        for error in errors[:5]:  # Solo mostrar primeros 5
            print(f"   {error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Eliminar archivos JSON duplicados por URL")
    parser.add_argument("--dir", default="./rag_document_data/diarios/noticias/",
                        help="Directorio con archivos JSON (default: %(default)s)")

    args = parser.parse_args()
    remove_duplicates_by_url(args.dir)