"""Tourism data extractor that reads Andalusian tourism Excel spreadsheets
and produces multiple RAG-optimised text representations per metric for
improved vector-search retrieval.
"""

import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Any
import json


class UltimateTourismExtractor:
    """
    Extractor definitivo que captura TODOS los datos turísticos
    optimizado para RAG con múltiples formatos de búsqueda
    """

    def __init__(self):
        self.nationality_patterns = {
            'britanicos': ['británic', 'britanic', 'british', 'uk', 'reino unido'],
            'alemanes': ['alemán', 'aleman', 'german', 'alemania'],
            'extranjeros': ['extranjero', 'foreign', 'internacional']
        }

        self.month_names = {
            'ene': 'enero', 'feb': 'febrero', 'mar': 'marzo', 'abr': 'abril',
            'may': 'mayo', 'jun': 'junio', 'jul': 'julio', 'ago': 'agosto',
            'sep': 'septiembre', 'oct': 'octubre', 'nov': 'noviembre', 'dic': 'diciembre'
        }

    def extract_from_excel(self, excel_path: str) -> List[Dict[str, Any]]:
        """
        Extrae datos de un Excel turístico con múltiples formatos para RAG
        """
        try:
            # Leer Excel saltando headers
            df = pd.read_excel(excel_path, header=None, skiprows=2)

            # Extraer metadatos del filename
            filename = Path(excel_path).stem
            nationality, month, year = self._parse_filename(filename)

            records = []
            current_section = ""

            for idx, row in df.iterrows():
                # Detectar secciones
                if pd.notna(row.iloc[1]) and any(x in str(row.iloc[1]).upper() for x in ['DATOS', 'BÁSICOS']):
                    current_section = str(row.iloc[1]).strip()
                    continue

                # Extraer datos numéricos
                if self._is_valid_data_row(row):
                    metric_name = str(row.iloc[1]).strip()
                    value = row.iloc[2]
                    variation = row.iloc[3] if pd.notna(row.iloc[3]) else None
                    period = str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else ""

                    # Crear múltiples registros optimizados para RAG
                    base_info = {
                        'archivo_origen': Path(excel_path).name,
                        'nacionalidad': nationality,
                        'mes': month,
                        'año': year,
                        'seccion': current_section,
                        'metrica_original': metric_name,
                        'valor': value,
                        'variacion': variation,
                        'periodo': period
                    }

                    # Generar múltiples formatos de texto para mejor retrieval
                    records.extend(self._generate_rag_records(base_info))

            return records

        except Exception as e:
            print(f"Error procesando {excel_path}: {e}")
            return []

    def _parse_filename(self, filename: str) -> tuple:
        """Extract nationality, month, and year from the Excel filename."""
        # Buscar nacionalidad
        nationality = "desconocido"
        for nat, patterns in self.nationality_patterns.items():
            if any(p in filename.lower() for p in patterns):
                nationality = nat
                break

        # Buscar mes
        month = "desconocido"
        for short, full in self.month_names.items():
            if short in filename.lower():
                month = full
                break

        # Buscar año
        year_match = re.search(r'(\d{2})', filename)
        year = f"20{year_match.group(1)}" if year_match else "2025"

        return nationality, month, year

    def _is_valid_data_row(self, row) -> bool:
        """Return True if the row contains a named metric with a positive numeric value."""
        return (
                pd.notna(row.iloc[2]) and
                isinstance(row.iloc[2], (int, float)) and
                row.iloc[2] > 0 and
                pd.notna(row.iloc[1]) and
                len(str(row.iloc[1]).strip()) > 5
        )

    def _generate_rag_records(self, base_info: Dict) -> List[Dict]:
        """
        Genera múltiples formatos de registro para optimizar RAG retrieval
        """
        records = []

        # Información base
        nationality = base_info['nacionalidad']
        month = base_info['mes']
        year = base_info['año']
        metric = base_info['metrica_original']
        value = base_info['valor']
        period = base_info['periodo']
        variation = base_info['variacion']

        # 1. FORMATO PRINCIPAL - Respuesta directa
        if 'viajeros' in metric.lower() or 'turistas' in metric.lower():
            main_text = f"Turistas {nationality} {period}: {value:,.0f} viajeros"
            if variation:
                main_text += f" (variación: {variation:.1%})"

            records.append({
                **base_info,
                'texto_completo': main_text,
                'tipo_registro': 'numero_turistas',
                'pregunta_objetivo': f"¿cuántos turistas {nationality} {month} {year}?",
                'respuesta_directa': f"{value:,.0f} turistas {nationality}"
            })

        # 2. FORMATO PERNOCTACIONES
        if 'pernoctaciones' in metric.lower():
            pernoc_text = f"Pernoctaciones {nationality} {period}: {value:,.0f} noches"
            if variation:
                pernoc_text += f" (variación: {variation:.1%})"

            records.append({
                **base_info,
                'texto_completo': pernoc_text,
                'tipo_registro': 'numero_pernoctaciones',
                'pregunta_objetivo': f"¿cuántas pernoctaciones {nationality} {month} {year}?",
                'respuesta_directa': f"{value:,.0f} pernoctaciones"
            })

        # 3. FORMATO BÚSQUEDA POR UBICACIÓN
        location_text = f"Andalucía recibió {value:,.0f} "
        if 'viajeros' in metric.lower():
            location_text += f"turistas {nationality} en {period}"
        elif 'pernoctaciones' in metric.lower():
            location_text += f"pernoctaciones de turistas {nationality} en {period}"

        records.append({
            **base_info,
            'texto_completo': location_text,
            'tipo_registro': 'datos_andalucia',
            'ubicacion': 'andalucia'
        })

        # 4. FORMATO TEMPORAL ESPECÍFICO
        temporal_text = f"En {month} de {year}, "
        if 'viajeros' in metric.lower():
            temporal_text += f"llegaron {value:,.0f} turistas {nationality} a Andalucía"
        elif 'pernoctaciones' in metric.lower():
            temporal_text += f"se registraron {value:,.0f} pernoctaciones de turistas {nationality}"

        records.append({
            **base_info,
            'texto_completo': temporal_text,
            'tipo_registro': 'busqueda_temporal',
            'mes_especifico': month,
            'año_especifico': year
        })

        # 5. FORMATO COMPARATIVO (si hay variación)
        if variation and abs(variation) > 0.001:
            comp_text = f"Los turistas {nationality} en {period} "
            if variation > 0:
                comp_text += f"aumentaron un {variation:.1%} hasta {value:,.0f}"
            else:
                comp_text += f"disminuyeron un {abs(variation):.1%} hasta {value:,.0f}"

            records.append({
                **base_info,
                'texto_completo': comp_text,
                'tipo_registro': 'analisis_comparativo',
                'tendencia': 'aumento' if variation > 0 else 'descenso'
            })

        # 6. FORMATO MÉTRICA TÉCNICA (para consultas específicas)
        tech_text = f"MÉTRICA: {metric} | VALOR: {value:,.0f} | PERÍODO: {period} | NACIONALIDAD: {nationality}"
        if variation:
            tech_text += f" | VARIACIÓN: {variation:.3f}"

        records.append({
            **base_info,
            'texto_completo': tech_text,
            'tipo_registro': 'datos_tecnicos'
        })

        return records

    def process_all_files(self, excel_directory: str) -> List[Dict]:
        """Process all .xlsx files in the given directory and return combined records."""
        all_records = []
        excel_files = list(Path(excel_directory).glob("*.xlsx"))

        print(f"🔄 Procesando {len(excel_files)} archivos Excel...")

        for excel_file in excel_files:
            records = self.extract_from_excel(str(excel_file))
            all_records.extend(records)
            print(f"✅ {excel_file.name}: {len(records)} registros extraídos")

        print(f"📊 TOTAL: {len(all_records)} registros para RAG")
        return all_records

    def save_for_rag(self, records: List[Dict], output_file: str):
        """Save records to a JSON file and print type-level statistics."""
        # Agrupar por tipos para estadísticas
        by_type = {}
        for record in records:
            tipo = record.get('tipo_registro', 'otros')
            by_type.setdefault(tipo, []).append(record)

        print(f"\n📈 ESTADÍSTICAS POR TIPO:")
        for tipo, items in by_type.items():
            print(f"  - {tipo}: {len(items)} registros")

        # Guardar
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"💾 Datos guardados en: {output_file}")


# FUNCIÓN PARA INTEGRAR CON TU SISTEMA EXISTENTE
def upgrade_your_rag_system():
    """Run the full extraction pipeline and save RAG-ready data to disk."""
    extractor = UltimateTourismExtractor()

    # Procesar todos los archivos
    records = extractor.process_all_files("./nexus/Excels_Limpios/")

    # Guardar para RAG
    extractor.save_for_rag(records, "./data/tourism_data_ultimate.json")

    print("\n🚀 SISTEMA ACTUALIZADO:")
    print("  - Múltiples formatos por dato (6 formatos por métrica)")
    print("  - Búsqueda optimizada por pregunta natural")
    print("  - Respuestas directas precalculadas")
    print("  - Contexto temporal y geográfico")
    print("  - Análisis comparativo automático")

    return records


# INTEGRACIÓN CON TU CÓDIGO EXISTENTE
def enhanced_tourism_processor(excel_files: List[str]) -> List[Dict]:
    """Drop-in replacement for TourismDataProcessor.extract_all_excel_data()."""
    extractor = UltimateTourismExtractor()
    all_records = []

    for excel_file in excel_files:
        records = extractor.extract_from_excel(excel_file)
        all_records.extend(records)

    return all_records


if __name__ == "__main__":
    # Ejecutar actualización completa
    upgrade_your_rag_system()