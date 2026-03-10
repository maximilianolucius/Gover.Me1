"""Recursive web scraper that uses an LLM to extract structured data from
Junta de Andalucia web pages, saving each page as a JSON document.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import json
import time
import random
from datetime import datetime
import hashlib
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import re
import argparse

load_dotenv()


@dataclass
class Config:
    """Runtime configuration loaded from environment variables."""

    GOOGLE_PLACES_API_KEY: str = os.getenv("GOOGLE_PLACES_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "NoLoNecesitas.")

    VLLM_BASE_URL = "http://172.24.250.17:8000/v1"
    VLLM_MODEL = "gemma-3-12b-it"


config = Config()


class ScraperRecursivoLLM:
    """Recursive web scraper that uses an LLM to parse page content into structured JSON."""

    def __init__(self, url_base, texto_filtro="juntadeandalucia.es",
                 directorio_base="./diarios/datos/", enlaces_por_nivel=4, max_depth=3):
        self.url_base = url_base
        self.texto_filtro = texto_filtro.lower()
        self.directorio_base = directorio_base
        self.enlaces_por_nivel = enlaces_por_nivel
        self.max_depth = max_depth
        self.urls_procesadas = set()
        self.url_archivos = {}  # Mapeo URL -> (archivo, fecha_extraccion)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # Inicializar LLM
        print(f'config.VLLM_BASE_URL: {config.VLLM_BASE_URL}')
        print(f'config.VLLM_MODEL: {config.VLLM_MODEL}')

        self.llm = ChatOpenAI(
            openai_api_base=config.VLLM_BASE_URL,
            model=config.VLLM_MODEL,
            temperature=0.38,
            api_key=config.OPENAI_API_KEY,
            max_tokens=1028,
        )

        # Crear directorio si no existe
        os.makedirs(directorio_base, exist_ok=True)
        self._cargar_urls_procesadas()

    def _cargar_urls_procesadas(self):
        """Carga URLs ya procesadas y mapeo URL -> archivo para evitar duplicados y permitir updates"""
        try:
            for archivo in os.listdir(self.directorio_base):
                if archivo.endswith('.json'):
                    try:
                        with open(os.path.join(self.directorio_base, archivo), 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'url_original' in data:
                                url = data['url_original']
                                fecha_extraccion = data.get('fecha_extraccion', '')
                                self.urls_procesadas.add(url)
                                self.url_archivos[url] = (archivo, fecha_extraccion)
                    except Exception as e:
                        print(f"Error leyendo archivo {archivo}: {e}")
        except Exception as e:
            print(f"Error cargando URLs procesadas: {e}")

    def _generar_nombre_archivo(self, url):
        """Genera nombre de archivo único basado en URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"dato_{timestamp}_{url_hash}.json"

    def _limpiar_texto_html(self, html_content):
        """Limpia y extrae texto relevante del HTML"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remover scripts, styles, y otros elementos no deseados
        for tag in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'iframe', 'noscript']):
            tag.decompose()

        # Extraer texto principal
        texto = soup.get_text(separator=' ', strip=True)

        # Limpiar texto más agresivamente
        lineas = (line.strip() for line in texto.splitlines())
        texto_limpio = '\n'.join(line for line in lineas if line and len(line) > 3)

        # Remover múltiples espacios y saltos de línea
        texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
        texto_limpio = re.sub(r'\n\s*\n', '\n', texto_limpio)

        # Remover cualquier resto de HTML que pueda quedar
        texto_limpio = re.sub(r'<[^>]+>', '', texto_limpio)
        texto_limpio = re.sub(r'&[a-zA-Z0-9]+;', '', texto_limpio)

        # Limitar longitud para el LLM (aproximadamente 3000 caracteres)
        if len(texto_limpio) > 3000:
            texto_limpio = texto_limpio[:3000] + "..."

        return texto_limpio.strip()

    def _extraer_con_llm(self, texto_contenido, url):
        """Usa LLM para extraer información estructurada de la página"""

        system_prompt = """Eres un experto extractor de información web. Tu tarea es analizar el contenido de una página web y extraer información estructurada en formato de "noticia" o "artículo".

IMPORTANTE: Todo el texto debe estar completamente limpio, SIN código HTML, SIN etiquetas, SIN caracteres especiales HTML. Solo texto plano y legible.

Extrae la siguiente información y devuélvela en formato JSON válido:
- titulo: El título principal de la página (SOLO TEXTO, sin HTML)
- subtitulo: Subtítulo o descripción secundaria (SOLO TEXTO, sin HTML, si existe)
- autor: Autor del contenido (SOLO TEXTO, sin HTML, si se menciona)
- fecha: Fecha de publicación en formato ISO (YYYY-MM-DDTHH:MM:SS+00:00, si se encuentra)
- fecha_formateada: Fecha en formato legible español (DD/MM/YYYY, si se encuentra)
- contenido: Array de párrafos del contenido principal (cada párrafo como string separado, SOLO TEXTO)

Si algún campo no está disponible, usa null. El JSON debe ser válido y estar entre ```json y ```.
TODO EL CONTENIDO DEBE SER TEXTO PLANO, COMPLETAMENTE LIMPIO DE HTML.

Ejemplo de respuesta:
```json
{
    "titulo": "Título principal del artículo",
    "subtitulo": "Subtítulo si existe",
    "autor": "Nombre del autor",
    "fecha": "2025-08-14T12:08:35+00:00",
    "fecha_formateada": "14/08/2025",
    "contenido": [
        "Primer párrafo del contenido completamente limpio de HTML.",
        "Segundo párrafo del contenido también limpio.",
        "Tercer párrafo y así sucesivamente."
    ]
}
```"""

        user_prompt = f"""Analiza el siguiente contenido de la página web y extrae la información solicitada:

URL: {url}

CONTENIDO:
{texto_contenido}

Devuelve la información extraída en formato JSON:"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.llm(messages)
            respuesta_texto = response.content

            # Extraer JSON de la respuesta
            json_match = re.search(r'```json\s*(.*?)\s*```', respuesta_texto, re.DOTALL)
            if json_match:
                json_texto = json_match.group(1)
                info_extraida = json.loads(json_texto)
                return info_extraida
            else:
                print(f"⚠️ No se encontró JSON válido en la respuesta del LLM")
                return None

        except Exception as e:
            print(f"❌ Error procesando con LLM: {e}")
            return None

    def obtener_enlaces_filtrados(self, url_base):
        """Extrae enlaces que contengan el texto filtro especificado"""
        try:
            print(f"Extrayendo enlaces de: {url_base}")
            response = requests.get(url_base, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            enlaces = soup.find_all('a', href=True)

            enlaces_filtrados = []
            dominio_base = urlparse(self.url_base).netloc

            for enlace in enlaces:
                href = enlace['href']
                url_completa = urljoin(url_base, href)

                # Filtros para enlaces válidos
                if (url_completa.startswith('http') and
                        dominio_base in url_completa):  # Mantener mismo dominio

                    # Aplicar filtro de texto si se especifica
                    if not self.texto_filtro or self.texto_filtro in url_completa.lower():
                        enlaces_filtrados.append(url_completa)

            # Eliminar duplicados y tomar muestra aleatoria
            enlaces_unicos = list(set(enlaces_filtrados))
            if len(enlaces_unicos) > self.enlaces_por_nivel:
                enlaces_unicos = random.sample(enlaces_unicos, self.enlaces_por_nivel)

            print(f"Encontrados {len(enlaces_unicos)} enlaces únicos")
            return enlaces_unicos

        except Exception as e:
            print(f"Error obteniendo enlaces de {url_base}: {e}")
            return []

    def _limpiar_salida_llm(self, info_extraida):
        """Limpia cualquier resto de HTML en la salida del LLM y formatea según estructura requerida"""
        if not info_extraida:
            return info_extraida

        campos_texto = ['titulo', 'subtitulo', 'autor', 'fecha_formateada']

        for campo in campos_texto:
            if campo in info_extraida and info_extraida[campo]:
                # Remover cualquier tag HTML que pueda haber quedado
                texto = str(info_extraida[campo])
                texto = re.sub(r'<[^>]+>', '', texto)
                texto = re.sub(r'&[a-zA-Z0-9]+;', '', texto)
                texto = texto.strip()
                info_extraida[campo] = texto if texto else None

        # Limpiar contenido si es una lista
        if 'contenido' in info_extraida and isinstance(info_extraida['contenido'], list):
            contenido_limpio = []
            for parrafo in info_extraida['contenido']:
                if parrafo:
                    parrafo_limpio = re.sub(r'<[^>]+>', '', str(parrafo))
                    parrafo_limpio = re.sub(r'&[a-zA-Z0-9]+;', '', parrafo_limpio)
                    parrafo_limpio = parrafo_limpio.strip()
                    if parrafo_limpio and len(parrafo_limpio) > 10:  # Solo párrafos con contenido significativo
                        contenido_limpio.append(parrafo_limpio)
            info_extraida['contenido'] = contenido_limpio

            # Crear contenido_completo uniendo los párrafos
            if contenido_limpio:
                info_extraida['contenido_completo'] = '\n\n'.join(contenido_limpio)
            else:
                info_extraida['contenido_completo'] = None
        else:
            info_extraida['contenido'] = []
            info_extraida['contenido_completo'] = None

        # Si no hay fecha, usar fecha actual
        fecha_actual = datetime.now()
        if not info_extraida.get('fecha'):
            info_extraida['fecha'] = fecha_actual.isoformat()
        if not info_extraida.get('fecha_formateada'):
            info_extraida['fecha_formateada'] = fecha_actual.strftime("%d/%m/%Y")

        return info_extraida

    def extraer_pagina(self, url):
        """Extrae información de una página usando LLM"""
        try:
            print(f"🔍 Extrayendo contenido de: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            # Limpiar y preparar texto para LLM
            texto_limpio = self._limpiar_texto_html(response.content)

            if len(texto_limpio.strip()) < 100:
                print(f"⚠️ Contenido muy corto, saltando página")
                return None

            # Extraer información usando LLM
            info_extraida = self._extraer_con_llm(texto_limpio, url)

            if info_extraida:
                # Limpiar salida del LLM
                info_extraida = self._limpiar_salida_llm(info_extraida)

                # Añadir metadatos adicionales
                info_extraida.update({
                    'url_original': url,
                    'fecha_extraccion': datetime.now().isoformat(),
                    'es_noticia_valida': bool(info_extraida.get('titulo') and
                                              info_extraida.get('contenido') and
                                              len(info_extraida.get('contenido', [])) > 0)
                })

                return info_extraida
            else:
                return None

        except Exception as e:
            print(f"Error extrayendo página {url}: {e}")
            return None

    def guardar_pagina(self, info_pagina, profundidad):
        """Guarda la información de la página en formato JSON, actualizando si es más reciente"""
        try:
            info_pagina['profundidad'] = profundidad
            url = info_pagina['url_original']
            fecha_nueva = info_pagina['fecha_extraccion']

            # Verificar si ya existe esta URL
            if url in self.url_archivos:
                archivo_existente, fecha_existente = self.url_archivos[url]

                # Comparar fechas (string ISO se puede comparar directamente)
                if fecha_nueva <= fecha_existente:
                    print(f"⭐️ URL ya existe con fecha más reciente, saltando: {url}")
                    return False
                else:
                    # La nueva es más reciente, eliminar archivo antiguo
                    try:
                        ruta_antigua = os.path.join(self.directorio_base, archivo_existente)
                        if os.path.exists(ruta_antigua):
                            os.remove(ruta_antigua)
                            print(f"🔄 Actualizando archivo más antiguo: {archivo_existente}")
                    except Exception as e:
                        print(f"⚠️ Error eliminando archivo antiguo: {e}")

            # Generar nombre de archivo y guardar
            nombre_archivo = self._generar_nombre_archivo(url)
            ruta_archivo = os.path.join(self.directorio_base, nombre_archivo)

            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                json.dump(info_pagina, f, ensure_ascii=False, indent=2)

            # Actualizar mapeo
            self.url_archivos[url] = (nombre_archivo, fecha_nueva)

            print(f"✅ Dato guardado: {nombre_archivo}")
            return True

        except Exception as e:
            print(f"❌ Error guardando dato: {e}")
            return False

    def procesar_nivel(self, urls, profundidad):
        """Procesa un nivel de URLs y retorna enlaces para el siguiente nivel"""
        print(f"\n🔍 PROCESANDO NIVEL {profundidad}")
        print(f"URLs a procesar: {len(urls)}")

        enlaces_siguiente_nivel = []
        datos_guardados = 0

        for i, url in enumerate(urls, 1):
            # No saltar URLs ya procesadas - permitir actualizaciones
            print(f"\n[{i}/{len(urls)}] Procesando: {url}")

            # Extraer información de la página
            info_pagina = self.extraer_pagina(url)

            if info_pagina and info_pagina['es_noticia_valida']:
                if self.guardar_pagina(info_pagina, profundidad):
                    datos_guardados += 1
                    self.urls_procesadas.add(url)
                    print(f"📄 Título: {info_pagina.get('titulo', 'Sin título')}")
            else:
                print(f"⚠️ No es contenido válido o error en extracción")

            # Si no hemos llegado al máximo depth, buscar más enlaces
            if profundidad < self.max_depth:
                nuevos_enlaces = self.obtener_enlaces_filtrados(url)
                enlaces_siguiente_nivel.extend(nuevos_enlaces)

            # Pausa entre requests para ser respetuoso
            time.sleep(2)

        print(f"\n📊 RESUMEN NIVEL {profundidad}:")
        print(f"   Datos guardados: {datos_guardados}")
        print(f"   Enlaces encontrados para siguiente nivel: {len(enlaces_siguiente_nivel)}")

        return list(set(enlaces_siguiente_nivel))  # Eliminar duplicados

    def ejecutar_scraping(self, url_inicial=None):
        """Ejecuta el scraping recursivo completo"""
        if url_inicial is None:
            url_inicial = self.url_base

        print("🚀 INICIANDO SCRAPING RECURSIVO CON LLM")
        print("=" * 60)
        print(f"URL inicial: {url_inicial}")
        print(f"Filtro de texto: '{self.texto_filtro}'" if self.texto_filtro else "Sin filtro")
        print(f"Profundidad máxima: {self.max_depth}")
        print(f"Enlaces por nivel: {self.enlaces_por_nivel}")
        print(f"Directorio destino: {self.directorio_base}")
        print(f"Modelo LLM: {config.VLLM_MODEL}")
        print("=" * 60)

        # Inicializar con URL base
        urls_actuales = [url_inicial]

        for profundidad in range(1, self.max_depth + 1):
            if not urls_actuales:
                print(f"❌ No hay más URLs para procesar en profundidad {profundidad}")
                break

            # Procesar nivel actual
            urls_siguientes = self.procesar_nivel(urls_actuales, profundidad)

            # Preparar URLs para siguiente nivel
            if urls_siguientes and profundidad < self.max_depth:
                if len(urls_siguientes) > self.enlaces_por_nivel:
                    urls_actuales = random.sample(urls_siguientes, self.enlaces_por_nivel)
                else:
                    urls_actuales = urls_siguientes
            else:
                urls_actuales = []

        print("\n🎉 SCRAPING COMPLETADO")
        print(f"Total URLs procesadas: {len(self.urls_procesadas)}")

        # Generar reporte final
        self.generar_reporte()

    def generar_reporte(self):
        """Genera un reporte del scraping realizado"""
        try:
            archivos = [f for f in os.listdir(self.directorio_base) if f.endswith('.json')]

            reporte = {
                'fecha_reporte': datetime.now().isoformat(),
                'total_datos': len(archivos),
                'datos_por_profundidad': {},
                'autores': {},
                'archivos': archivos
            }

            for archivo in archivos:
                try:
                    with open(os.path.join(self.directorio_base, archivo), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # Contar por profundidad
                        prof = data.get('profundidad', 'desconocido')
                        reporte['datos_por_profundidad'][prof] = reporte['datos_por_profundidad'].get(prof, 0) + 1

                        # Contar por autor
                        autor = data.get('autor', 'Sin autor')
                        if autor:
                            reporte['autores'][autor] = reporte['autores'].get(autor, 0) + 1
                except:
                    continue

            # Guardar reporte
            with open(os.path.join(self.directorio_base, 'reporte_scraping_llm.json'), 'w', encoding='utf-8') as f:
                json.dump(reporte, f, ensure_ascii=False, indent=2)

            print(f"\n📋 REPORTE GENERADO:")
            print(f"   Total datos: {reporte['total_datos']}")
            print(f"   Por profundidad: {reporte['datos_por_profundidad']}")
            print(f"   Top autores: {dict(list(reporte['autores'].items())[:5])}")
            print(f"   Reporte guardado en: reporte_scraping_llm.json")

        except Exception as e:
            print(f"Error generando reporte: {e}")


def main():
    """Función principal que maneja argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Scraper recursivo con LLM para extraer información estructurada de sitios web",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python scraper_recursivo_dato.py
  python scraper_recursivo_dato.py --url "https://ejemplo.com" --depth 3
  python scraper_recursivo_dato.py --url "https://ejemplo.com" --filtro "ejemplo.com" --enlaces 5 --depth 2 --directorio "./datos/"
        """
    )

    parser.add_argument(
        '--url', '--url_base',
        default="https://www.juntadeandalucia.es/datosabiertos/portal.html",
        help="URL base para iniciar el scraping (default: %(default)s)"
    )

    parser.add_argument(
        '--filtro', '--texto_filtro',
        default="juntadeandalucia.es",
        help="Texto que deben contener los enlaces para ser procesados (default: %(default)s)"
    )

    parser.add_argument(
        '--directorio', '--directorio_base',
        default="./diarios/datos/junta_andalucia/",
        help="Directorio base donde guardar los datos extraídos (default: %(default)s)"
    )

    parser.add_argument(
        '--enlaces', '--enlaces_por_nivel',
        type=int,
        default=3,
        help="Número máximo de enlaces a procesar por nivel (default: %(default)d)"
    )

    parser.add_argument(
        '--depth', '--max_depth',
        type=int,
        default=2,
        help="Profundidad máxima del scraping recursivo (default: %(default)d)"
    )

    args = parser.parse_args()

    # Crear y ejecutar scraper con los argumentos proporcionados
    scraper = ScraperRecursivoLLM(
        url_base=args.url,
        texto_filtro=args.filtro,
        directorio_base=args.directorio,
        enlaces_por_nivel=args.enlaces,
        max_depth=args.depth
    )

    scraper.ejecutar_scraping()


if __name__ == "__main__":
    main()


# python scraper_recursivo_dato.py --url "https://www.juntadeandalucia.es/datosabiertos/portal.html" --filtro "juntadeandalucia.es" --directorio "./rag_document_data/datos/junta_andalucia/" --enlaces 3 --depth 2
# python scraper_recursivo_dato.py --url "https://www.juntadeandalucia.es/organismos/ieca/buscar.html" --filtro "juntadeandalucia.es" --directorio "./rag_document_data/datos/junta_andalucia/" --enlaces 3 --depth 2
