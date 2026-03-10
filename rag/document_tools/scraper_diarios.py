"""Recursive news scraper for Andalusian newspapers (ABC.es, Diario Sur, Diario de Sevilla).

Crawls article pages, extracts structured metadata (title, author, date, body)
using newspaper-specific HTML selectors, and saves each article as a JSON file.
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
import argparse


class ScraperRecursivoABC:
    """Recursive HTML scraper for ABC.es, Diario Sur, and Diario de Sevilla articles."""

    def __init__(self, url_base="https://www.abc.es/sevilla/", texto_filtro="sevilla",
                 directorio_base="./diarios/noticias/", enlaces_por_nivel=4, max_depth=4):
        self.url_base = url_base
        self.texto_filtro = texto_filtro.lower()
        self.directorio_base = directorio_base
        self.enlaces_por_nivel = enlaces_por_nivel
        self.max_depth = max_depth
        self.urls_procesadas = set()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # Crear directorio si no existe
        os.makedirs(directorio_base, exist_ok=True)
        self._cargar_urls_procesadas()

    def _cargar_urls_procesadas(self):
        """Carga URLs ya procesadas para evitar duplicados"""
        try:
            for archivo in os.listdir(self.directorio_base):
                if archivo.endswith('.json'):
                    with open(os.path.join(self.directorio_base, archivo), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'url_original' in data:
                            self.urls_procesadas.add(data['url_original'])
        except Exception as e:
            print(f"Error cargando URLs procesadas: {e}")

    def _generar_nombre_archivo(self, url):
        """Genera nombre de archivo único basado en URL"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        return f"noticia_{timestamp}_{url_hash}.json"

    def _es_diario_sevilla(self, url):
        """Detecta si la URL es de diariodesevilla.es"""
        return 'diariodesevilla.es' in url.lower()

    def obtener_enlaces_filtrados(self, url_base):
        """Extrae enlaces que contengan el texto filtro especificado"""
        try:
            print(f"Extrayendo enlaces de: {url_base}")
            response = requests.get(url_base, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            enlaces = soup.find_all('a', href=True)

            enlaces_filtrados = []
            for enlace in enlaces:
                href = enlace['href']
                url_completa = urljoin(url_base, href)

                if (self.texto_filtro in url_completa.lower() and
                        url_completa.startswith('http') and
                        url_completa not in self.urls_procesadas):
                    enlaces_filtrados.append(url_completa)

            # Eliminar duplicados y tomar muestra aleatoria
            enlaces_unicos = list(set(enlaces_filtrados))
            if len(enlaces_unicos) > self.enlaces_por_nivel:
                enlaces_unicos = random.sample(enlaces_unicos, self.enlaces_por_nivel)

            print(f"Encontrados {len(enlaces_unicos)} enlaces únicos con '{self.texto_filtro}'")
            return enlaces_unicos

        except Exception as e:
            print(f"Error obteniendo enlaces de {url_base}: {e}")
            return []

    def extraer_noticia(self, url):
        """Extrae información de una noticia individual"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # MÉTODO ORIGINAL - ABC/Diario Sur
            titulo = None
            titulo_h1 = soup.find('h1', class_='voc-title')
            if titulo_h1:
                titulo = titulo_h1.get_text(strip=True)
            else:
                titulo_tag = soup.find('title')
                if titulo_tag:
                    titulo = titulo_tag.get_text(strip=True)

            autor = None
            autor_elem = soup.find('p', class_='voc-author__name')
            if autor_elem:
                autor_link = autor_elem.find('a')
                if autor_link:
                    autor = autor_link.get_text(strip=True)
                else:
                    autor = autor_elem.get_text(strip=True)

            fecha = None
            fecha_formateada = None
            time_elem = soup.find('time', class_='voc-author__time')
            if time_elem:
                fecha_formateada = time_elem.get_text(strip=True)
                datetime_attr = time_elem.get('datetime')
                if datetime_attr:
                    try:
                        fecha = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00')).isoformat()
                    except:
                        pass

            subtitulo = None
            subtitulo_elem = soup.find('h2', class_='voc-subtitle')
            if subtitulo_elem:
                subtitulo = subtitulo_elem.get_text(strip=True)

            contenido = []
            paragrafos = soup.find_all('p', class_='voc-p')
            for p in paragrafos:
                texto = p.get_text(strip=True)
                if texto:
                    contenido.append(texto)

            # FALLBACK DIARIO DE SEVILLA - Comprobar primero si es diariodesevilla.es
            if (not titulo or not contenido) and self._es_diario_sevilla(url):
                print(f"📄 Fallback Diario de Sevilla activado para {url}")

                # Título fallback Diario Sevilla
                if not titulo:
                    titulo_h1 = soup.find('h1', class_='headline-atom')
                    if titulo_h1:
                        titulo = titulo_h1.get_text(strip=True)

                # Autor fallback Diario Sevilla - buscar en metadatos
                if not autor:
                    meta_autor = soup.find('meta', attrs={'property': 'mrf:authors'})
                    if meta_autor:
                        autor = meta_autor.get('content', '').strip()

                # Fecha fallback Diario Sevilla
                if not fecha:
                    # Buscar en timestamp-atom
                    timestamp_elem = soup.find('p', class_='timestamp-atom')
                    if timestamp_elem:
                        fecha_formateada = timestamp_elem.get_text(strip=True)

                    # Buscar en meta published_time
                    meta_fecha = soup.find('meta', attrs={'property': 'article:published_time'})
                    if meta_fecha:
                        try:
                            fecha = datetime.fromisoformat(
                                meta_fecha.get('content', '').replace('Z', '+00:00')).isoformat()
                        except:
                            pass

                # Subtítulo fallback Diario Sevilla
                if not subtitulo:
                    subtitulo_elem = soup.find('h2', class_='subtitle-atom')
                    if subtitulo_elem:
                        subtitulo = subtitulo_elem.get_text(strip=True)

                # Contenido fallback Diario Sevilla
                if not contenido:
                    # Buscar párrafos con clase paragraph-atom
                    paragrafos = soup.find_all('p', class_='paragraph-atom')
                    for p in paragrafos:
                        texto = p.get_text(strip=True)
                        if texto and len(texto) > 20:
                            contenido.append(texto)

                    # También buscar contenido en div con clase bbnx-body
                    if not contenido:
                        bbnx_body = soup.find('div', class_='bbnx-body')
                        if bbnx_body:
                            paragrafos_body = bbnx_body.find_all('p')
                            for p in paragrafos_body:
                                texto = p.get_text(strip=True)
                                if texto and len(texto) > 20:
                                    contenido.append(texto)

            # FALLBACK ORIGINAL (ABC/Diario Sur) - Si aún no hay contenido y no es Diario Sevilla
            if (not titulo or not contenido) and not self._es_diario_sevilla(url):
                print(f"📄 Fallback ABC/Diario Sur activado para {url}")

                # Título fallback
                if not titulo:
                    titulo_h1 = soup.find('h1', class_='v-a-t')
                    if titulo_h1:
                        titulo = titulo_h1.get_text(strip=True)

                # Autor fallback
                if not autor:
                    autor_elem = soup.find('p', class_='v-mdl-ath__p v-mdl-ath__p--2')
                    if autor_elem:
                        autor_link = autor_elem.find('a')
                        if autor_link:
                            autor = autor_link.get_text(strip=True)
                        else:
                            autor = autor_elem.get_text(strip=True)

                    # Meta tag fallback para autor
                    if not autor:
                        meta_autor = soup.find('meta', attrs={'name': 'author'})
                        if meta_autor:
                            autor = meta_autor.get('content', '').strip()

                # Fecha fallback
                if not fecha:
                    time_elem = soup.find('time', class_='v-mdl-ath__tm')
                    if time_elem:
                        fecha_formateada = time_elem.get_text(strip=True)
                        datetime_attr = time_elem.get('datetime')
                        if datetime_attr:
                            try:
                                fecha = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00')).isoformat()
                            except:
                                pass

                    # Meta tag fallback para fecha
                    if not fecha:
                        meta_fecha = soup.find('meta', attrs={'property': 'article:published_time'})
                        if meta_fecha:
                            try:
                                fecha = datetime.fromisoformat(
                                    meta_fecha.get('content', '').replace('Z', '+00:00')).isoformat()
                            except:
                                pass

                # Subtítulo fallback
                if not subtitulo:
                    subtitulo_elem = soup.find('h2', class_='v-a-sub-t')
                    if subtitulo_elem:
                        subtitulo = subtitulo_elem.get_text(strip=True)

                # Contenido fallback
                if not contenido:
                    paragrafos = soup.find_all('p', class_='v-p')
                    for p in paragrafos:
                        texto = p.get_text(strip=True)
                        if texto and len(texto) > 20:
                            contenido.append(texto)

                    # Buscar en área de paywall si aún no hay contenido
                    if not contenido:
                        paywall_area = soup.find('div', class_='paywall')
                        if paywall_area:
                            paragrafos_paywall = paywall_area.find_all('p')
                            for p in paragrafos_paywall:
                                texto = p.get_text(strip=True)
                                if texto and len(texto) > 20:
                                    contenido.append(texto)

            return {
                'titulo': titulo,
                'subtitulo': subtitulo,
                'autor': autor,
                'fecha': fecha,
                'fecha_formateada': fecha_formateada,
                'contenido': contenido,
                'contenido_completo': '\n\n'.join(contenido) if contenido else None,
                'url_original': url,
                'fecha_extraccion': datetime.now().isoformat(),
                'es_noticia_valida': bool(titulo and contenido),
                'diario_origen': 'diariodesevilla' if self._es_diario_sevilla(url) else 'abc_diariosur'
            }

        except Exception as e:
            print(f"Error extrayendo noticia de {url}: {e}")
            return None

    def guardar_noticia(self, info_noticia, profundidad):
        """Guarda la noticia en formato JSON"""
        try:
            info_noticia['profundidad'] = profundidad

            nombre_archivo = self._generar_nombre_archivo(info_noticia['url_original'])
            ruta_archivo = os.path.join(self.directorio_base, nombre_archivo)

            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                json.dump(info_noticia, f, ensure_ascii=False, indent=2)

            print(f"✅ Noticia guardada: {nombre_archivo}")
            return True

        except Exception as e:
            print(f"❌ Error guardando noticia: {e}")
            return False

    def procesar_nivel(self, urls, profundidad):
        """Procesa un nivel de URLs y retorna enlaces para el siguiente nivel"""
        print(f"\n🔍 PROCESANDO NIVEL {profundidad}")
        print(f"URLs a procesar: {len(urls)}")

        enlaces_siguiente_nivel = []
        noticias_guardadas = 0

        for i, url in enumerate(urls, 1):
            # La URL inicial (nivel 1) nunca debe ser saltada como duplicada
            if url in self.urls_procesadas and profundidad > 1:
                print(f"⏭️ Saltando URL ya procesada: {url}")
                continue

            print(f"\n[{i}/{len(urls)}] Procesando: {url}")

            # Extraer información de la noticia
            info_noticia = self.extraer_noticia(url)

            if info_noticia and info_noticia['es_noticia_valida']:
                if self.guardar_noticia(info_noticia, profundidad):
                    noticias_guardadas += 1
                    self.urls_procesadas.add(url)
            else:
                print(f"⚠️ No es una noticia válida o error en extracción")

            # Si no hemos llegado al máximo depth, buscar más enlaces
            if profundidad < self.max_depth:
                nuevos_enlaces = self.obtener_enlaces_filtrados(url)
                enlaces_siguiente_nivel.extend(nuevos_enlaces)

            # Pausa entre requests
            time.sleep(1)

        print(f"\n📊 RESUMEN NIVEL {profundidad}:")
        print(f"   Noticias guardadas: {noticias_guardadas}")
        print(f"   Enlaces encontrados para siguiente nivel: {len(enlaces_siguiente_nivel)}")

        return list(set(enlaces_siguiente_nivel))  # Eliminar duplicados

    def ejecutar_scraping(self, url_inicial=None):
        """Ejecuta el scraping recursivo completo"""
        if url_inicial is None:
            url_inicial = self.url_base

        print("🚀 INICIANDO SCRAPING RECURSIVO ABC")
        print("=" * 60)
        print(f"URL inicial: {url_inicial}")
        print(f"Filtro de texto: '{self.texto_filtro}'")
        print(f"Profundidad máxima: {self.max_depth}")
        print(f"Enlaces por nivel: {self.enlaces_por_nivel}")
        print(f"Directorio destino: {self.directorio_base}")
        print("=" * 60)

        # Inicializar con URL base
        urls_actuales = [url_inicial]

        for profundidad in range(1, self.max_depth + 1):
            if not urls_actuales:
                print(f"❌ No hay más URLs para procesar en profundidad {profundidad}")
                break

            # Procesar nivel actual
            urls_siguientes = self.procesar_nivel(urls_actuales, profundidad)

            # Preparar URLs para siguiente nivel (tomar muestra aleatoria)
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
                'total_noticias': len(archivos),
                'noticias_por_profundidad': {},
                'noticias_por_diario': {},
                'autores': {},
                'archivos': archivos
            }

            for archivo in archivos:
                with open(os.path.join(self.directorio_base, archivo), 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # Contar por profundidad
                    prof = data.get('profundidad', 'desconocido')
                    reporte['noticias_por_profundidad'][prof] = reporte['noticias_por_profundidad'].get(prof, 0) + 1

                    # Contar por diario origen
                    diario = data.get('diario_origen', 'desconocido')
                    reporte['noticias_por_diario'][diario] = reporte['noticias_por_diario'].get(diario, 0) + 1

                    # Contar por autor
                    autor = data.get('autor', 'Sin autor')
                    reporte['autores'][autor] = reporte['autores'].get(autor, 0) + 1

            # Guardar reporte
            with open(os.path.join(self.directorio_base, 'reporte_scraping.json'), 'w', encoding='utf-8') as f:
                json.dump(reporte, f, ensure_ascii=False, indent=2)

            print(f"\n📋 REPORTE GENERADO:")
            print(f"   Total noticias: {reporte['total_noticias']}")
            print(f"   Por profundidad: {reporte['noticias_por_profundidad']}")
            print(f"   Por diario: {reporte['noticias_por_diario']}")
            print(f"   Reporte guardado en: reporte_scraping.json")

        except Exception as e:
            print(f"Error generando reporte: {e}")


def main():
    """Función principal que maneja argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Scraper recursivo para extraer noticias de diarios digitales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python scraper_recursivo_diarios.py
  python scraper_recursivo_diarios.py --url "https://www.abc.es/madrid/" --filtro "madrid"
  python scraper_recursivo_diarios.py --url "https://www.diariosur.es/" --filtro "diariosur" --depth 3
  python scraper_recursivo_diarios.py --url "https://www.diariodesevilla.es" --filtro "diariodesevilla.es" --depth 3
        """
    )

    parser.add_argument(
        '--url', '--url_base',
        default="https://www.abc.es/sevilla/",
        help="URL base para iniciar el scraping (default: %(default)s)"
    )

    parser.add_argument(
        '--filtro', '--texto_filtro',
        default="sevilla",
        help="Texto que deben contener los enlaces para ser procesados (default: %(default)s)"
    )

    parser.add_argument(
        '--directorio', '--directorio_base',
        default="./diarios/noticias/",
        help="Directorio base donde guardar las noticias (default: %(default)s)"
    )

    parser.add_argument(
        '--enlaces', '--enlaces_por_nivel',
        type=int,
        default=4,
        help="Número máximo de enlaces a procesar por nivel (default: %(default)d)"
    )

    parser.add_argument(
        '--depth', '--max_depth',
        type=int,
        default=4,
        help="Profundidad máxima del scraping recursivo (default: %(default)d)"
    )

    args = parser.parse_args()

    # Crear y ejecutar scraper con los argumentos proporcionados
    scraper = ScraperRecursivoABC(
        url_base=args.url,
        texto_filtro=args.filtro,
        directorio_base=args.directorio,
        enlaces_por_nivel=args.enlaces,
        max_depth=args.depth
    )

    scraper.ejecutar_scraping()


if __name__ == "__main__":
    main()
"""
Ejemplos de uso para los tres diarios:

# ABC Sevilla
python scraper_recursivo_diarios.py --url "https://www.abc.es/sevilla/" --filtro "sevilla" --directorio "./rag_document_data/noticias/" --enlaces 500 --depth 20

# Diario Sur
python scraper_recursivo_diarios.py --url "https://www.diariosur.es/" --filtro "diariosur.es" --directorio "./rag_document_data/noticias/" --enlaces 500 --depth 20

# Diario de Sevilla
python scraper_recursivo_diarios.py --url "https://www.diariodesevilla.es" --filtro "diariodesevilla.es" --directorio "./rag_document_data/noticias/" --enlaces 500 --depth 20


python rag/document_tools/scraper_diarios.py --url "https://www.abc.es/sevilla/" --filtro "sevilla" --directorio "./rag_document_data/noticias/" --enlaces 10 --depth 3
python rag/document_tools/scraper_diarios.py --url "https://www.diariosur.es/" --filtro "diariosur.es" --directorio "./rag_document_data/noticias/" --enlaces 10 --depth 3
python rag/document_tools/scraper_diarios.py --url "https://www.diariodesevilla.es" --filtro "diariodesevilla.es" --directorio "./rag_document_data/noticias/" --enlaces 10 --depth 3


python rag/document_tools/scraper_diarios.py --url "https://www.abc.es/espana/andalucia/aulas-andalucia-pierden-18000-alumnos-curso-escolar-20250826182155-nts.html" --filtro "andalucia" --directorio "./rag_document_data/noticias/" --enlaces 10 --depth 4
python rag/document_tools/scraper_diarios.py --url "https://www.abc.es/sevilla/" --filtro "sevilla" --directorio "./rag_document_data/noticias/" --enlaces 5000 --depth 12
python scraper_recursivo_diarios.py --url "https://www.diariosur.es/" --filtro "diariosur.es" --directorio "./rag_document_data/noticias/" --enlaces 5000 --depth 30
"""
