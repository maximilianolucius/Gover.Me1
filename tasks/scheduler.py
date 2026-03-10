#!/usr/bin/env python3
"""Scheduler daemon for periodic scraping and document uploading.

Runs the RAG document uploader every 4 hours and launches web scrapers
(government data portals, newspapers) daily at 21:00 in parallel threads.
"""

import schedule
import subprocess
import time
import logging
import threading
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_uploader_scheduler.log'),
        logging.StreamHandler()
    ]
)


def run_uploader():
    """Ejecutar el uploader de documentos"""
    try:
        logging.info("🚀 Iniciando carga de documentos...")

        result = subprocess.run(
            ['python', 'rag/document_tools/uploader.py', './rag_document_data/'],
            capture_output=True,
            text=True,
            timeout=3600  # Timeout de 1 hora
        )

        if result.returncode == 0:
            logging.info("✅ Carga completada exitosamente")
            logging.info(f"Output: {result.stdout}")
        else:
            logging.error(f"❌ Error en la carga: {result.stderr}")

    except subprocess.TimeoutExpired:
        logging.error("⏰ Timeout: La carga tomó más de 1 hora")
    except Exception as e:
        logging.error(f"💥 Error ejecutando uploader: {e}")


def run_single_scraper(scraper_cmd, scraper_id):
    """Ejecutar un scraper individual"""
    try:
        logging.info(f"🕷️ Iniciando scraper {scraper_id}...")

        result = subprocess.run(
            scraper_cmd,
            capture_output=True,
            text=True,
            timeout=7200  # Timeout de 2 horas
        )

        if result.returncode == 0:
            logging.info(f"✅ Scraper {scraper_id} completado exitosamente")
            logging.info(f"Output scraper {scraper_id}: {result.stdout}")
        else:
            logging.error(f"❌ Error en scraper {scraper_id}: {result.stderr}")

    except subprocess.TimeoutExpired:
        logging.error(f"⏰ Timeout: Scraper {scraper_id} tomó más de 2 horas")
    except Exception as e:
        logging.error(f"💥 Error ejecutando scraper {scraper_id}: {e}")


def run_scrapers():
    """Ejecutar los scrapers de datos en paralelo"""
    scrapers = [
        # Scrapers de datos oficiales (actualizados)
        ['python', 'rag/document_tools/scraper_dato.py', '--url',
         'https://www.juntadeandalucia.es/datosabiertos/portal.html',
         '--filtro', 'juntadeandalucia.es', '--directorio', './rag_document_data/datos/junta_andalucia/',
         '--enlaces', '1000', '--depth', '20'],
        ['python', 'rag/document_tools/scraper_dato.py', '--url',
         'https://www.juntadeandalucia.es/organismos/ieca/buscar.html',
         '--filtro', 'juntadeandalucia.es', '--directorio', './rag_document_data/datos/junta_andalucia/',
         '--enlaces', '1000', '--depth', '20'],

        # Scrapers de noticias (nuevos)
        ['python', 'rag/document_tools/scraper_diarios.py', '--url', 'https://www.abc.es/sevilla/',
         '--filtro', 'sevilla', '--directorio', './rag_document_data/noticias/',
         '--enlaces', '500', '--depth', '20'],
        ['python', 'rag/document_tools/scraper_diarios.py', '--url', 'https://www.diariosur.es/',
         '--filtro', 'diariosur.es', '--directorio', './rag_document_data/noticias/',
         '--enlaces', '500', '--depth', '20'],
        ['python', 'rag/document_tools/scraper_diarios.py', '--url', 'https://www.diariodesevilla.es',
         '--filtro', 'diariodesevilla.es', '--directorio', './rag_document_data/noticias/',
         '--enlaces', '500', '--depth', '20']
    ]

    logging.info(f"🚀 Iniciando {len(scrapers)} scrapers en paralelo...")

    # Crear threads para ejecutar scrapers en paralelo
    threads = []
    for i, scraper_cmd in enumerate(scrapers, 1):
        thread = threading.Thread(
            target=run_single_scraper,
            args=(scraper_cmd, i)
        )
        threads.append(thread)
        thread.start()

    # Esperar a que todos los threads terminen
    for thread in threads:
        thread.join()

    logging.info("🏁 Todos los scrapers han terminado")


def main():
    """Función principal del scheduler"""
    logging.info("📅 Scheduler iniciado")
    logging.info("   📤 Uploader: cada 4 horas")
    logging.info("   🕷️ Scrapers: diario a las 21:00")

    # Programar uploader cada 4 horas
    schedule.every(4).hours.do(run_uploader)

    # Programar scrapers diarios a las 21:00
    schedule.every().day.at("21:00").do(run_scrapers)

    # Ejecutar uploader inmediatamente al inicio
    logging.info("🔄 Ejecutando primera carga de uploader...")
    run_uploader()

    # Loop principal
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Verificar cada minuto
        except KeyboardInterrupt:
            logging.info("⚠️ Scheduler detenido por el usuario")
            break
        except Exception as e:
            logging.error(f"💥 Error en scheduler: {e}")
            time.sleep(300)  # Esperar 5 minutos antes de continuar


if __name__ == "__main__":
    main()