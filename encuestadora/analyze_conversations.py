"""LLM-powered analysis of ElevenLabs survey conversations.

Reads simplified conversation transcripts, uses an LLM to extract user
responses to predefined survey questions, and generates per-question
synthesis reports suitable for pollsters.
"""

import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Cargar variables de entorno
load_dotenv()


class ConversationAnalyzer:
    """Analizador de conversaciones usando LLM para extraer respuestas específicas"""

    def __init__(self):
        """Initialize the LLM client and define the target survey questions."""
        # Configurar LLM
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("VLLM_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VLLM_API_KEY") or "EMPTY"
        chat_model = os.getenv("VLLM_MODEL", "llama-3-8b-instruct")

        self.llm = ChatOpenAI(
            model=chat_model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.2
        )

        # Definir las preguntas objetivo
        self.questions = {
            "pregunta_1": {
                "tema": "problemas_acceso_vivienda",
                "pregunta": "Puede enumerar todos los problemas que considera que existen respecto al acceso actual a la vivienda. Puede mencionar tantos como quiera.",
                "prompt": """Analiza la siguiente conversación y extrae la respuesta u opinión del USUARIO sobre los problemas de acceso a la vivienda.

INSTRUCCIONES:
- Solo extrae lo que dice el USUARIO, no el agente
- Si el usuario menciona múltiples problemas, inclúyelos todos
- Si no hay respuesta clara, devuelve "No se encontró respuesta específica"
- Mantén las palabras exactas del usuario cuando sea posible

CONVERSACIÓN:
{dialog}

RESPUESTA DEL USUARIO SOBRE PROBLEMAS DE ACCESO A LA VIVIENDA:"""
            },

            "pregunta_2": {
                "tema": "viviendas_turisticas_andalucia",
                "pregunta": "¿Cuál es su opinión respecto a la controversia actual en Andalucía respecto a las viviendas turísticas?",
                "prompt": """Analiza la siguiente conversación y extrae la respuesta u opinión del USUARIO sobre las viviendas turísticas en Andalucía.

INSTRUCCIONES:
- Solo extrae lo que dice el USUARIO, no el agente
- Busca opiniones sobre viviendas turísticas, Airbnb, regulación, etc.
- Si no hay respuesta clara, devuelve "No se encontró respuesta específica"
- Mantén las palabras exactas del usuario cuando sea posible

CONVERSACIÓN:
{dialog}

RESPUESTA DEL USUARIO SOBRE VIVIENDAS TURÍSTICAS EN ANDALUCÍA:"""
            },

            "pregunta_3": {
                "tema": "digitalizacion_andalucia",
                "pregunta": "En una escala del 1 al 10, donde 1 es muy deficiente y 10 es excelente, ¿cuál es el actual nivel que considera que tiene Andalucía respecto a la digitalización y por qué?",
                "prompt": """Analiza la siguiente conversación y extrae la respuesta u opinión del USUARIO sobre el nivel de digitalización en Andalucía.

INSTRUCCIONES:
- Solo extrae lo que dice el USUARIO, no el agente
- Busca calificaciones numéricas (1-10) y razones
- Si no hay respuesta clara, devuelve "No se encontró respuesta específica"
- Mantén las palabras exactas del usuario cuando sea posible

CONVERSACIÓN:
{dialog}

RESPUESTA DEL USUARIO SOBRE DIGITALIZACIÓN EN ANDALUCÍA:"""
            }
        }

    def format_dialog_for_analysis(self, dialog: List[Dict[str, str]]) -> str:
        """Convierte el diálogo a formato texto para análisis"""
        formatted_lines = []
        for message in dialog:
            role = message.get("role", "unknown").upper()
            text = message.get("message", "")
            formatted_lines.append(f"{role}: {text}")
        return "\n".join(formatted_lines)

    def extract_user_response(self, dialog: List[Dict[str, str]], question_key: str) -> Dict[str, str]:
        """
        Extrae la respuesta del usuario para una pregunta específica

        Args:
            dialog: Lista de mensajes del diálogo
            question_key: Clave de la pregunta (pregunta_1, pregunta_2, pregunta_3)

        Returns:
            Diccionario con la respuesta extraída
        """
        if question_key not in self.questions:
            return {"error": f"Pregunta {question_key} no encontrada"}

        question_data = self.questions[question_key]
        formatted_dialog = self.format_dialog_for_analysis(dialog)

        # Crear prompt
        prompt = question_data["prompt"].format(dialog=formatted_dialog)

        try:
            # Hacer inferencia con el LLM
            messages = [
                SystemMessage(
                    content="Eres un asistente experto en análisis de conversaciones. Extrae información específica de manera precisa y concisa."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            extracted_response = response.content.strip()

            return {
                "tema": question_data["tema"],
                "pregunta": question_data["pregunta"],
                "respuesta_extraida": extracted_response,
                "status": "success"
            }

        except Exception as e:
            return {
                "tema": question_data["tema"],
                "pregunta": question_data["pregunta"],
                "respuesta_extraida": f"Error en extracción: {str(e)}",
                "status": "error"
            }

    def analyze_conversation(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza una conversación completa extrayendo respuestas a todas las preguntas

        Args:
            conversation: Conversación simplificada

        Returns:
            Conversación con respuestas extraídas
        """
        dialog = conversation.get("dialog", [])

        if not dialog:
            return {
                **conversation,
                "analysis": {
                    "pregunta_1": {"error": "No hay diálogo para analizar"},
                    "pregunta_2": {"error": "No hay diálogo para analizar"},
                    "pregunta_3": {"error": "No hay diálogo para analizar"}
                }
            }

        # Extraer respuestas para cada pregunta
        analysis = {}
        for question_key in self.questions.keys():
            print(f"  🔍 Analizando {question_key}...")
            analysis[question_key] = self.extract_user_response(dialog, question_key)

        return {
            **conversation,
            "analysis": analysis
        }

    def synthesize_responses_by_question(self, conversations: List[Dict[str, Any]], question_key: str) -> Dict[
        str, str]:
        """
        Sintetiza todas las respuestas de una pregunta específica usando el LLM

        Args:
            conversations: Lista de conversaciones analizadas
            question_key: Clave de la pregunta (pregunta_1, pregunta_2, pregunta_3)

        Returns:
            Diccionario con la síntesis generada
        """
        if question_key not in self.questions:
            return {"error": f"Pregunta {question_key} no encontrada"}

        question_data = self.questions[question_key]

        # Recopilar todas las respuestas válidas para esta pregunta
        valid_responses = []

        for conversation in conversations:
            analysis = conversation.get("analysis", {})
            if question_key in analysis:
                result = analysis[question_key]
                respuesta = result.get("respuesta_extraida", "")

                if (respuesta and
                        respuesta != "No se encontró respuesta específica" and
                        not respuesta.startswith("Error")):
                    conv_id = conversation.get("conversation_id", "unknown")
                    valid_responses.append(f"[{conv_id}] {respuesta}")

        if not valid_responses:
            return {
                "tema": question_data["tema"],
                "pregunta": question_data["pregunta"],
                "sintesis": "No se encontraron respuestas válidas para sintetizar",
                "total_respuestas": 0,
                "status": "no_data"
            }

        # Crear prompt para síntesis
        responses_text = "\n\n".join(valid_responses)

        synthesis_prompt = f"""Como encuestador profesional, necesitas sintetizar las siguientes respuestas de los usuarios sobre el tema: {question_data['tema']}.

PREGUNTA REALIZADA:
{question_data['pregunta']}

RESPUESTAS DE LOS USUARIOS:
{responses_text}

INSTRUCCIONES PARA LA SÍNTESIS:
- Identifica los temas y patrones principales mencionados por los usuarios
- Agrupa opiniones similares y destaca las diferencias
- Presenta las perspectivas más comunes y las minoritarias
- Usa un lenguaje profesional pero accesible
- Mantén un tono neutro y objetivo
- No excedas 300 palabras
- Incluye datos cuantitativos cuando sea relevante (ej: "X de Y usuarios mencionaron...")

SÍNTESIS PARA EL ENCUESTADOR:"""

        try:
            # Hacer inferencia con el LLM
            messages = [
                SystemMessage(
                    content="Eres un analista experto en encuestas y síntesis de opiniones públicas. Tu objetivo es proporcionar resúmenes claros y útiles para encuestadores."),
                HumanMessage(content=synthesis_prompt)
            ]

            response = self.llm.invoke(messages)
            synthesis_text = response.content.strip()

            return {
                "tema": question_data["tema"],
                "pregunta": question_data["pregunta"],
                "sintesis": synthesis_text,
                "total_respuestas": len(valid_responses),
                "status": "success"
            }

        except Exception as e:
            return {
                "tema": question_data["tema"],
                "pregunta": question_data["pregunta"],
                "sintesis": f"Error en síntesis: {str(e)}",
                "total_respuestas": len(valid_responses),
                "status": "error"
            }

    def generate_survey_synthesis(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Genera síntesis completa para todas las preguntas de la encuesta

        Args:
            conversations: Lista de conversaciones analizadas

        Returns:
            Lista con síntesis por pregunta
        """
        synthesis_results = []

        for question_key in self.questions.keys():
            print(f"📄 Sintetizando respuestas para {question_key}...")
            synthesis = self.synthesize_responses_by_question(conversations, question_key)
            synthesis_results.append(synthesis)

            if synthesis.get("status") == "success":
                print(f"✅ Síntesis completada: {synthesis['total_respuestas']} respuestas procesadas")
            else:
                print(f"⚠️  {synthesis.get('sintesis', 'Error en síntesis')}")

        return synthesis_results


def simplify_conversation_transcript(transcript: List[Dict]) -> List[Dict[str, str]]:
    """
    Simplifica el transcript de una conversación a solo role y message

    Args:
        transcript: Lista de elementos del transcript complejo

    Returns:
        Lista simplificada con solo role y message
    """
    simplified = []

    for item in transcript:
        simplified_item = {
            "role": item.get("role", "unknown"),
            "message": item.get("message", "")
        }
        simplified.append(simplified_item)

    return simplified


def process_elevenlabs_conversations(input_file: str, output_file: str = None, analyze_responses: bool = True) -> Dict[
    str, Any]:
    """
    Procesa el archivo JSON de conversaciones de ElevenLabs y simplifica los diálogos

    Args:
        input_file: Ruta al archivo JSON de entrada
        output_file: Ruta opcional para guardar el resultado simplificado
        analyze_responses: Si True, analiza las respuestas con LLM

    Returns:
        Diccionario con las conversaciones simplificadas y analizadas
    """
    # Leer el archivo JSON de entrada
    print(f"📖 Leyendo archivo: {input_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {input_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Error al decodificar JSON: {e}")
        return {}

    # Extraer información básica
    agent_id = data.get("agent_id", "unknown")
    filter_date = data.get("filter_date", "unknown")
    total_conversations = data.get("total_conversations", 0)
    extraction_date = data.get("extraction_date", "unknown")
    conversations = data.get("conversations", [])

    print(f"📊 Información del archivo:")
    print(f"   - Agent ID: {agent_id}")
    print(f"   - Fecha filtro: {filter_date}")
    print(f"   - Total conversaciones: {total_conversations}")
    print(f"   - Fecha extracción: {extraction_date}")

    # Inicializar analizador si se requiere análisis
    analyzer = None
    if analyze_responses:
        print(f"🤖 Inicializando analizador LLM...")
        try:
            analyzer = ConversationAnalyzer()
            print(f"✅ LLM configurado correctamente")
        except Exception as e:
            print(f"❌ Error configurando LLM: {e}")
            analyze_responses = False

    # Procesar cada conversación
    simplified_conversations = []

    for i, conversation in enumerate(conversations):
        conversation_id = conversation.get("conversation_id", f"conv_{i}")
        status = conversation.get("status", "unknown")
        transcript = conversation.get("transcript", [])

        # Simplificar el transcript
        simplified_transcript = simplify_conversation_transcript(transcript)

        # Crear conversación simplificada
        simplified_conv = {
            "conversation_id": conversation_id,
            "status": status,
            "dialog": simplified_transcript
        }

        # Analizar con LLM si está habilitado
        if analyze_responses and analyzer:
            print(f"🔍 Analizando conversación {i + 1}/{len(conversations)}: {conversation_id}")
            simplified_conv = analyzer.analyze_conversation(simplified_conv)

        simplified_conversations.append(simplified_conv)
        print(
            f"✅ Procesada conversación {i + 1}/{len(conversations)}: {conversation_id} ({len(simplified_transcript)} mensajes)")

    # Crear resultado final
    result = {
        "agent_id": agent_id,
        "filter_date": filter_date,
        "total_conversations": len(simplified_conversations),
        "extraction_date": extraction_date,
        "processing_date": datetime.now().isoformat(),
        "llm_analysis_enabled": analyze_responses,
        "conversations": simplified_conversations
    }

    # Guardar resultado si se especifica archivo de salida
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"💾 Resultado guardado en: {output_file}")

    return result


def print_conversation_sample(conversations: List[Dict], conv_index: int = 0, max_messages: int = 5):
    """
    Imprime una muestra de una conversación simplificada

    Args:
        conversations: Lista de conversaciones simplificadas
        conv_index: Índice de la conversación a mostrar
        max_messages: Número máximo de mensajes a mostrar
    """
    if not conversations or conv_index >= len(conversations):
        print("❌ No hay conversaciones para mostrar")
        return

    conversation = conversations[conv_index]
    dialog = conversation.get("dialog", [])

    print(f"\n📋 Muestra de conversación {conv_index + 1}:")
    print(f"   ID: {conversation.get('conversation_id')}")
    print(f"   Status: {conversation.get('status')}")
    print(f"   Mensajes: {len(dialog)}")
    print(f"\n--- DIÁLOGO ---")

    for i, message in enumerate(dialog[:max_messages]):
        role = message.get("role", "unknown")
        text = message.get("message", "")
        print(f"{role.upper()}: {text}")

        if i < len(dialog[:max_messages]) - 1:
            print()

    if len(dialog) > max_messages:
        print(f"\n... (y {len(dialog) - max_messages} mensajes más)")


def print_analysis_results(conversations: List[Dict], max_conversations: int = 3):
    """
    Imprime los resultados del análisis LLM

    Args:
        conversations: Lista de conversaciones analizadas
        max_conversations: Número máximo de conversaciones a mostrar
    """
    analyzed_conversations = [conv for conv in conversations if "analysis" in conv]

    if not analyzed_conversations:
        print("❌ No hay conversaciones con análisis disponible")
        return

    print(f"\n🎯 RESULTADOS DEL ANÁLISIS LLM")
    print("=" * 60)

    for i, conversation in enumerate(analyzed_conversations[:max_conversations]):
        conv_id = conversation.get("conversation_id", f"conv_{i}")
        status = conversation.get("status", "unknown")
        analysis = conversation.get("analysis", {})

        print(f"\n📋 Conversación {i + 1}: {conv_id} (Status: {status})")
        print("-" * 40)

        for question_key, result in analysis.items():
            if "error" in result:
                print(f"❌ {question_key}: {result['error']}")
                continue

            tema = result.get("tema", "unknown")
            respuesta = result.get("respuesta_extraida", "No disponible")

            print(f"\n🔹 {question_key.upper()} ({tema}):")
            print(f"   Respuesta: {respuesta[:200]}{'...' if len(respuesta) > 200 else ''}")


def get_conversation_statistics(conversations: List[Dict]) -> Dict[str, Any]:
    """
    Genera estadísticas de las conversaciones simplificadas

    Args:
        conversations: Lista de conversaciones simplificadas

    Returns:
        Diccionario con estadísticas
    """
    if not conversations:
        return {}

    # Estadísticas de status
    status_count = {}
    total_messages = 0
    message_counts = []

    for conv in conversations:
        status = conv.get("status", "unknown")
        status_count[status] = status_count.get(status, 0) + 1

        dialog = conv.get("dialog", [])
        message_count = len(dialog)
        total_messages += message_count
        message_counts.append(message_count)

    stats = {
        "total_conversations": len(conversations),
        "status_distribution": status_count,
        "total_messages": total_messages,
        "average_messages_per_conversation": round(total_messages / len(conversations), 2) if conversations else 0,
        "min_messages": min(message_counts) if message_counts else 0,
        "max_messages": max(message_counts) if message_counts else 0
    }

    return stats


def print_conversation_sample(conversations: List[Dict], conv_index: int = 0, max_messages: int = 5):
    """
    Imprime una muestra de una conversación simplificada

    Args:
        conversations: Lista de conversaciones simplificadas
        conv_index: Índice de la conversación a mostrar
        max_messages: Número máximo de mensajes a mostrar
    """
    if not conversations or conv_index >= len(conversations):
        print("❌ No hay conversaciones para mostrar")
        return

    conversation = conversations[conv_index]
    dialog = conversation.get("dialog", [])

    print(f"\n📋 Muestra de conversación {conv_index + 1}:")
    print(f"   ID: {conversation.get('conversation_id')}")
    print(f"   Status: {conversation.get('status')}")
    print(f"   Mensajes: {len(dialog)}")
    print(f"\n--- DIÁLOGO ---")

    for i, message in enumerate(dialog[:max_messages]):
        role = message.get("role", "unknown")
        text = message.get("message", "")
        print(f"{role.upper()}: {text}")

        if i < len(dialog[:max_messages]) - 1:
            print()

    if len(dialog) > max_messages:
        print(f"\n... (y {len(dialog) - max_messages} mensajes más)")


def print_analysis_results(conversations: List[Dict], max_conversations: int = 3):
    """
    Crea un resumen del análisis de todas las conversaciones

    Args:
        conversations: Lista de conversaciones analizadas

    Returns:
        Diccionario con resumen del análisis
    """
    analyzed_conversations = [conv for conv in conversations if "analysis" in conv]

    if not analyzed_conversations:
        return {"error": "No hay conversaciones analizadas"}

    summary = {
        "total_analyzed": len(analyzed_conversations),
        "questions_summary": {
            "pregunta_1": {"responses_found": 0, "responses": []},
            "pregunta_2": {"responses_found": 0, "responses": []},
            "pregunta_3": {"responses_found": 0, "responses": []}
        }
    }

    for conversation in analyzed_conversations:
        analysis = conversation.get("analysis", {})

        for question_key, result in analysis.items():
            if question_key in summary["questions_summary"]:
                respuesta = result.get("respuesta_extraida", "")

                if (respuesta and
                        respuesta != "No se encontró respuesta específica" and
                        not respuesta.startswith("Error")):
                    summary["questions_summary"][question_key]["responses_found"] += 1
                    summary["questions_summary"][question_key]["responses"].append({
                        "conversation_id": conversation.get("conversation_id"),
                        "response": respuesta[:100] + "..." if len(respuesta) > 100 else respuesta
                    })

    return summary


def create_analysis_summary(conversations: List[Dict]) -> Dict[str, Any]:
    """
    Crea un resumen del análisis de todas las conversaciones

    Args:
        conversations: Lista de conversaciones analizadas

    Returns:
        Diccionario con resumen del análisis
    """
    analyzed_conversations = [conv for conv in conversations if "analysis" in conv]

    if not analyzed_conversations:
        return {"error": "No hay conversaciones analizadas"}

    summary = {
        "total_analyzed": len(analyzed_conversations),
        "questions_summary": {
            "pregunta_1": {"responses_found": 0, "responses": []},
            "pregunta_2": {"responses_found": 0, "responses": []},
            "pregunta_3": {"responses_found": 0, "responses": []}
        }
    }

    for conversation in analyzed_conversations:
        analysis = conversation.get("analysis", {})

        for question_key, result in analysis.items():
            if question_key in summary["questions_summary"]:
                respuesta = result.get("respuesta_extraida", "")

                if (respuesta and
                        respuesta != "No se encontró respuesta específica" and
                        not respuesta.startswith("Error")):
                    summary["questions_summary"][question_key]["responses_found"] += 1
                    summary["questions_summary"][question_key]["responses"].append({
                        "conversation_id": conversation.get("conversation_id"),
                        "response": respuesta[:100] + "..." if len(respuesta) > 100 else respuesta
                    })

    return summary


def save_survey_synthesis(synthesis_results: List[Dict[str, str]], output_file: str = "data/analisis_encuesta.json"):
    """
    Guarda la síntesis de la encuesta en un archivo JSON

    Args:
        synthesis_results: Lista con síntesis por pregunta
        output_file: Nombre del archivo de salida
    """
    try:
        # Preparar datos para guardar
        final_synthesis = []

        for synthesis in synthesis_results:
            final_synthesis.append({
                "tema": synthesis.get("tema", "unknown"),
                "pregunta": synthesis.get("pregunta", ""),
                "sintesis": synthesis.get("sintesis", "No disponible")
            })

        # Guardar en archivo JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_synthesis, f, indent=2, ensure_ascii=False)

        print(f"💾 Síntesis de encuesta guardada en: {output_file}")
        return True

    except Exception as e:
        print(f"❌ Error guardando síntesis: {e}")
        return False


def copy_base_analysis_if_no_data(base_file: str = "data/analisis_encuesta_base.json",
                                  output_file: str = "data/analisis_encuesta.json"):
    """
    Copia el archivo base de análisis si no hay datos para procesar

    Args:
        base_file: Archivo base de análisis
        output_file: Archivo de salida
    """
    try:
        if os.path.exists(base_file):
            shutil.copy2(base_file, output_file)
            print(f"📋 Copiado archivo base: {base_file} → {output_file}")
            return True
        else:
            print(f"⚠️  No se encontró el archivo base: {base_file}")
            return False
    except Exception as e:
        print(f"❌ Error copiando archivo base: {e}")
        return False


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    INPUT_FILE = "data/8G3toaZ2Tyu7HlUaWRO8_2025-08-27.json"  # Tu archivo descargado
    OUTPUT_FILE = "data/8G3toaZ2Tyu7HlUaWRO8_2025-08-25_analyzed.json"  # Archivo de salida
    SYNTHESIS_FILE = "data/analisis_encuesta.json"  # Archivo de síntesis final
    BASE_ANALYSIS_FILE = "data/analisis_encuesta_base.json"  # Archivo base para copiar si no hay datos
    ANALYZE_WITH_LLM = True  # Cambiar a False si no quieres análisis LLM
    GENERATE_SYNTHESIS = True  # Cambiar a False si no quieres síntesis final

    print("🤖 Procesador y Analizador de Conversaciones ElevenLabs")
    print("=" * 60)
    print(f"📁 Archivo de entrada: {INPUT_FILE}")
    print(f"📁 Archivo de salida: {OUTPUT_FILE}")
    print(f"📁 Archivo de síntesis: {SYNTHESIS_FILE}")
    print(f"📁 Archivo base: {BASE_ANALYSIS_FILE}")
    print(f"🧠 Análisis LLM: {'✅ Habilitado' if ANALYZE_WITH_LLM else '❌ Deshabilitado'}")
    print(f"🎯 Síntesis final: {'✅ Habilitada' if GENERATE_SYNTHESIS else '❌ Deshabilitada'}")
    print("-" * 60)

    # Procesar conversaciones
    result = process_elevenlabs_conversations(
        INPUT_FILE,
        OUTPUT_FILE,
        analyze_responses=ANALYZE_WITH_LLM
    )

    if result:
        conversations = result.get("conversations", [])

        # Verificar si hay conversaciones válidas para procesar
        valid_conversations = [conv for conv in conversations if conv.get("dialog")]

        if not valid_conversations:
            print(f"\n⚠️  No se encontraron conversaciones válidas para procesar")
            print(f"📋 Copiando archivo base de análisis...")
            copy_base_analysis_if_no_data(BASE_ANALYSIS_FILE, SYNTHESIS_FILE)
            print(f"✅ PROCESAMIENTO COMPLETADO - Usando análisis base")
        else:
            # Mostrar estadísticas básicas
            stats = get_conversation_statistics(conversations)

            print(f"\n📊 ESTADÍSTICAS BÁSICAS:")
            print(f"   - Total conversaciones: {stats.get('total_conversations', 0)}")
            print(f"   - Conversaciones válidas: {len(valid_conversations)}")
            print(f"   - Total mensajes: {stats.get('total_messages', 0)}")
            print(f"   - Promedio mensajes por conversación: {stats.get('average_messages_per_conversation', 0)}")
            print(f"   - Min/Max mensajes: {stats.get('min_messages', 0)}/{stats.get('max_messages', 0)}")

            print(f"\n📈 Distribución por status:")
            for status, count in stats.get('status_distribution', {}).items():
                print(f"   - {status}: {count}")

            # Si hay análisis LLM, mostrar resultados
            if ANALYZE_WITH_LLM and result.get("llm_analysis_enabled"):
                print_analysis_results(conversations, max_conversations=5)

                # Crear y mostrar resumen del análisis
                analysis_summary = create_analysis_summary(conversations)

                if "error" not in analysis_summary:
                    print(f"\n🎯 RESUMEN DEL ANÁLISIS:")
                    print(f"   - Conversaciones analizadas: {analysis_summary['total_analyzed']}")

                    # Verificar si hay respuestas válidas para síntesis
                    total_valid_responses = 0
                    for question_key, question_data in analysis_summary['questions_summary'].items():
                        responses_found = question_data['responses_found']
                        total = analysis_summary['total_analyzed']
                        percentage = (responses_found / total * 100) if total > 0 else 0
                        total_valid_responses += responses_found

                        print(
                            f"   - {question_key}: {responses_found}/{total} respuestas encontradas ({percentage:.1f}%)")

                    # Generar síntesis final si está habilitada y hay respuestas válidas
                    if GENERATE_SYNTHESIS:
                        if total_valid_responses > 0:
                            print(f"\n🎯 GENERANDO SÍNTESIS FINAL...")
                            print("=" * 60)

                            # Usar el mismo analyzer del proceso principal
                            try:
                                analyzer = ConversationAnalyzer()
                                synthesis_results = analyzer.generate_survey_synthesis(conversations)

                                # Guardar síntesis en archivo
                                if save_survey_synthesis(synthesis_results, SYNTHESIS_FILE):
                                    print(f"\n✅ SÍNTESIS COMPLETADA Y GUARDADA!")
                                    print(f"📁 Archivo de síntesis: {SYNTHESIS_FILE}")
                                else:
                                    print(f"\n❌ Error guardando síntesis en archivo")

                            except Exception as e:
                                print(f"❌ Error generando síntesis: {e}")
                                print(f"📋 Copiando archivo base de análisis...")
                                copy_base_analysis_if_no_data(BASE_ANALYSIS_FILE, SYNTHESIS_FILE)
                        else:
                            print(f"\n⚠️  No se encontraron respuestas válidas para síntesis")
                            print(f"📋 Copiando archivo base de análisis...")
                            copy_base_analysis_if_no_data(BASE_ANALYSIS_FILE, SYNTHESIS_FILE)
                else:
                    print(f"❌ Error en análisis: {analysis_summary['error']}")
                    if GENERATE_SYNTHESIS:
                        print(f"📋 Copiando archivo base de análisis...")
                        copy_base_analysis_if_no_data(BASE_ANALYSIS_FILE, SYNTHESIS_FILE)

            elif GENERATE_SYNTHESIS:
                # Si no hay análisis LLM pero se quiere síntesis, usar archivo base
                print(f"📋 Sin análisis LLM disponible, copiando archivo base...")
                copy_base_analysis_if_no_data(BASE_ANALYSIS_FILE, SYNTHESIS_FILE)

            if ANALYZE_WITH_LLM:
                print(f"\n🧠 Las respuestas extraídas por el LLM están disponibles en: {OUTPUT_FILE}")
                if GENERATE_SYNTHESIS:
                    print(f"🎯 La síntesis final está disponible en: {SYNTHESIS_FILE}")

            # Mostrar muestra de conversación
            print(f"\n" + "=" * 60)
            print_conversation_sample(conversations, conv_index=0, max_messages=6)

            print(f"\n✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
            print(f"📁 Archivo generado: {OUTPUT_FILE}")
            if GENERATE_SYNTHESIS:
                print(f"📁 Síntesis generada: {SYNTHESIS_FILE}")

    else:
        print("❌ Error en el procesamiento")
        if GENERATE_SYNTHESIS:
            print(f"📋 Copiando archivo base de análisis...")
            copy_base_analysis_if_no_data(BASE_ANALYSIS_FILE, SYNTHESIS_FILE)