"""Simplifies ElevenLabs conversation transcripts to a minimal format.

Strips verbose metadata from each transcript entry, keeping only the
speaker role and message text, and computes basic conversation statistics.
"""

import json
from datetime import datetime
from typing import List, Dict, Any


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


def process_elevenlabs_conversations(input_file: str, output_file: str = None) -> Dict[str, Any]:
    """
    Procesa el archivo JSON de conversaciones de ElevenLabs y simplifica los diálogos

    Args:
        input_file: Ruta al archivo JSON de entrada
        output_file: Ruta opcional para guardar el resultado simplificado

    Returns:
        Diccionario con las conversaciones simplificadas
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


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    INPUT_FILE = "data/8G3toaZ2Tyu7HlUaWRO8_2025-08-27.json"  # Tu archivo descargado
    OUTPUT_FILE = "data/8G3toaZ2Tyu7HlUaWRO8_2025-08-25_simplified.json"  # Archivo de salida

    print("🤖 Procesador de Conversaciones ElevenLabs")
    print("=" * 50)

    # Procesar conversaciones
    result = process_elevenlabs_conversations(INPUT_FILE, OUTPUT_FILE)

    if result:
        # Mostrar estadísticas
        conversations = result.get("conversations", [])
        stats = get_conversation_statistics(conversations)

        print(f"\n📊 Estadísticas:")
        print(f"   - Total conversaciones: {stats.get('total_conversations', 0)}")
        print(f"   - Total mensajes: {stats.get('total_messages', 0)}")
        print(f"   - Promedio mensajes por conversación: {stats.get('average_messages_per_conversation', 0)}")
        print(f"   - Min/Max mensajes: {stats.get('min_messages', 0)}/{stats.get('max_messages', 0)}")

        print(f"\n📈 Distribución por status:")
        for status, count in stats.get('status_distribution', {}).items():
            print(f"   - {status}: {count}")

        # Mostrar muestra de conversación
        print_conversation_sample(conversations, conv_index=0, max_messages=8)

        print(f"\n✅ Procesamiento completado exitosamente!")
        print(f"📁 Archivo simplificado: {OUTPUT_FILE}")

    else:
        print("❌ Error en el procesamiento")