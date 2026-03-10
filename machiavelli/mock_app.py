"""Mock version of the Machiavelli API for testing and frontend development.

Returns hardcoded responses for all endpoints (fact-checking, RAG queries,
surveys, audio streaming) without requiring real backend services like
Milvus, LLMs, or ElevenLabs.
"""

from flask import Flask, request, Response, jsonify
from flask_socketio import SocketIO, emit, disconnect
import json
import time
import uuid
from datetime import datetime
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mock_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory storage
jobs = {}
sessions = {}
audio_jobs = {}
encuestas = {}


# ============== FACT CHECKER ENDPOINTS ==============

@app.route('/api/v1/machiavelli/fact_checker', methods=['POST'])
def submit_fact_check():
    """Submit text for fact-checking and get a job_id"""
    data = request.get_json()
    user_text = data.get('user_text', '')

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'status': 'processing',
        'user_text': user_text,
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'result': None
    }

    # Simulate completion
    jobs[job_id]['status'] = 'completed'
    jobs[job_id]['completed_at'] = datetime.now().isoformat()
    jobs[job_id]['result'] = get_mock_fact_check_result(user_text)

    return jsonify({
        'job_id': job_id,
        'status': 'submitted',
        'message': 'Fact-checking job submitted successfully'
    })


@app.route('/api/v1/machiavelli/query', methods=['POST'])
def machiavelli_query():
    """Query processing with streaming response"""
    data = request.get_json()
    user_text = data.get('user_text', '')
    mode = data.get('mode', 'datos')

    # Validate mode
    valid_modes = ['preguntas_frecuentes', 'datos', 'medios', 'debates', 'encuestas']
    if mode not in valid_modes:
        return jsonify({'error': f'Invalid mode. Must be one of: {valid_modes}'}), 400

    def generate_query_stream():
        # Step 1: Searching
        yield f"data: {json.dumps(get_search_response())}\n\n"
        time.sleep(1.5)

        # Step 2: Analyzing
        yield f"data: {json.dumps(get_analyzing_response())}\n\n"
        time.sleep(2)

        # Step 3: Final answer
        yield f"data: {json.dumps(get_final_query_response(user_text, mode))}\n\n"

    return Response(
        generate_query_stream(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )


# ============== POLLING/SURVEY ENDPOINTS ==============

@app.route('/api/v1/validacion-intervencion', methods=['POST'])
def create_survey_intervention():
    """Create and validate new survey intervention"""
    data = request.get_json()

    encuesta_id = str(uuid.uuid4())
    encuestas[encuesta_id] = {
        'titulo': data.get('titulo', ''),
        'fecha': data.get('fecha', ''),
        'base_datos': data.get('base_datos', ''),
        'preguntas': data.get('preguntas', []),
        'numero_encuestados': 0,
        'created_at': datetime.now().isoformat()
    }

    return jsonify({
        'id': encuesta_id,
        'status': 'created',
        'numero_encuestados': 0,
        'fecha_creacion': datetime.now().isoformat(),
        'url_encuesta': f'https://encuesta.com/{encuesta_id}'
    })


@app.route('/api/v1/encuesta/<encuesta_id>/resultados', methods=['GET'])
def get_survey_results(encuesta_id):
    """Get survey results with optional filtering"""
    if encuesta_id not in encuestas:
        return jsonify({'error': 'Survey not found'}), 404

    # Mock filters (not actually used in mock)
    filtro_genero = request.args.get('filtro_genero', 'todos')
    filtro_edad = request.args.get('filtro_edad', 'cualquiera')

    return jsonify({
        'encuesta_id': encuesta_id,
        'titulo': 'Validacion de la intervencion (vivienda)',
        'numero_encuestados': 654,
        'fecha': '04/08/2025',
        'resultados': {
            'pregunta_1': {
                'texto': 'Valoracion de la intervencion',
                'respuestas': {
                    'sin_cambios': {'count': 412, 'porcentaje': 63.1},
                    'requiere_reformulacion': {'count': 106, 'porcentaje': 16.2},
                    'indiferente': {'count': 91, 'porcentaje': 13.9},
                    'prefiere_no_decirlo': {'count': 45, 'porcentaje': 6.8}
                }
            },
            'pregunta_2': {
                'texto': 'Refleja adecuadamente el argumentario del partido?',
                'nps_score': 41.3,
                'media_esperada': 35,
                'distribucion': {
                    'a_favor': {'porcentaje': 77.5, 'count': 507},
                    'parcialmente': {'porcentaje': 18.2, 'count': 119},
                    'no': {'porcentaje': 4.3, 'count': 28}
                }
            }
        }
    })


@app.route('/api/v1/encuesta/<encuesta_id>/insights', methods=['GET'])
def get_survey_insights(encuesta_id):
    """Get AI-generated insights about survey results"""
    if encuesta_id not in encuestas:
        return jsonify({'error': 'Survey not found'}), 404

    return jsonify({
        'insights_positivos': [
            {'texto': 'Se entiende, pero falta un golpe de efecto final.', 'score': 51, 'categoria': 'neutral'},
            {'texto': 'Correcta, aunque no deja huella.', 'score': 39, 'categoria': 'neutral'},
            {'texto': 'Funciona en contexto parlamentario, no tanto en medios.', 'score': 17, 'categoria': 'neutral'}
        ],
        'frases_debiles': [
            {'texto': 'No genera emocion ni orgullo de replica.', 'score': -31, 'categoria': 'negativo'},
            {'texto': 'Demasiado defensiva para ser creible.', 'score': -19, 'categoria': 'negativo'},
            {'texto': 'Parece hecha para no molestar, no para convencer.', 'score': -8, 'categoria': 'negativo'}
        ],
        'insights_gente_alta': [
            {'texto': 'Este discurso se puede defender en cualquier territorio.', 'count': 220,
             'sentimiento': 'positivo'},
            {'texto': 'Contundente sin sonar agresivo. Perfecto para el clima actual.', 'count': 196,
             'sentimiento': 'positivo'},
            {'texto': 'Es una intervencion que ordena el marco y genera autoridad.', 'count': 188,
             'sentimiento': 'positivo'}
        ]
    })


@app.route('/api/v1/chatbot/consulta', methods=['POST'])
def chatbot_query():
    """Chatbot queries about survey results"""
    data = request.get_json()
    encuesta_id = data.get('encuesta_id')
    pregunta = data.get('pregunta', '')

    if encuesta_id and encuesta_id not in encuestas:
        return jsonify({'error': 'Survey not found'}), 404

    return jsonify({
        'respuesta': 'Basandome en los resultados de la encuesta sobre la intervencion de vivienda, observo que hay una aceptacion mayoritaria (77.5% a favor) pero con un NPS de 41.3 que indica espacio de mejora. Las criticas se centran en la falta de impacto emocional...',
        'fuentes': ['pregunta_1_resultados', 'nps_score', 'insights_negativos'],
        'confianza': 0.85
    })


@app.route('/api/v1/encuesta/<encuesta_id>/segmentacion', methods=['GET'])
def get_survey_segmentation(encuesta_id):
    """Get demographic segmentation analysis"""
    if encuesta_id not in encuestas:
        return jsonify({'error': 'Survey not found'}), 404

    segmento = request.args.get('segmento', 'genero')

    segmentation_data = {
        'genero': {
            'masculino': {'count': 324, 'porcentaje_total': 49.5, 'nps_score': 43.1,
                          'pregunta_1_dominante': 'sin_cambios'},
            'femenino': {'count': 330, 'porcentaje_total': 50.5, 'nps_score': 39.7,
                         'pregunta_1_dominante': 'sin_cambios'}
        },
        'edad': {
            '18-25': {'count': 98, 'porcentaje_total': 15.0, 'nps_score': 38.2,
                      'pregunta_1_dominante': 'requiere_reformulacion'},
            '26-35': {'count': 156, 'porcentaje_total': 23.9, 'nps_score': 42.1, 'pregunta_1_dominante': 'sin_cambios'},
            '36-50': {'count': 245, 'porcentaje_total': 37.5, 'nps_score': 43.8, 'pregunta_1_dominante': 'sin_cambios'},
            '50+': {'count': 155, 'porcentaje_total': 23.6, 'nps_score': 40.5, 'pregunta_1_dominante': 'sin_cambios'}
        }
    }

    return jsonify({
        'segmento': segmento,
        'resultados': segmentation_data.get(segmento, segmentation_data['genero'])
    })

###



@app.route('/api/v1/machiavelli/fact_checker/<job_id>', methods=['GET'])
def get_fact_check_result(job_id):
    """Get fact-checking results via Server-Sent Events"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404

    job = jobs[job_id]
    if job['status'] != 'completed':
        return jsonify({'error': 'Job still processing'}), 202

    def generate_sse_response():
        time.sleep(1)
        data_line = f"data: {json.dumps(job['result'])}\n\n"
        yield data_line

    return Response(
        generate_sse_response(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )


# ============== AUDIO STREAMING ENDPOINTS ==============

@app.route('/api/v1/jobs', methods=['POST'])
def create_audio_job():
    """Create transcription job"""
    data = request.get_json() or {}
    session_id = data.get('session_id', str(uuid.uuid4()))
    config = data.get('config', {})

    job_id = str(uuid.uuid4())

    # Store session
    if session_id not in sessions:
        sessions[session_id] = {'jobs': [], 'created_at': datetime.now().isoformat()}

    audio_jobs[job_id] = {
        'session_id': session_id,
        'status': 'initialized',
        'config': config,
        'progress': {
            'transcription_status': 'pending',
            'llm_response_status': 'pending',
            'completion_percentage': 0
        },
        'results': {
            'transcribed_text': '',
            'llm_response': ''
        },
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }

    sessions[session_id]['jobs'].append(job_id)

    return jsonify({
        'job_id': job_id,
        'session_id': session_id,
        'status': 'initialized',
        'websocket_url': f'ws://localhost:5000/api/v1/stream/{job_id}',
        'created_at': audio_jobs[job_id]['created_at']
    })


@app.route('/api/v1/jobs/<job_id>', methods=['GET'])
def get_audio_job_status(job_id):
    """Get job status"""
    if job_id not in audio_jobs:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify(audio_jobs[job_id])


@app.route('/api/v1/jobs/<job_id>', methods=['DELETE'])
def cancel_audio_job(job_id):
    """Cancel job"""
    if job_id not in audio_jobs:
        return jsonify({'error': 'Job not found'}), 404

    audio_jobs[job_id]['status'] = 'cancelled'
    audio_jobs[job_id]['updated_at'] = datetime.now().isoformat()

    return jsonify({
        'job_id': job_id,
        'status': 'cancelled',
        'message': 'Job cancelled successfully'
    })


@app.route('/api/v1/sessions/<session_id>/jobs', methods=['GET'])
def list_session_jobs(session_id):
    """List jobs for session"""
    if session_id not in sessions:
        return jsonify({'error': 'Session not found'}), 404

    session_jobs = []
    for job_id in sessions[session_id]['jobs']:
        if job_id in audio_jobs:
            job = audio_jobs[job_id]
            session_jobs.append({
                'job_id': job_id,
                'status': job['status'],
                'created_at': job['created_at']
            })

    return jsonify({
        'session_id': session_id,
        'jobs': session_jobs,
        'total': len(session_jobs)
    })


# ============== WEBSOCKET HANDLERS ==============

@socketio.on('connect')
def handle_connect():
    """Log WebSocket connection and send confirmation to client."""
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to audio streaming service'})


@socketio.on('disconnect')
def handle_disconnect():
    """Log WebSocket disconnection."""
    print(f'Client disconnected: {request.sid}')


@socketio.on('start_recording')
def handle_start_recording(data):
    """Mark an audio job as recording and notify the client."""
    job_id = data.get('job_id')
    if job_id not in audio_jobs:
        emit('error', {'message': 'Job not found'})
        return

    audio_jobs[job_id]['status'] = 'recording'
    audio_jobs[job_id]['updated_at'] = datetime.now().isoformat()

    emit('job_status_changed', {
        'job_id': job_id,
        'status': 'recording',
        'message': 'Recording started'
    })


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Simulate processing an incoming audio chunk and emit a partial transcription."""
    job_id = data.get('job_id')
    # audio_data = data.get('audio_data')  # Base64 encoded audio

    if job_id not in audio_jobs:
        emit('error', {'message': 'Job not found'})
        return

    # Simulate transcription update
    mock_partial_text = "Transcribing audio... partial text here..."
    emit('transcription_update', {
        'job_id': job_id,
        'partial_text': mock_partial_text,
        'confidence': 0.85
    })


@socketio.on('stop_recording')
def handle_stop_recording(data):
    """Stop recording, simulate transcription and LLM fact-check in a background thread."""
    job_id = data.get('job_id')
    if job_id not in audio_jobs:
        emit('error', {'message': 'Job not found'})
        return

    # Start processing simulation
    def process_audio_job(job_id):
        time.sleep(1)  # Simulate transcription processing

        # Complete transcription
        transcribed_text = "El 20.4% de la poblacion esta en riesgo de pobreza segun el INE."
        audio_jobs[job_id]['results']['transcribed_text'] = transcribed_text
        audio_jobs[job_id]['status'] = 'transcribing'
        audio_jobs[job_id]['progress']['transcription_status'] = 'completed'
        audio_jobs[job_id]['progress']['completion_percentage'] = 50

        socketio.emit('transcription_complete', {
            'job_id': job_id,
            'transcribed_text': transcribed_text,
            'confidence': 0.92
        })

        # Start LLM response
        audio_jobs[job_id]['status'] = 'generating_response'
        audio_jobs[job_id]['progress']['llm_response_status'] = 'streaming'

        socketio.emit('llm_response_start', {
            'job_id': job_id,
            'message': 'Starting LLM response generation'
        })

        # Stream LLM response (same as fact checker)
        fact_check_result = get_mock_fact_check_result(transcribed_text)
        response_chunks = [
            "Los datos mas actualizados del INE ",
            "indican que la tasa de riesgo de pobreza ",
            "es del 20.7%, no del 20.4% como se afirma."
        ]

        for i, chunk in enumerate(response_chunks):
            time.sleep(0.5)
            socketio.emit('llm_response_chunk', {
                'job_id': job_id,
                'chunk': chunk,
                'chunk_index': i
            })

        # Complete response
        full_response = ''.join(response_chunks)
        audio_jobs[job_id]['results']['llm_response'] = full_response
        audio_jobs[job_id]['status'] = 'completed'
        audio_jobs[job_id]['progress']['llm_response_status'] = 'completed'
        audio_jobs[job_id]['progress']['completion_percentage'] = 100
        audio_jobs[job_id]['updated_at'] = datetime.now().isoformat()

        socketio.emit('llm_response_complete', {
            'job_id': job_id,
            'full_response': full_response,
            'fact_check_result': fact_check_result
        })

        socketio.emit('job_status_changed', {
            'job_id': job_id,
            'status': 'completed',
            'message': 'Job completed successfully'
        })

    # Start processing in background thread
    thread = threading.Thread(target=process_audio_job, args=(job_id,))
    thread.daemon = True
    thread.start()

    emit('job_status_changed', {
        'job_id': job_id,
        'status': 'processing',
        'message': 'Processing audio...'
    })


# ============== HELPER FUNCTIONS ==============

def get_mock_fact_check_result(user_text):
    """Generate mock fact-checking result"""
    return {
        "code": 0,
        "data": {
            "ideas_de_fuerza": {
                "data": [
                    {
                        "claim": "El 20.4% de la poblacion esta en riesgo de pobreza segun el INE",
                        "incoherencia_detectada": "Se afirma que el 20.4% de la poblacion esta en riesgo de pobreza, pero los datos mas recientes del INE muestran un 20.7%",
                        "resultado_verificado": "Segun el ultimo informe del INE de 2024, la tasa de riesgo de pobreza es del 20.7%, no del 20.4%",
                        "respuesta_sugerida": "Los datos mas actualizados del INE indican que la tasa de riesgo de pobreza es del 20.7%",
                        "referencias": [
                            {
                                "fuente": "Fuente 1",
                                "id": 1,
                                "title": "Encuesta de Condiciones de Vida 2024 - INE",
                                "url": "https://ine.es/encuesta-condiciones-vida-2024"
                            }
                        ]
                    }
                ],
                "score_line": []
            },
            "discurso": {
                "text": user_text,
                "color": [-0.2, -0.2, -0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "claim_mask": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            },
            "Titulares": {
                "Afirmaciones Criticas": [
                    {
                        "Titulo": "Tasa de pobreza INE",
                        "respuesta_sugerida": "Los datos mas actualizados del INE indican que la tasa de riesgo de pobreza es del 20.7%"
                    }
                ],
                "Afirmaciones Ambiguas": []
            }
        },
        "error": "",
        "is_done": True
    }


def get_search_response():
    """Generate initial search response"""
    return {
        "code": 0,
        "data": {
            "interim_answer": "Buscando informacion en fuentes...",
            "final_answer": "",
            "fuentes": []
        },
        "error": "",
        "is_done": False
    }


def get_analyzing_response():
    """Generate analyzing response"""
    return {
        "code": 0,
        "data": {
            "interim_answer": "... analizando BOAM, y BOCm",
            "final_answer": "",
            "fuentes": []
        },
        "error": "",
        "is_done": False
    }


def get_final_query_response(user_text, mode):
    """Generate final query response based on mode"""
    mode_responses = {
        'preguntas_frecuentes': "Segun las fuentes analizadas, esta es una consulta frecuente sobre {topic}...",
        'datos': "Los datos oficiales muestran que {topic} presenta las siguientes cifras...",
        'medios': "La cobertura mediatica sobre {topic} indica diversas perspectivas...",
        'debates': "El debate publico sobre {topic} incluye diferentes posturas...",
        'encuestas': "Las encuestas recientes sobre {topic} reflejan la opinion ciudadana..."
    }

    # Determine topic from user_text
    topic = "ayudas economicas para familias" if "ayuda" in user_text.lower() else "el tema consultado"

    base_response = mode_responses.get(mode, mode_responses['datos']).format(topic=topic)

    final_answer = f"""{base_response}

El PaÃ­s - 2024-08-10 https://elpais.com/ayudas-madrid

Las medidas incluyen bonos de 1.200 euros para familias con ingresos inferiores a 35.000 euros anuales..."""

    return {
        "code": 0,
        "data": {
            "interim_answer": "",
            "final_answer": final_answer,
            "fuentes": [
                {
                    "idx": 1,
                    "fuente": "El Pais",
                    "publish_date": "2024-08-10",
                    "ref": "https://elpais.com/ayudas-madrid"
                },
                {
                    "idx": 2,
                    "fuente": "BOCM - Boletin Oficial",
                    "publish_date": "2024-08-05",
                    "ref": "http://goverme.hiprax.com:5000/static/sources/bocm_2024_08_05.pdf"
                }
            ]
        },
        "error": "",
        "is_done": True
    }


# ============== CORS & OPTIONS ==============

@app.route('/api/v1/machiavelli/fact_checker', methods=['OPTIONS'])
@app.route('/api/v1/machiavelli/fact_checker/<job_id>', methods=['OPTIONS'])
@app.route('/api/v1/machiavelli/query', methods=['OPTIONS'])
@app.route('/api/v1/validacion-intervencion', methods=['OPTIONS'])
@app.route('/api/v1/encuesta/<encuesta_id>/resultados', methods=['OPTIONS'])
@app.route('/api/v1/encuesta/<encuesta_id>/insights', methods=['OPTIONS'])
@app.route('/api/v1/encuesta/<encuesta_id>/segmentacion', methods=['OPTIONS'])
@app.route('/api/v1/chatbot/consulta', methods=['OPTIONS'])
@app.route('/api/v1/jobs', methods=['OPTIONS'])
@app.route('/api/v1/jobs/<job_id>', methods=['OPTIONS'])
@app.route('/api/v1/sessions/<session_id>/jobs', methods=['OPTIONS'])
def handle_options(**kwargs):
    """Respond to CORS preflight OPTIONS requests."""
    return Response(
        status=200,
        headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS'
        }
    )


@app.route('/health', methods=['GET'])
def health_check():
    """Return mock server health status with active resource counts."""
    return jsonify({
        "status": "OK",
        "message": "Mock API is running",
        "active_jobs": len(jobs),
        "active_audio_jobs": len(audio_jobs),
        "active_sessions": len(sessions),
        "active_surveys": len(encuestas)
    })


if __name__ == '__main__':
    print("Mock API started with full functionality!")
    print("\nFact Checking:")
    print("  POST http://localhost:5000/api/v1/machiavelli/fact_checker")
    print("  POST http://localhost:5000/api/v1/machiavelli/query")
    print("\nPolitical Surveys:")
    print("  POST http://localhost:5000/api/v1/validacion-intervencion")
    print("  GET  http://localhost:5000/api/v1/encuesta/{id}/resultados")
    print("  GET  http://localhost:5000/api/v1/encuesta/{id}/insights")
    print("  GET  http://localhost:5000/api/v1/encuesta/{id}/segmentacion")
    print("  POST http://localhost:5000/api/v1/chatbot/consulta")
    print("\nAudio Streaming:")
    print("  POST http://localhost:5000/api/v1/jobs")
    print("  WS   ws://localhost:5000/socket.io/")
    print("\nHealth: GET http://localhost:5000/health")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)