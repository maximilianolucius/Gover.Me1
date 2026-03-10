"""API endpoint tests for the Machiavelli mock application.

Validates request/response structure for fact-checker, survey, audio streaming,
health-check, and CORS endpoints using the Flask test client.
"""

import pytest
import json
import uuid
from machiavelli.mock_app import app, jobs, sessions, audio_jobs, encuestas


@pytest.fixture
def client():
    """Provide a Flask test client configured for testing."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def clear_storage():
    """Clear in-memory storage before each test"""
    jobs.clear()
    sessions.clear()
    audio_jobs.clear()
    encuestas.clear()


class TestFactCheckerEndpoints:
    """Tests for the fact-checker submission and result retrieval endpoints."""

    def test_submit_fact_check_structure(self, client, clear_storage):
        response = client.post('/api/v1/machiavelli/fact_checker',
                               json={'user_text': 'Test fact to check'})

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'job_id' in data
        assert 'status' in data
        assert 'message' in data
        assert data['status'] == 'submitted'
        assert isinstance(data['job_id'], str)

    def test_get_fact_check_result_structure(self, client, clear_storage):
        # First create a job
        submit_response = client.post('/api/v1/machiavelli/fact_checker',
                                      json={'user_text': 'Test fact'})
        job_id = submit_response.get_json()['job_id']

        response = client.get(f'/api/v1/machiavelli/fact_checker/{job_id}')

        assert response.status_code == 200
        assert response.mimetype == 'text/event-stream'

    def test_query_processing_structure(self, client, clear_storage):
        response = client.post('/api/v1/machiavelli/query',
                               json={'user_text': 'Test query', 'mode': 'datos'})

        assert response.status_code == 200
        assert response.mimetype == 'text/event-stream'

    def test_query_invalid_mode(self, client, clear_storage):
        response = client.post('/api/v1/machiavelli/query',
                               json={'user_text': 'Test', 'mode': 'invalid_mode'})

        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data


class TestSurveyEndpoints:
    """Tests for survey creation, results, insights, segmentation, and chatbot endpoints."""

    def test_create_survey_intervention_structure(self, client, clear_storage):
        survey_data = {
            'titulo': 'Test Survey',
            'fecha': '2025-08-19',
            'base_datos': 'test_db',
            'preguntas': ['Question 1', 'Question 2']
        }

        response = client.post('/api/v1/validacion-intervencion', json=survey_data)

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'id' in data
        assert 'status' in data
        assert 'numero_encuestados' in data
        assert 'fecha_creacion' in data
        assert 'url_encuesta' in data
        assert data['status'] == 'created'
        assert data['numero_encuestados'] == 0

    def test_get_survey_results_structure(self, client, clear_storage):
        # First create a survey
        survey_response = client.post('/api/v1/validacion-intervencion',
                                      json={'titulo': 'Test Survey'})
        survey_id = survey_response.get_json()['id']

        response = client.get(f'/api/v1/encuesta/{survey_id}/resultados')

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'encuesta_id' in data
        assert 'titulo' in data
        assert 'numero_encuestados' in data
        assert 'fecha' in data
        assert 'resultados' in data
        assert 'pregunta_1' in data['resultados']
        assert 'pregunta_2' in data['resultados']

    def test_get_survey_insights_structure(self, client, clear_storage):
        # First create a survey
        survey_response = client.post('/api/v1/validacion-intervencion',
                                      json={'titulo': 'Test Survey'})
        survey_id = survey_response.get_json()['id']

        response = client.get(f'/api/v1/encuesta/{survey_id}/insights')

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'insights_positivos' in data
        assert 'frases_debiles' in data
        assert 'insights_gente_alta' in data

        # Check structure of insights
        for insight in data['insights_positivos']:
            assert 'texto' in insight
            assert 'score' in insight
            assert 'categoria' in insight

    def test_get_survey_segmentation_structure(self, client, clear_storage):
        # First create a survey
        survey_response = client.post('/api/v1/validacion-intervencion',
                                      json={'titulo': 'Test Survey'})
        survey_id = survey_response.get_json()['id']

        response = client.get(f'/api/v1/encuesta/{survey_id}/segmentacion?segmento=genero')

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'segmento' in data
        assert 'resultados' in data
        assert data['segmento'] == 'genero'

    def test_chatbot_query_structure(self, client, clear_storage):
        # First create a survey
        survey_response = client.post('/api/v1/validacion-intervencion',
                                      json={'titulo': 'Test Survey'})
        survey_id = survey_response.get_json()['id']

        response = client.post('/api/v1/chatbot/consulta',
                               json={'encuesta_id': survey_id, 'pregunta': 'Test question'})

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'respuesta' in data
        assert 'fuentes' in data
        assert 'confianza' in data
        assert isinstance(data['fuentes'], list)
        assert isinstance(data['confianza'], float)

    def test_survey_not_found(self, client, clear_storage):
        fake_id = str(uuid.uuid4())

        response = client.get(f'/api/v1/encuesta/{fake_id}/resultados')
        assert response.status_code == 404

        response = client.get(f'/api/v1/encuesta/{fake_id}/insights')
        assert response.status_code == 404

        response = client.get(f'/api/v1/encuesta/{fake_id}/segmentacion')
        assert response.status_code == 404


class TestAudioStreamingEndpoints:
    """Tests for audio job creation, status polling, cancellation, and session listing."""

    def test_create_audio_job_structure(self, client, clear_storage):
        job_data = {
            'session_id': str(uuid.uuid4()),
            'config': {'sample_rate': 16000}
        }

        response = client.post('/api/v1/jobs', json=job_data)

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'job_id' in data
        assert 'session_id' in data
        assert 'status' in data
        assert 'websocket_url' in data
        assert 'created_at' in data
        assert data['status'] == 'initialized'

    def test_get_audio_job_status_structure(self, client, clear_storage):
        # First create a job
        job_response = client.post('/api/v1/jobs', json={})
        job_id = job_response.get_json()['job_id']

        response = client.get(f'/api/v1/jobs/{job_id}')

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'session_id' in data
        assert 'status' in data
        assert 'config' in data
        assert 'progress' in data
        assert 'results' in data
        assert 'created_at' in data
        assert 'updated_at' in data

    def test_cancel_audio_job_structure(self, client, clear_storage):
        # First create a job
        job_response = client.post('/api/v1/jobs', json={})
        job_id = job_response.get_json()['job_id']

        response = client.delete(f'/api/v1/jobs/{job_id}')

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'job_id' in data
        assert 'status' in data
        assert 'message' in data
        assert data['status'] == 'cancelled'

    def test_list_session_jobs_structure(self, client, clear_storage):
        # First create a session and jobs
        session_id = str(uuid.uuid4())
        client.post('/api/v1/jobs', json={'session_id': session_id})
        client.post('/api/v1/jobs', json={'session_id': session_id})

        response = client.get(f'/api/v1/sessions/{session_id}/jobs')

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'session_id' in data
        assert 'jobs' in data
        assert 'total' in data
        assert isinstance(data['jobs'], list)
        assert data['total'] == len(data['jobs'])

    def test_audio_job_not_found(self, client, clear_storage):
        fake_id = str(uuid.uuid4())

        response = client.get(f'/api/v1/jobs/{fake_id}')
        assert response.status_code == 404

        response = client.delete(f'/api/v1/jobs/{fake_id}')
        assert response.status_code == 404

    def test_session_not_found(self, client, clear_storage):
        fake_session_id = str(uuid.uuid4())

        response = client.get(f'/api/v1/sessions/{fake_session_id}/jobs')
        assert response.status_code == 404


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_structure(self, client, clear_storage):
        response = client.get('/health')

        assert response.status_code == 200
        data = response.get_json()

        # Check response structure
        assert 'status' in data
        assert 'message' in data
        assert 'active_jobs' in data
        assert 'active_audio_jobs' in data
        assert 'active_sessions' in data
        assert 'active_surveys' in data
        assert data['status'] == 'OK'


class TestCORSHandling:
    """Tests that CORS OPTIONS pre-flight requests are handled on key endpoints."""

    def test_options_endpoints(self, client):
        endpoints = [
            '/api/v1/machiavelli/fact_checker',
            '/api/v1/machiavelli/query',
            '/api/v1/validacion-intervencion',
            '/api/v1/jobs'
        ]

        for endpoint in endpoints:
            response = client.options(endpoint)
            assert response.status_code == 200
            # Flask's built-in OPTIONS handling may override custom handlers
            # Check that endpoint accepts OPTIONS method
            assert 'OPTIONS' in response.headers.get('Allow', '')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])