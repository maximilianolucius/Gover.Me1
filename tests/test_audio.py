"""WebSocket audio streaming test for the Machiavelli fact-checker.

Loads a local audio file, streams it in real-time chunks over Socket.IO,
and polls the job status endpoint until transcription and fact-checking complete.
"""

import socketio
import time
import numpy as np
import requests
from pydub import AudioSegment
import os
import threading
from datetime import datetime

# Connect to ngrok URL
sio = socketio.Client()
job_id = None
AUDIO_FILE = "Grabación (10).m4a"
status_thread = None
stop_polling = threading.Event()


@sio.event
def connect():
    """Handle successful WebSocket connection."""
    print("🔗 Connected to server")


@sio.event
def disconnect():
    """Handle WebSocket disconnection."""
    print("❌ Disconnected from server")


# Catch all events for debugging
@sio.on('*')
def catch_all(event, data):
    """Log any unhandled Socket.IO event for debugging."""
    print(f"📨 Event: {event} | Data: {data}")


@sio.event
def recording_started(data):
    """Store the job ID and start the status-polling thread."""
    global job_id, status_thread
    job_id = data.get('job_id')
    print(f"🎤 Recording started - job_id: {job_id}")
    
    # Start parallel status polling once we have job_id
    if job_id and not status_thread:
        status_thread = threading.Thread(target=poll_status_continuously, daemon=True)
        status_thread.start()
        print("🔄 Started parallel status polling thread")


@sio.event
def recording_stopped(data):
    """Handle the recording-stopped acknowledgement from the server."""
    global job_id
    if not job_id:
        job_id = data.get('job_id')
    print(f"ℹ️ Recording stopped response: {data}")


@sio.event
def transcription_update(data):
    """Log a partial transcription result as it arrives."""
    print(f"🎯 Partial: {data.get('text')} (job_id: {data.get('job_id')})")


@sio.event
def transcription_complete(data):
    """Log the final transcription result."""
    print(f"✅ Final: {data.get('text')} (job_id: {data.get('job_id')})")


@sio.event
def fact_check_started(data):
    """Log when the server begins fact-checking the transcription."""
    print(f"🔍 Fact-check started (job_id: {data.get('job_id')})")


@sio.event
def fact_check_complete(data):
    """Log the completed fact-check result."""
    print(f"📋 Fact-check done (job_id: {data.get('job_id')})")
    print(f"Result: {data.get('result', {}).get('data', {})}")


@sio.event
def error(data):
    """Log any server-side error event."""
    print(f"❌ Error: {data}")


def poll_status_continuously():
    """Continuously poll job status every 2 seconds in parallel"""
    global job_id
    
    while not stop_polling.is_set() and job_id:
        try:
            response = requests.get(
                f"https://94c06b5d1c66.ngrok-free.app/api/v1/machiavelli/fact_checker/{job_id}/status",
                timeout=5
            )
            status_data = response.json()
            
            current_time = datetime.now().strftime("%H:%M:%S")
            status = status_data.get('status', 'unknown')
            user_text = status_data.get('user_text', '')
            
            print(f"⏰ {current_time} | Status: {status} | Text: '{user_text}'")
            
            # Stop polling if job completed or failed
            if status in ['completed', 'failed']:
                print(f"🏁 Stopping status polling - job {status}")
                break
                
        except Exception as e:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"⚠️ {current_time} | Error polling status: {e}")
        
        # Wait 2 seconds before next poll
        stop_polling.wait(2)


def load_and_prepare_audio(file_path):
    """Load M4A file and convert to format expected by server"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    print(f"📁 Loading audio file: {file_path}")

    # Load audio file using pydub
    audio = AudioSegment.from_file(file_path)

    # Convert to mono, 16-bit PCM at 48kHz (common browser sample rate)
    audio = audio.set_channels(1)  # Mono
    audio = audio.set_sample_width(2)  # 16-bit
    audio = audio.set_frame_rate(48000)  # 48kHz

    # Get raw audio data
    raw_audio = audio.raw_data

    # Convert to numpy array for chunking
    audio_array = np.frombuffer(raw_audio, dtype=np.int16)

    # Limit to 6 seconds
    max_samples = int(audio.frame_rate * 25)  # 6 seconds worth of samples

    # max_samples = 1e100


    if len(audio_array) > max_samples:
        audio_array = audio_array[:max_samples]
        print(f"🔪 Audio truncated to 6 seconds")

    print(
        f"🎵 Audio loaded: {len(audio_array)} samples, {len(audio_array) / audio.frame_rate:.1f}s duration, {audio.frame_rate}Hz")

    return audio_array, audio.frame_rate


def send_audio_in_chunks(audio_array, sample_rate, chunk_duration_ms=100):
    """Send audio data in timed chunks to simulate real-time streaming over Socket.IO."""

    # Calculate samples per chunk
    samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)

    total_chunks = len(audio_array) // samples_per_chunk
    if len(audio_array) % samples_per_chunk > 0:
        total_chunks += 1

    print(f"📡 Sending {total_chunks} audio chunks ({chunk_duration_ms}ms each)...")

    for i in range(total_chunks):
        start_idx = i * samples_per_chunk
        end_idx = min(start_idx + samples_per_chunk, len(audio_array))

        chunk_data = audio_array[start_idx:end_idx]
        audio_bytes = chunk_data.tobytes()

        sio.emit('audio_chunk', {
            'audio_data': list(audio_bytes),
            'chunk_number': i + 1,
            'sample_rate': sample_rate,
            'has_signal': True
        })

        # Progress indicator
        if (i + 1) % 20 == 0:
            progress = (i + 1) / total_chunks * 100
            print(f"📶 Progress: {progress:.1f}% ({i + 1}/{total_chunks} chunks)")

        time.sleep(chunk_duration_ms / 1000)  # Real-time simulation


def main():
    """Load an audio file, connect via WebSocket, stream it, and wait for results."""
    global job_id, status_thread, stop_polling

    try:
        # Check if audio file exists
        if not os.path.exists(AUDIO_FILE):
            print(f"❌ Audio file '{AUDIO_FILE}' not found in current directory")
            print("📂 Current directory contents:")
            for f in os.listdir('.'):
                if f.endswith(('.m4a', '.mp3', '.wav', '.mp4')):
                    print(f"   🎵 {f}")
            return

        # Load audio file
        audio_array, sample_rate = load_and_prepare_audio(AUDIO_FILE)

        # Connect to server
        print("🔌 Connecting to server...")
        sio.connect('https://94c06b5d1c66.ngrok-free.app', transports=['websocket'])

        # Start recording
        print("📤 Sending start_recording...")
        sio.emit('start_recording')
        time.sleep(2)  # Wait for response

        # Check if we got job_id
        if not job_id:
            print("⚠️ No job_id received, continuing anyway...")

        # Send actual audio file in chunks
        send_audio_in_chunks(audio_array, sample_rate)

        # Stop recording
        print("📤 Sending stop_recording...")
        sio.emit('stop_recording')
        print(f"ℹ️ Stopped recording - job_id: {job_id}")

        # Wait for final completion (status thread will handle polling)
        print("⏳ Waiting for job completion (up to 60 seconds)...")
        time.sleep(60)  # Let the status thread do the work
        
        # Get final result
        if job_id:
            try:
                result_response = requests.get(
                    f"https://94c06b5d1c66.ngrok-free.app/api/v1/machiavelli/fact_checker/{job_id}/result")
                result_json = result_response.json()
                print(f"📋 Final Result: {result_json}")
            except Exception as e:
                print(f"❌ Error getting final result: {e}")

    except FileNotFoundError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        # Stop the polling thread
        stop_polling.set()
        if status_thread and status_thread.is_alive():
            print("🛑 Stopping status polling thread...")
            status_thread.join(timeout=3)
        
        try:
            sio.disconnect()
        except:
            pass


if __name__ == "__main__":
    print("🎵 Machiavelli Audio File Test")
    print(f"📁 Target file: {AUDIO_FILE}")
    print("=" * 50)
    main()


len('Señoras y señores del GobiernoEn manos del gobierno salen ustedes a la palestra con cifras grandilocuentes y fuentes hablando de un verano histórico, pero la realidad que esconden sus propios informes es la de un modelo agotado y una gestión negligenteSu triunfalismo se cae como un castillo de naipes en cuanto uno rasca la superficieHablan de éxito, pero gran éxito, pero el aeropuerto de Málaga, la joya de la corona y el principal termómetroÉxito, pero el aeropuerto de Málaga, la joya de la corona y el principal termómetro')
len('Señoras y señores del Gobierno. En manos del gobierno salen ustedes a la palestra con cifras grandilocuentes y fuentes hablando de un verano histórico, pero la realidad que esconden sus propios informes es la de un modelo agotado y una gestión negligente. Su triunfalismo se cae como un castillo de naipes en cuanto uno rasca la superficie. Hablan de éxito, pero gran éxito, pero el aeropuerto de Málaga, la joya de la corona y el principal termómetro. Éxito, pero el aeropuerto de Málaga, la joya de la corona y el principal termómetro.')
len([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])