from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import random
import yaml
import os
import sys
import time
import uuid
import glob
import torch
import tempfile
import numpy as np
import librosa

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agents.monitoring_agent import MonitoringAgent
from src.agents.human_review_agent import HumanReviewAgent
from src.agents.learning_agent import LearningAgent

from src.continual_learning.trainer import ContinualLearner
from src.audio.preprocess import extract_log_mel_spectrogram

app = Flask(__name__, static_folder='../frontend')
CORS(app)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

@app.route('/')
def serve_desktop():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/mobile')
def serve_mobile():
    return send_from_directory(FRONTEND_DIR, 'mobile.html')

# Initialize Real Learner
model_config = yaml.safe_load(open("config/model.yaml"))
cl_config = yaml.safe_load(open("config/continual_learning.yaml"))
config = {**model_config, **cl_config}
learner = ContinualLearner(config)

model_version = 0
model_files = glob.glob("results/models/RESONA_v*_RESONA_Replay.pth")
if model_files:
    latest_model = sorted(model_files)[-1]
    model_version = int(latest_model.split('_v')[1].split('_')[0])
    learner.model.load_state_dict(torch.load(latest_model, map_location=learner.device))
    print(f"Loaded trained model: {latest_model}")
else:
    print("Warning: No trained model found. Using random weights.")

learning_agent = LearningAgent(learner, config)
human_review_agent = HumanReviewAgent(learning_agent)
monitoring_agent = MonitoringAgent()

# --- REAL BACKEND STATE ---
# Stores the real live state of the monitored machine
backend_state = {
    "current_machine": "Pump-00",
    "status": "NORMAL",
    "prediction": 0,
    "confidence": 0.0,
    "unknown_score": 0.0,
    "is_unknown": False,
    "last_alert": None,
    "latency": 0.0,
    "model_version": model_version,
    "waveform": [],
    "spectrogram": [],
    "active_review_id": None
}

# 10 second rolling buffer (16kHz = 160,000 samples)
audio_buffer = np.zeros(160000, dtype=np.float32)

import threading
simulation_thread = None
simulation_active = False

def simulate_wav_stream(file_path):
    global audio_buffer, simulation_active
    simulation_active = True
    try:
        y, sr = librosa.load(file_path, sr=16000)
        chunk_size = 16000 # 1 second
        
        for i in range(0, len(y), chunk_size):
            if not simulation_active:
                break
                
            chunk = y[i:i+chunk_size]
            if len(chunk) < chunk_size:
                # pad with zeros if end of file
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
            # Append to rolling buffer
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk
            
            # Save rolling buffer temporarily to extract features
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            import soundfile as sf
            sf.write(temp_path, audio_buffer, 16000)
            os.close(temp_fd)
            
            process_audio_file(temp_path)
            os.remove(temp_path)
            
            time.sleep(1) # simulate real time
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        simulation_active = False

@app.route('/api/simulate', methods=['POST'])
def simulate():
    global simulation_thread, simulation_active
    data = request.json
    test_case = data.get('test_case')
    
    # Stop existing simulation
    simulation_active = False
    if simulation_thread and simulation_thread.is_alive():
        simulation_thread.join()
        
    wav_file = None
    if test_case == 1:
        wav_file = "data/raw/0_dB_pump/pump/id_00/normal/00000000.wav"
        backend_state["current_machine"] = "Pump-00"
    elif test_case == 2:
        wav_file = "data/raw/0_dB_pump/pump/id_02/abnormal/00000027.wav"
        backend_state["current_machine"] = "Pump-02"
    elif test_case == 3:
        wav_file = "data/raw/0_dB_pump/pump/id_00/normal/00000001.wav"
        backend_state["current_machine"] = "Pump-00"
    elif test_case == 4:
        wav_file = "data/raw/0_dB_pump/pump/id_02/abnormal/00000028.wav"
        backend_state["current_machine"] = "Pump-02"
    elif test_case == 5:
        wav_file = "data/raw/0_dB_pump/pump/id_04/abnormal/00000000.wav"
        backend_state["current_machine"] = "Pump-04"
    elif test_case == 6:
        # trigger human loop
        wav_file = "data/raw/0_dB_pump/pump/id_04/abnormal/00000001.wav"
        backend_state["current_machine"] = "Pump-04"
    elif test_case == 7:
        wav_file = "data/raw/0_dB_pump/pump/id_00/normal/00000002.wav"
        backend_state["current_machine"] = "Pump-00"
    elif test_case == 8:
        # We don't stream for 8, just change state to show evaluating
        backend_state["status"] = "EVALUATING OLD TASKS..."
        return jsonify({"success": True})
        
    if wav_file and os.path.exists(wav_file):
        simulation_thread = threading.Thread(target=simulate_wav_stream, args=(wav_file,))
        simulation_thread.daemon = True
        simulation_thread.start()
        return jsonify({"success": True, "message": f"Started streaming {wav_file}"})
    else:
        return jsonify({"success": False, "message": "WAV file not found or invalid test case."})

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(backend_state)

@app.route('/api/human_feedback', methods=['POST'])
def human_feedback():
    data = request.json
    label = data.get('label') # e.g. "NEW CONDITION"
    
    event = {
        "machine_id": backend_state["current_machine"],
        "status": backend_state["status"]
    }
    
    # Process feedback (this pushes to replay buffer and might update model)
    human_review_agent.process_human_feedback(event, label)
    
    backend_state["status"] = "NORMAL"
    backend_state["active_review_id"] = None
    
    # Update model version if new model was saved
    model_files = glob.glob("results/models/RESONA_v*_RESONA_Replay.pth")
    if model_files:
        latest_model = sorted(model_files)[-1]
        backend_state["model_version"] = int(latest_model.split('_v')[1].split('_')[0])

    return jsonify({"success": True, "message": "Feedback received. Adaptation started."})

@app.route('/api/results', methods=['GET'])
def get_results():
    res_path = "results/results_RESONA_Replay.txt"
    if os.path.exists(res_path):
        with open(res_path, "r") as f:
            content = f.read()
        return jsonify({"success": True, "data": content})
    return jsonify({"success": False, "data": "Results not generated yet."})

def process_audio_file(file_path):
    start_time = time.time()
    
    # Extract features using exact preprocessing
    mel_spec = extract_log_mel_spectrogram(file_path)
    if mel_spec is None:
        raise Exception("Preprocessing failed")
        
    # Get raw waveform for UI (downsampled for performance)
    y, sr = librosa.load(file_path, sr=16000)
    waveform_downsampled = y[::160].tolist() # 100 points per second
    
    # Format mel_spec for rendering (normalize for UI 0-255)
    mel_normalized = (mel_spec - np.min(mel_spec)) / (np.max(mel_spec) - np.min(mel_spec) + 1e-6)
    mel_flat = mel_normalized.tolist()
        
    # Pad or truncate to max_time_frames (313) for the CNN
    time_frames = mel_spec.shape[1]
    max_time_frames = 313
    if time_frames < max_time_frames:
        pad_width = max_time_frames - time_frames
        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
    elif time_frames > max_time_frames:
        mel_spec = mel_spec[:, :max_time_frames]
        
    # Add channel dim: shape (1, 64, 313)
    mel_spec_tensor = np.expand_dims(mel_spec, axis=0)
    
    # Actual Model Inference
    pred, conf, unknown_score, is_unknown = learner.predict_with_ood(mel_spec_tensor)
    
    latency = (time.time() - start_time) * 1000 # in ms
    
    # Actual decision logic
    # 0 = Normal, 1 = Anomaly, 2 = New Condition
    if is_unknown:
        decision_status = "UNKNOWN CONDITION"
    elif pred == 1:
        decision_status = "ANOMALOUS"
    elif pred == 2:
        decision_status = "NEW CONDITION"
    else:
        decision_status = "NORMAL"

    # Update backend state
    backend_state.update({
        "status": decision_status,
        "prediction": pred,
        "confidence": conf,
        "unknown_score": float(unknown_score),
        "is_unknown": bool(is_unknown),
        "latency": latency,
        "waveform": waveform_downsampled,
        "spectrogram": mel_flat
    })
    
    # Deduplicate review alerts (only create one review ID if multiple UNKNOWNs happen)
    if decision_status == "UNKNOWN CONDITION" and not backend_state["active_review_id"]:
        backend_state["active_review_id"] = str(uuid.uuid4())[:8].upper()
    elif decision_status != "UNKNOWN CONDITION":
        backend_state["active_review_id"] = None

    # Alert triggering
    event = monitoring_agent.analyze_prediction(
        backend_state["current_machine"], 
        backend_state["prediction"], 
        backend_state["confidence"], 
        backend_state["unknown_score"], 
        backend_state["is_unknown"]
    )
    backend_state["last_alert"] = event["decision"]
    
    return backend_state

@app.route('/api/process_audio_chunk', methods=['POST'])
def process_audio_chunk():
    global audio_buffer
    if 'audio' not in request.files:
        return jsonify({"success": False, "message": "No audio file provided"})
    
    file = request.files['audio']
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        file.save(temp_path)
        
        # Load the 1s chunk
        y_chunk, sr = librosa.load(temp_path, sr=16000)
        os.remove(temp_path)
        
        # Append to rolling buffer
        audio_buffer = np.roll(audio_buffer, -len(y_chunk))
        audio_buffer[-len(y_chunk):] = y_chunk
        
        # Save rolling buffer temporarily to extract features
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        import soundfile as sf
        sf.write(temp_path, audio_buffer, 16000)
        os.close(temp_fd)
        
        process_audio_file(temp_path)
        os.remove(temp_path)
        
        return jsonify({"success": True, "state": backend_state})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"success": False, "message": "No audio file provided"})
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"})
        
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        file.save(temp_path)
        
        # Determine machine from filename if present (e.g. pump_00_normal.wav)
        if "pump_02" in file.filename:
            backend_state["current_machine"] = "Pump-02"
        elif "pump_04" in file.filename:
            backend_state["current_machine"] = "Pump-04"
        else:
            backend_state["current_machine"] = "Pump-00"
            
        process_audio_file(temp_path)
        os.remove(temp_path)
        
        return jsonify({"success": True, "state": backend_state})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
