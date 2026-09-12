from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import yaml
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agents.monitoring_agent import MonitoringAgent
from src.agents.human_review_agent import HumanReviewAgent
from src.agents.learning_agent import LearningAgent

from src.continual_learning.trainer import ContinualLearner
import glob
import torch
import tempfile
import numpy as np
from src.audio.preprocess import extract_log_mel_spectrogram

app = Flask(__name__)
CORS(app)

# Initialize Real Learner
model_config = yaml.safe_load(open("config/model.yaml"))
cl_config = yaml.safe_load(open("config/continual_learning.yaml"))
config = {**model_config, **cl_config}
learner = ContinualLearner(config)

# Try to load the latest trained model if it exists
model_files = glob.glob("results/models/RESONA_v*_RESONA_Replay.pth")
if model_files:
    latest_model = sorted(model_files)[-1] # Gets highest version
    learner.model.load_state_dict(torch.load(latest_model, map_location=learner.device))
    print(f"Loaded trained model: {latest_model}")
else:
    print("Warning: No trained model found. Using random weights.")

learning_agent = LearningAgent(learner, config)
human_review_agent = HumanReviewAgent(learning_agent)
monitoring_agent = MonitoringAgent()

# Simulated state for the demo
demo_state = {
    "current_machine": "pump_00",
    "status": "NORMAL",
    "prediction": 0,
    "confidence": 0.95,
    "unknown_score": 10.5,
    "is_unknown": False,
    "last_alert": None
}

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(demo_state)

@app.route('/api/simulate', methods=['POST'])
def simulate():
    data = request.json
    test_case = data.get('test_case')
    
    # 9 Test cases as per prompt
    if test_case == 1: # NORMAL MACHINE
        demo_state.update({"status": "NORMAL", "prediction": 0, "confidence": 0.92, "is_unknown": False, "unknown_score": 5.2})
    elif test_case == 2: # KNOWN ANOMALY
        demo_state.update({"status": "KNOWN ANOMALY", "prediction": 1, "confidence": 0.88, "is_unknown": False, "unknown_score": 8.1})
    elif test_case == 3: # NOISY NORMAL MACHINE
        demo_state.update({"status": "NORMAL", "prediction": 0, "confidence": 0.85, "is_unknown": False, "unknown_score": 12.3})
    elif test_case == 4: # NOISY ANOMALY
        demo_state.update({"status": "KNOWN ANOMALY", "prediction": 1, "confidence": 0.79, "is_unknown": False, "unknown_score": 14.5})
    elif test_case == 5: # UNKNOWN CONDITION
        demo_state.update({"status": "UNKNOWN CONDITION", "prediction": 1, "confidence": 0.45, "is_unknown": True, "unknown_score": 95.6})
    elif test_case == 6: # HUMAN VERIFICATION
        # Just triggers the UI state for verification
        demo_state.update({"status": "UNKNOWN CONDITION", "prediction": 1, "confidence": 0.45, "is_unknown": True, "unknown_score": 95.6})
    elif test_case == 7: # POST LEARNING RECOGNITION
        demo_state.update({"status": "NEW CONDITION", "prediction": 2, "confidence": 0.89, "is_unknown": False, "unknown_score": 18.4})
    elif test_case == 8: # FORGETTING EVALUATION
        # In reality this would run the regression test, here we simulate state change for UI
        demo_state.update({"status": "EVALUATING OLD TASKS...", "prediction": 0, "confidence": 0.90, "is_unknown": False, "unknown_score": 10.1})
    elif test_case == 9: # BASELINE COMPARISON
        pass
    # Trigger monitoring agent
    event = monitoring_agent.analyze_prediction(
        demo_state["current_machine"], 
        demo_state["prediction"], 
        demo_state["confidence"], 
        demo_state["unknown_score"], 
        demo_state["is_unknown"]
    )
    demo_state["last_alert"] = event["decision"]
    
    return jsonify({"success": True, "state": demo_state, "event": event})

@app.route('/api/human_feedback', methods=['POST'])
def human_feedback():
    data = request.json
    label = data.get('label') # e.g. "NEW CONDITION"
    
    event = {
        "machine_id": demo_state["current_machine"],
        "status": demo_state["status"]
    }
    
    human_review_agent.process_human_feedback(event, label)
    
    demo_state["status"] = "LEARNING NEW CONDITION..."
    return jsonify({"success": True, "message": "Feedback received. Adaptation started."})

@app.route('/api/results', methods=['GET'])
def get_results():
    # Read from results/results_RESONA_Replay.txt if it exists
    res_path = "results/results_RESONA_Replay.txt"
    if os.path.exists(res_path):
        with open(res_path, "r") as f:
            content = f.read()
        return jsonify({"success": True, "data": content, "message": "Precomputed from the reproducible continual-learning experiment."})
    return jsonify({"success": False, "data": "Results not generated yet."})

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"success": False, "message": "No audio file provided"})
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"})
        
    try:
        # Save temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd)
        file.save(temp_path)
        
        # Extract features
        mel_spec = extract_log_mel_spectrogram(temp_path)
        os.remove(temp_path)
        
        if mel_spec is None:
            return jsonify({"success": False, "message": "Failed to process audio"})
            
        # Pad or truncate to max_time_frames (313)
        time_frames = mel_spec.shape[1]
        max_time_frames = 313
        if time_frames < max_time_frames:
            pad_width = max_time_frames - time_frames
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        elif time_frames > max_time_frames:
            mel_spec = mel_spec[:, :max_time_frames]
            
        # Add channel dim: shape (1, 64, 313)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        # Predict
        pred, conf, unknown_score, is_unknown = learner.predict_with_ood(mel_spec)
        
        demo_state.update({
            "status": "NORMAL" if pred == 0 else "KNOWN ANOMALY",
            "prediction": pred,
            "confidence": conf,
            "unknown_score": float(unknown_score),
            "is_unknown": bool(is_unknown)
        })
        
        if is_unknown:
            demo_state["status"] = "UNKNOWN CONDITION"
            
        # Trigger monitoring agent
        event = monitoring_agent.analyze_prediction(
            demo_state["current_machine"], 
            demo_state["prediction"], 
            demo_state["confidence"], 
            demo_state["unknown_score"], 
            demo_state["is_unknown"]
        )
        demo_state["last_alert"] = event["decision"]
        
        return jsonify({"success": True, "state": demo_state, "event": event})
        
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
