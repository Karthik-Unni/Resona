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

app = Flask(__name__)
CORS(app)

# Initialize Agents
config = yaml.safe_load(open("config/continual_learning.yaml"))
class DummyLearner:
    method = config.get("method", "RESONA_Replay")
    
learner = DummyLearner()
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
        # Triggered via different endpoint
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
        return jsonify({"success": True, "data": content})
    return jsonify({"success": False, "data": "Results not generated yet."})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
