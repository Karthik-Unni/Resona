import datetime

class MonitoringAgent:
    def __init__(self):
        self.logs = []
        
    def analyze_prediction(self, machine_id, pred_label, confidence, unknown_score, is_unknown):
        """
        Receives model output and decides the workflow path.
        """
        timestamp = datetime.datetime.now().isoformat()
        
        if is_unknown:
            decision = "REQUEST_REVIEW"
            status = "UNKNOWN CONDITION"
        elif pred_label == 1:
            decision = "ALERT"
            status = "KNOWN ANOMALY"
        else:
            decision = "LOG_ONLY"
            status = "NORMAL"
            
        event = {
            "timestamp": timestamp,
            "machine_id": machine_id,
            "status": status,
            "prediction": pred_label,
            "confidence": confidence,
            "unknown_score": unknown_score,
            "decision": decision
        }
        self.logs.append(event)
        
        if decision == "ALERT":
            self.create_alert(event)
        elif decision == "REQUEST_REVIEW":
            self.request_review(event)
            
        return event

    def create_alert(self, event):
        print(f"[MONITORING AGENT] 🚨 ALERT! Machine {event['machine_id']} requires inspection. Status: {event['status']} (Confidence: {event['confidence']:.2f})")
        # In a real system, this would trigger an API call to a mobile app or notification service.

    def request_review(self, event):
        print(f"[MONITORING AGENT] ⚠️ UNKNOWN CONDITION detected on Machine {event['machine_id']}. Requesting human verification.")
