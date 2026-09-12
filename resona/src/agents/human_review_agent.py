class HumanReviewAgent:
    def __init__(self, learning_agent):
        self.learning_agent = learning_agent
        
    def process_human_feedback(self, event, human_label):
        """
        event: The event dictionary from the MonitoringAgent
        human_label: "NORMAL", "KNOWN ANOMALY", or "NEW CONDITION"
        """
        print(f"[HUMAN REVIEW AGENT] Received human feedback for Machine {event['machine_id']}: {human_label}")
        
        if human_label == "NEW CONDITION":
            print(f"[HUMAN REVIEW AGENT] Validating feedback and forwarding to Learning Agent...")
            # Validate feedback (e.g., ensure we have the audio sample saved)
            # Trigger continual learning workflow
            self.learning_agent.trigger_adaptation(event, human_label)
        else:
            print(f"[HUMAN REVIEW AGENT] Condition marked as {human_label}. Logging feedback.")
            # Would update the database with the verified label to improve existing classes
