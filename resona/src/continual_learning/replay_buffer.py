import random
import torch

class ReplayBuffer:
    def __init__(self, memory_budget_per_class=100):
        self.memory_budget = memory_budget_per_class
        # Dictionary mapping class_label -> list of (mel_spec, label, machine_id)
        # Note: label is condition (Normal 0, Anomaly 1). We should also distinguish task/machine
        # To truly prevent catastrophic forgetting across domains, we store samples per (machine_id, condition).
        self.buffer = {}
        
    def add_samples(self, df_subset, features=None):
        """
        df_subset: pandas dataframe of the new task data.
        If we wanted to use 'herding' (like iCaRL), we'd use features. For now, random selection.
        """
        # Group by machine_id and label
        grouped = df_subset.groupby(['machine_id', 'label'])
        
        for (machine_id, label), group in grouped:
            key = f"{machine_id}_{label}"
            
            # Randomly select up to memory_budget samples
            if len(group) > self.memory_budget:
                selected = group.sample(n=self.memory_budget, random_state=42)
            else:
                selected = group
                
            if key not in self.buffer:
                self.buffer[key] = selected
            else:
                # In case we append, we ensure we don't exceed budget
                # For this setup, tasks don't overlap in machine_id, so it's a simple assignment
                self.buffer[key] = selected
                
    def get_buffer_df(self):
        """
        Returns a single dataframe containing all replay samples.
        """
        import pandas as pd
        if not self.buffer:
            return pd.DataFrame()
        return pd.concat(self.buffer.values(), ignore_index=True)
