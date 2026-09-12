import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import copy
from src.models.cnn_classifier import SimpleCNN
from src.models.dataset import MIMIIDataset
from src.ood.detector import MahalanobisDetector
from src.continual_learning.replay_buffer import ReplayBuffer

class ContinualLearner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        
        self.method = config.get("method", "RESONA_Replay")
        self.buffer = ReplayBuffer(memory_budget_per_class=config.get("memory_budget_per_class", 100))
        
        self.task_history = [] # list of task names we have learned
        self.ood_detector = MahalanobisDetector(threshold_percentile=config.get("ood_threshold_percentile", 95))
        
    def train_on_task(self, task_name, train_df, val_df=None):
        print(f"--- Training on {task_name} using method {self.method} ---")
        
        # Depending on the method, we construct the actual training dataset
        if self.method == "RESONA_Replay":
            # Combine current task data with replay buffer
            buffer_df = self.buffer.get_buffer_df()
            if not buffer_df.empty:
                import pandas as pd
                combined_df = pd.concat([train_df, buffer_df], ignore_index=True)
                print(f"Added {len(buffer_df)} replay samples. Total training size: {len(combined_df)}")
            else:
                combined_df = train_df
        elif self.method == "Naive_FT":
            # Only use current task
            combined_df = train_df
        elif self.method == "Joint":
            # We assume train_df contains all data (Task 1 + Task 2 + etc)
            combined_df = train_df
        else:
            combined_df = train_df
            
        train_dataset = MIMIIDataset(combined_df)
        train_loader = DataLoader(train_dataset, batch_size=self.config.get("batch_size", 64), shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get("learning_rate", 0.001))
        criterion = nn.CrossEntropyLoss()
        
        epochs = self.config.get("epochs_per_task", 20)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
            acc = 100 * correct / total
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
                
        # Update replay buffer if we are using RESONA_Replay
        if self.method == "RESONA_Replay":
            self.buffer.add_samples(train_df)
            
        # Fit OOD Detector on the latest representation
        # Extract features for all current + past data (from buffer) to calibrate
        self.calibrate_ood(combined_df)
            
        self.task_history.append(task_name)
        
        # Save model version
        version = len(self.task_history)
        save_path = f"results/models/RESONA_v{version}_{self.method}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    def calibrate_ood(self, df):
        self.model.eval()
        dataset = MIMIIDataset(df)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                _, features = self.model(batch_x, return_features=True)
                all_features.append(features.cpu().numpy())
                all_labels.append(batch_y.numpy())
                
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        self.ood_detector.fit(all_features, all_labels)
        print("OOD detector calibrated.")
        
    def evaluate(self, df):
        self.model.eval()
        dataset = MIMIIDataset(df)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        acc = 100 * correct / total
        return acc

    def predict_with_ood(self, mel_spec):
        """
        mel_spec: single numpy array (1, n_mels, time)
        Returns: (prediction_label, confidence, unknown_score, is_unknown)
        """
        self.model.eval()
        tensor_x = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).to(self.device) # Add batch dim
        
        with torch.no_grad():
            logits, features = self.model(tensor_x, return_features=True)
            probs = torch.nn.functional.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            
            features_np = features.cpu().numpy()
            
            is_unknown = self.ood_detector.is_unknown(features_np)[0]
            # OOD distance for UI
            unknown_score = self.ood_detector.score_samples(features_np)[0]
            
        return pred.item(), conf.item(), unknown_score, is_unknown
