import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        self.num_classes = config.get("num_classes", 2)
        # Instantiate the custom SimpleCNN
        self.model = SimpleCNN(input_channels=1, num_classes=self.num_classes).to(self.device)
        
        self.method = config.get("method", "RESONA_Replay")
        self.buffer = ReplayBuffer(memory_budget_per_class=config.get("memory_budget_per_class", 100))
        
        self.task_history = [] # list of task names we have learned
        self.ood_detector = MahalanobisDetector(threshold_percentile=config.get("ood_threshold_percentile", 95))
        
        # EWC Specific
        self.ewc_lambda = config.get("ewc_lambda", 5000)
        self.fisher = {}
        self.optpar = {}
        
        # LwF Specific
        self.lwf_alpha = config.get("lwf_alpha", 1.0)
        self.prev_model = None
        
        # DER++ Specific
        self.der_alpha = config.get("der_alpha", 0.5)

    def compute_fisher(self, df):
        self.model.eval()
        dataset = MIMIIDataset(df)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            self.model.zero_grad()
            logits = self.model(batch_x)
            # Log likelihood
            log_probs = F.log_softmax(logits, dim=1)
            # True labels Fisher
            loss = F.nll_loss(log_probs, batch_y)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(loader)
                    
        self.fisher = fisher
        self.optpar = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        
    def train_on_task(self, task_name, train_df, val_df=None):
        print(f"--- Training on {task_name} using method {self.method} ---")
        
        # Depending on the method, we construct the actual training dataset
        if self.method in ["RESONA_Replay", "DER++"]:
            # Combine current task data with replay buffer
            buffer_df = self.buffer.get_buffer_df()
            if not buffer_df.empty:
                import pandas as pd
                combined_df = pd.concat([train_df, buffer_df], ignore_index=True)
                print(f"Added {len(buffer_df)} replay samples. Total training size: {len(combined_df)}")
            else:
                combined_df = train_df
        elif self.method in ["Naive_FT", "Joint", "EWC", "LwF", "RESONA_NoReplay"]:
            # Only use current task (or all for Joint)
            combined_df = train_df
        else:
            combined_df = train_df
            
        train_dataset = MIMIIDataset(combined_df)
        train_loader = DataLoader(train_dataset, batch_size=self.config.get("batch_size", 64), shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.get("learning_rate", 0.001))
        criterion = nn.CrossEntropyLoss()
        mse_criterion = nn.MSELoss()
        
        epochs = self.config.get("epochs_per_task", 20)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for batch in train_loader:
                if len(batch) == 3:
                    batch_x, batch_y, batch_logits = batch
                    batch_logits = batch_logits.to(self.device)
                else:
                    batch_x, batch_y = batch
                    batch_logits = None
                    
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                
                # EWC Penalty
                if self.method == "EWC" and self.fisher:
                    ewc_loss = 0
                    for n, p in self.model.named_parameters():
                        if p.requires_grad and n in self.fisher:
                            ewc_loss += (self.fisher[n] * (p - self.optpar[n]) ** 2).sum()
                    loss += (self.ewc_lambda / 2) * ewc_loss
                    
                # LwF Penalty
                if self.method == "LwF" and self.prev_model is not None:
                    with torch.no_grad():
                        prev_logits = self.prev_model(batch_x)
                    T = 2.0
                    log_p = F.log_softmax(logits / T, dim=1)
                    q = F.softmax(prev_logits / T, dim=1)
                    lwf_loss = F.kl_div(log_p, q, reduction='batchmean') * (T**2)
                    loss += self.lwf_alpha * lwf_loss
                    
                # DER++ Penalty
                if self.method == "DER++" and batch_logits is not None:
                    valid_mask = ~torch.isnan(batch_logits[:, 0])
                    if valid_mask.sum() > 0:
                        loss += self.der_alpha * mse_criterion(logits[valid_mask], batch_logits[valid_mask])
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
            acc = 100 * correct / total
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
                
        # Post-training steps per method
        if self.method in ["RESONA_Replay", "DER++", "RESONA_NoReplay"]:
            if self.method == "DER++":
                self.model.eval()
                task_ds = MIMIIDataset(train_df)
                task_loader = DataLoader(task_ds, batch_size=64, shuffle=False)
                all_logits = []
                with torch.no_grad():
                    for batch_data in task_loader:
                        bx = batch_data[0].to(self.device)
                        logs = self.model(bx)
                        all_logits.append(logs.cpu().numpy())
                all_logits = np.concatenate(all_logits, axis=0)
                train_df_copy = train_df.copy()
                train_df_copy['logit_0'] = all_logits[:, 0]
                train_df_copy['logit_1'] = all_logits[:, 1]
                self.buffer.add_samples(train_df_copy)
            else:
                self.buffer.add_samples(train_df)
                
        if self.method == "EWC":
            self.compute_fisher(train_df)
            
        if self.method == "LwF":
            self.prev_model = copy.deepcopy(self.model)
            self.prev_model.eval()
            
        # Fit OOD Detector on the latest representation
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
            for batch_x, batch_y, _ in loader:
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
            for batch_x, batch_y, _ in loader:
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
