import torch
import numpy as np
from sklearn.covariance import EmpiricalCovariance

class MahalanobisDetector:
    def __init__(self, threshold_percentile=95):
        self.class_means = {}
        self.class_cov_inv = None
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        
    def fit(self, features, labels):
        """
        features: numpy array of shape (N, D)
        labels: numpy array of shape (N,)
        """
        unique_labels = np.unique(labels)
        
        # Calculate class means
        for label in unique_labels:
            class_features = features[labels == label]
            self.class_means[label] = np.mean(class_features, axis=0)
            
        # Calculate shared covariance matrix (tied covariance)
        # Center the features
        centered_features = np.zeros_like(features)
        for label in unique_labels:
            idx = (labels == label)
            centered_features[idx] = features[idx] - self.class_means[label]
            
        # Use EmpiricalCovariance from sklearn
        cov_estimator = EmpiricalCovariance(assume_centered=True)
        cov_estimator.fit(centered_features)
        
        # Add small ridge to diagonal for numerical stability
        cov = cov_estimator.covariance_ + np.eye(features.shape[1]) * 1e-6
        self.class_cov_inv = np.linalg.inv(cov)
        
        # Calibrate threshold
        distances = self.score_samples(features, labels)
        self.threshold = np.percentile(distances, self.threshold_percentile)
        
    def score_samples(self, features, labels=None):
        """
        If labels is None, calculates the minimum distance across all classes.
        Returns: distances array
        """
        distances = []
        for i in range(len(features)):
            x = features[i]
            if not self.class_means:
                distances.append(0.0)
                continue
                
            if labels is not None:
                mean = self.class_means[labels[i]]
                diff = x - mean
                dist = np.dot(np.dot(diff.T, self.class_cov_inv), diff)
                distances.append(dist)
            else:
                min_dist = float('inf')
                for label, mean in self.class_means.items():
                    diff = x - mean
                    dist = np.dot(np.dot(diff.T, self.class_cov_inv), diff)
                    if dist < min_dist:
                        min_dist = dist
                distances.append(min_dist)
        return np.array(distances)
        
    def is_unknown(self, features):
        """
        Returns boolean array: True if Unknown, False if Known
        """
        if self.threshold is None:
            return np.zeros(len(features), dtype=bool)
        distances = self.score_samples(features)
        return distances > self.threshold
