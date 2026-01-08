# python_backend/routes/tracker.py
import numpy as np
import torch
import torch.nn as nn
import cv2
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

class AdaptiveSphereTracker:
    """基于可变球体的手术器械轨迹追踪器，抗人体组织干扰"""
    def __init__(self, initial_position, min_radius=5, max_radius=30, 
                 feature_dim=128, tissue_threshold=0.3, confidence_threshold=0.6):
        self.current_position = np.array(initial_position, dtype=np.float32)
        self.trajectory = [self.current_position.copy()]
        self.radius = min_radius
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # 特征提取器
        self.feature_encoder = nn.Sequential(
            nn.Linear(3 + feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 组织干扰检测器
        self.tissue_detector = IsolationForest(
            contamination=0.2,
            random_state=42,
            n_estimators=100
        )
        
        self.tissue_thresh = tissue_threshold
        self.conf_thresh = confidence_threshold
        self.feature_dim = feature_dim

    def _adjust_radius(self, feature_variation):
        new_radius = self.radius * (1 + 0.5 * feature_variation)
        return np.clip(new_radius, self.min_radius, self.max_radius)

    def _sample_sphere_points(self, position, radius, num_samples=100):
        phi = np.random.uniform(0, np.pi, num_samples)
        theta = np.random.uniform(0, 2*np.pi, num_samples)
        x = position[0] + radius * np.sin(phi) * np.cos(theta)
        y = position[1] + radius * np.sin(phi) * np.sin(theta)
        z = position[2] + radius * np.cos(phi)
        return np.column_stack([x, y, z])

    def _filter_tissue_points(self, points, features):
        X = np.hstack([points, features])
        is_tissue = self.tissue_detector.predict(X) == -1
        instrument_features = np.mean(features[~is_tissue], axis=0) if np.sum(~is_tissue) > 0 else np.zeros(self.feature_dim)
        tissue_score = np.linalg.norm(features - instrument_features, axis=1)
        is_valid = (~is_tissue) & (tissue_score < self.tissue_thresh)
        return points[is_valid], features[is_valid]

    def track_step(self, scene_features, num_samples=200):
        sample_points = self._sample_sphere_points(self.current_position, self.radius, num_samples)
        features = np.array([scene_features(p[0], p[1], p[2]) for p in sample_points])
        valid_points, valid_features = self._filter_tissue_points(sample_points, features)
        
        if len(valid_points) < num_samples * 0.3:
            self.radius = min(self.radius * 1.5, self.max_radius)
            return self.current_position, 0.0, 0.0
        
        feature_variation = np.var(valid_features)
        self.radius = self._adjust_radius(feature_variation)
        vectors = valid_points - self.current_position
        
        with torch.no_grad():
            device = next(self.feature_encoder.parameters()).device
            input_data = torch.tensor(np.hstack([vectors, valid_features]), dtype=torch.float32).to(device)
            confidences = self.feature_encoder(input_data).cpu().numpy().flatten()
        
        weights = np.clip(confidences, 0, 1)
        weights /= np.sum(weights) + 1e-8
        new_position = self.current_position + np.sum(vectors * weights[:, np.newaxis], axis=0)
        track_confidence = np.mean(confidences)
        move_distance = np.linalg.norm(new_position - self.current_position)
        
        if track_confidence > self.conf_thresh:
            self.current_position = new_position
            self.trajectory.append(new_position.copy())
        
        return self.current_position, track_confidence, move_distance

    def train_tissue_detector(self, tissue_points, tissue_features):
        X = np.hstack([tissue_points, tissue_features])
        self.tissue_detector.fit(X)

    def train_feature_encoder(self, instrument_data, tissue_data, epochs=50):
        instrument_X = np.hstack([instrument_data['vectors'], instrument_data['features']])
        instrument_y = np.ones(len(instrument_X))
        tissue_X = np.hstack([tissue_data['vectors'], tissue_data['features']])
        tissue_y = np.zeros(len(tissue_X))
        
        X = np.vstack([instrument_X, tissue_X])
        y = np.hstack([instrument_y, tissue_y])
        
        device = next(self.feature_encoder.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)
        
        optimizer = torch.optim.Adam(self.feature_encoder.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.feature_encoder(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    pred = torch.sigmoid(outputs) > 0.5
                    acc = (pred.float() == y_tensor).float().mean()
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

    def get_trajectory_length(self):
        if len(self.trajectory) < 2:
            return 0.0
        total_length = 0.0
        for i in range(1, len(self.trajectory)):
            total_length += np.linalg.norm(self.trajectory[i] - self.trajectory[i-1])
        return total_length

    def visualize_trajectory(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        trajectory_np = np.array(self.trajectory)
        ax.plot(trajectory_np[:, 0], trajectory_np[:, 1], trajectory_np[:, 2], 'b-', linewidth=2, label='Trajectory')
        ax.scatter(trajectory_np[0, 0], trajectory_np[0, 1], trajectory_np[0, 2], c='green', s=100, label='Start')
        ax.scatter(trajectory_np[-1, 0], trajectory_np[-1, 1], trajectory_np[-1, 2], c='red', s=100, label='End')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Surgical Instrument Trajectory (Total Length: {self.get_trajectory_length():.2f})')
        ax.legend()
        plt.show()