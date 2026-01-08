import numpy as np
import torch
import torch.nn as nn
import cv2  # 基于opencv-python 4.11.0.86
from scipy.spatial import KDTree  # 兼容scipy 1.13.1
from sklearn.ensemble import IsolationForest  # 基于scikit-learn 1.5.1
import matplotlib.pyplot as plt  # 基于matplotlib 3.9.4
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  # 基于tqdm 4.67.1


class AdaptiveSphereTracker:
    """基于可变球体的手术器械轨迹追踪器，抗人体组织干扰"""
    
    def __init__(self, initial_position, min_radius=5, max_radius=30, 
                 feature_dim=128, tissue_threshold=0.3, confidence_threshold=0.6):
        """
        初始化追踪器
        :param initial_position: 初始球心位置 (x, y, z)
        :param min_radius/max_radius: 球体半径范围（自适应调整）
        :param feature_dim: 特征矩阵维度
        :param tissue_threshold: 组织干扰过滤阈值（低于此值视为干扰）
        :param confidence_threshold: 有效追踪置信度阈值
        """
        self.current_position = np.array(initial_position, dtype=np.float32)
        self.trajectory = [self.current_position.copy()]  # 轨迹记录
        self.radius = min_radius  # 初始半径
        self.min_radius = min_radius
        self.max_radius = max_radius
        
        # 特征提取器（用于球体区域特征编码）
        self.feature_encoder = nn.Sequential(
            nn.Linear(3 + feature_dim, 256),  # 3D坐标+特征值
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出特征置信度（区分器械与组织）
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 组织干扰检测器
        self.tissue_detector = IsolationForest(
            contamination=0.2,  # 假设20%为组织干扰
            random_state=42,
            n_estimators=100  # 兼容scikit-learn 1.5.1
        )
        
        # 阈值参数
        self.tissue_thresh = tissue_threshold
        self.conf_thresh = confidence_threshold
        self.feature_dim = feature_dim

    def _adjust_radius(self, feature_variation):
        """根据特征变化自适应调整球体半径"""
        # 特征变化大时增大半径（提高鲁棒性），变化小时减小半径（提高精度）
        new_radius = self.radius * (1 + 0.5 * feature_variation)
        return np.clip(new_radius, self.min_radius, self.max_radius)

    def _sample_sphere_points(self, position, radius, num_samples=100):
        """在球体表面均匀采样点"""
        phi = np.random.uniform(0, np.pi, num_samples)
        theta = np.random.uniform(0, 2*np.pi, num_samples)
        
        x = position[0] + radius * np.sin(phi) * np.cos(theta)
        y = position[1] + radius * np.sin(phi) * np.sin(theta)
        z = position[2] + radius * np.cos(phi)
        
        return np.column_stack([x, y, z])

    def _filter_tissue_points(self, points, features):
        """过滤球体中属于人体组织的干扰点"""
        # 特征拼接用于检测
        X = np.hstack([points, features])
        
        # 预测干扰点（-1表示组织干扰）
        is_tissue = self.tissue_detector.predict(X) == -1
        
        # 双重过滤：异常检测+特征阈值
        # 修正：中文变量名改为英文，修复缩进
        instrument_features = np.mean(features[~is_tissue], axis=0) if np.sum(~is_tissue) > 0 else np.zeros(self.feature_dim)
        tissue_score = np.linalg.norm(features - instrument_features, axis=1)
        is_valid = (~is_tissue) & (tissue_score < self.tissue_thresh)
        
        return points[is_valid], features[is_valid]

    def track_step(self, scene_features, num_samples=200):
        """
        单步追踪
        :param scene_features: 场景特征函数 (x,y,z) -> 特征向量
        :param num_samples: 球体采样点数
        :return: 新位置、置信度、移动距离
        """
        # 1. 在当前球体范围内采样
        sample_points = self._sample_sphere_points(
            self.current_position, 
            self.radius, 
            num_samples
        )
        
        # 2. 提取采样点的特征
        features = np.array([scene_features(p[0], p[1], p[2]) for p in sample_points])
        
        # 3. 过滤组织干扰点
        valid_points, valid_features = self._filter_tissue_points(
            sample_points, features
        )
        
        if len(valid_points) < num_samples * 0.3:  # 有效点太少时增大半径
            self.radius = min(self.radius * 1.5, self.max_radius)
            return self.current_position, 0.0, 0.0
        
        # 4. 计算特征变化率（用于调整半径）
        feature_variation = np.var(valid_features)
        self.radius = self._adjust_radius(feature_variation)
        
        # 5. 计算每个有效点到当前球心的向量
        vectors = valid_points - self.current_position
        
        # 6. 用特征编码器计算每个点的置信度（器械特征可能性）
        with torch.no_grad():
            device = next(self.feature_encoder.parameters()).device
            input_data = torch.tensor(
                np.hstack([vectors, valid_features]), 
                dtype=torch.float32
            ).to(device)
            confidences = self.feature_encoder(input_data).cpu().numpy().flatten()
        
        # 7. 加权平均计算新位置（置信度高的点权重更大）
        weights = np.clip(confidences, 0, 1)
        weights /= np.sum(weights) + 1e-8  # 归一化权重
        new_position = self.current_position + np.sum(vectors * weights[:, np.newaxis], axis=0)
        
        # 8. 计算追踪置信度
        track_confidence = np.mean(confidences)
        
        # 9. 计算移动距离
        move_distance = np.linalg.norm(new_position - self.current_position)
        
        # 10. 更新位置和轨迹（仅保留高置信度的更新）
        if track_confidence > self.conf_thresh:
            self.current_position = new_position
            self.trajectory.append(new_position.copy())
        
        return self.current_position, track_confidence, move_distance

    def train_tissue_detector(self, tissue_points, tissue_features):
        """训练组织干扰检测器"""
        X = np.hstack([tissue_points, tissue_features])
        # 修正：变量名错误（trajectory_detector -> tissue_detector）
        self.tissue_detector.fit(X)

    def train_feature_encoder(self, instrument_data, tissue_data, epochs=50):
        """训练特征编码器（区分器械和组织特征）"""
        # 器械数据标记为1，组织数据标记为0
        instrument_X = np.hstack([instrument_data['vectors'], instrument_data['features']])
        instrument_y = np.ones(len(instrument_X))
        
        tissue_X = np.hstack([tissue_data['vectors'], tissue_data['features']])
        tissue_y = np.zeros(len(tissue_X))
        
        # 合并训练数据
        X = np.vstack([instrument_X, tissue_X])
        y = np.hstack([instrument_y, tissue_y])
        
        # 转换为Tensor
        device = next(self.feature_encoder.parameters()).device
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)
        
        # 训练
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
        """计算轨迹总长度"""
        if len(self.trajectory) < 2:
            return 0.0
        total_length = 0.0
        for i in range(1, len(self.trajectory)):
            total_length += np.linalg.norm(self.trajectory[i] - self.trajectory[i-1])
        return total_length

    def visualize_trajectory(self):
        """可视化追踪轨迹"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取轨迹坐标
        trajectory_np = np.array(self.trajectory)
        ax.plot(
            trajectory_np[:, 0],
            trajectory_np[:, 1],
            trajectory_np[:, 2],
            'b-', linewidth=2, label='Trajectory'
        )
        
        # 标记起点和终点
        ax.scatter(
            trajectory_np[0, 0],
            trajectory_np[0, 1],
            trajectory_np[0, 2],
            c='green', s=100, label='Start'
        )
        ax.scatter(
            trajectory_np[-1, 0],
            trajectory_np[-1, 1],
            trajectory_np[-1, 2],
            c='red', s=100, label='End'
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Surgical Instrument Trajectory (Total Length: {self.get_trajectory_length():.2f})')
        ax.legend()
        plt.show()


# ------------------------------
# 示例使用
# ------------------------------
if __name__ == "__main__":
    # 1. 生成模拟场景特征函数（实际应用中替换为真实传感器数据）
    def create_scene_feature_function(actual_trajectory):
        """创建模拟场景特征函数，包含器械特征和组织干扰"""
        def scene_features(x, y, z):
            # 计算到实际轨迹的距离
            pos = np.array([x, y, z])
            dists = np.linalg.norm(actual_trajectory - pos, axis=1)
            min_dist = np.min(dists)
            
            # 器械特征（距离越近特征越明显）
            instrument_feature = np.exp(-0.5 * (min_dist **2)) * np.sin(np.linspace(0, 2*np.pi, 128))
            
            # 添加组织干扰特征
            tissue_noise = np.random.normal(0, 0.3, 128)
            tissue_mask = 1 / (1 + np.exp(-(min_dist - 15)))  # 距离远时干扰增强
            
            return instrument_feature + tissue_mask * tissue_noise
        
        return scene_features

    # 2. 生成模拟的实际轨迹（螺旋线）
    t = np.linspace(0, 10, 100)
    actual_trajectory = np.column_stack([
        50 * np.cos(t),
        50 * np.sin(t),
        t * 5  # 上升的螺旋
    ])

    # 3. 创建场景特征函数
    scene_features = create_scene_feature_function(actual_trajectory)

    # 4. 初始化追踪器（初始位置设为轨迹起点）
    tracker = AdaptiveSphereTracker(
        initial_position=actual_trajectory[0],
        min_radius=5,
        max_radius=25,
        feature_dim=128
    )

    # 5. 生成训练数据（用于区分器械和组织）
    def generate_training_data(num_samples=1000):
        # 器械数据
        instrument_points = []
        for pos in actual_trajectory[::5]:  # 每隔5步取一个点
            samples = tracker._sample_sphere_points(pos, 10, num_samples//10)
            instrument_points.extend(samples)
        instrument_points = np.array(instrument_points)
        instrument_vectors = instrument_points - actual_trajectory[0]
        instrument_features = np.array([scene_features(p[0], p[1], p[2]) for p in instrument_points])
        
        # 组织数据（远离轨迹的点）
        tissue_points = np.random.uniform(-100, 100, (num_samples, 3))
        tissue_vectors = tissue_points - actual_trajectory[0]
        tissue_features = np.array([scene_features(p[0], p[1], p[2]) for p in tissue_points])
        
        return {
            'instrument': {'vectors': instrument_vectors, 'features': instrument_features},
            'tissue': {'vectors': tissue_vectors, 'features': tissue_features}
        }

    # 生成并训练模型
    training_data = generate_training_data()
    tracker.train_feature_encoder(
        training_data['instrument'],
        training_data['tissue'],
        epochs=30
    )
    tracker.train_tissue_detector(
        training_data['tissue']['vectors'],
        training_data['tissue']['features']
    )

    # 6. 执行追踪
    print("开始追踪器械轨迹...")
    for _ in tqdm(range(80)):  # 追踪80步
        new_pos, conf, dist = tracker.track_step(scene_features)
        
    # 7. 输出结果
    print(f"\n追踪完成！")
    print(f"轨迹总长度: {tracker.get_trajectory_length():.2f} 单位")
    print(f"最终位置: {tracker.current_position}")
    print(f"实际终点: {actual_trajectory[-1]}")
    print(f"终点误差: {np.linalg.norm(tracker.current_position - actual_trajectory[-1]):.2f} 单位")

    # 8. 可视化轨迹
    tracker.visualize_trajectory()
import numpy as np
import torch
import torch.nn as nn
import cv2  # 基于opencv-python 4.11.0.86
from scipy.ndimage import gaussian_filter  # 兼容scipy 1.13.1
from sklearn.ensemble import RandomForestClassifier  # 基于scikit-learn 1.5.1
import matplotlib.pyplot as plt  # 基于matplotlib 3.9.4
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm  # 基于tqdm 4.67.1


class GlareInstrumentAnalyzer:
    """反光手术器械逆向投影分析系统，通过组织矩阵变化逆推器械位置"""
    
    def __init__(self, tissue_reflectance_model=None, 
                 low_threshold=0.2, high_threshold=0.7):
        """
        初始化分析器
        :param tissue_reflectance_model: 人体组织反射模型
        :param low_threshold: 低置信度阈值（过滤噪声）
        :param high_threshold: 高置信度阈值（确认有效信号）
        """
        self.tissue_model = tissue_reflectance_model or self._build_default_tissue_model()
        self.low_thresh = low_threshold
        self.high_thresh = high_threshold
        
        # 逆渲染网络（从观察到的反射变化预测光源/器械位置）
        self.inverse_renderer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 64 * 64, 512),  # 假设输入256x256图像
            nn.ReLU(),
            nn.Linear(512, 3)  # 输出3D坐标 (x,y,z)
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 干扰分类器（区分器械反光与组织自然变化）
        self.interference_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42
        )
        
        # 记录分析结果
        self.estimated_positions = []
        self.confidence_map = None

    def _build_default_tissue_model(self):
        """构建默认人体组织反射模型（朗伯体假设）"""
        return {
            'albedo': 0.3,  # 基础反射率
            'roughness': 0.8,  # 粗糙度（越高越不反光）
            'scattering_coef': 0.15,  # 散射系数
            'absorption_coef': 0.05   # 吸收系数
        }

    def scan_region_intensity_variation(self, scene_sequence):
        """
        扫描区域强度变化矩阵，核心逆向分析基础
        :param scene_sequence: 时间序列图像 (T, H, W, 3)
        :return: 强度变化方差图、时间梯度图
        """
        T, H, W, C = scene_sequence.shape
        
        # 计算每个像素的强度方差（变化剧烈程度）
        intensity_variance_map = np.var(scene_sequence, axis=0).mean(axis=-1)
        
        # 计算时间梯度（变化率）
        time_gradient = np.zeros((H, W), dtype=np.float32)
        for t in range(1, T):
            time_gradient += np.abs(
                scene_sequence[t].mean(axis=-1) - scene_sequence[t-1].mean(axis=-1)
            )
        time_gradient /= (T - 1)  # 归一化
        
        # 平滑处理（减少噪声影响）
        intensity_variance_map = gaussian_filter(intensity_variance_map, sigma=1.5)
        time_gradient = gaussian_filter(time_gradient, sigma=1.5)
        
        return intensity_variance_map, time_gradient

    def extract_reflection_features(self, variance_map, gradient_map, window_size=7):
        """提取反光特征（用于区分器械反光与组织干扰）"""
        H, W = variance_map.shape
        features = []
        positions = []
        
        # 滑动窗口提取特征
        for i in range(window_size//2, H - window_size//2):
            for j in range(window_size//2, W - window_size//2):
                # 窗口区域
                window_var = variance_map[
                    i-window_size//2:i+window_size//2+1,
                    j-window_size//2:j+window_size//2+1
                ]
                window_grad = gradient_map[
                    i-window_size//2:i+window_size//2+1,
                    j-window_size//2:j+window_size//2+1
                ]
                
                # 提取统计特征
                feat = [
                    np.mean(window_var), np.std(window_var),
                    np.max(window_var), np.min(window_var),
                    np.mean(window_grad), np.std(window_grad),
                    np.max(window_grad),
                    # 反光区域通常有明显的峰值特征
                    np.sum(window_var > np.percentile(window_var, 90))
                ]
                features.append(feat)
                positions.append((i, j))
        
        return np.array(features), np.array(positions)

    def inverse_glare_analysis(self, scene_sequence):
        """
        核心算法：通过组织矩阵变化逆推反光器械位置
        :param scene_sequence: 时间序列图像 (T, H, W, 3)
        :return: 估计的器械3D位置序列、置信度
        """
        T, H, W, C = scene_sequence.shape
        estimated_positions = []
        
        # 1. 计算区域强度变化矩阵
        variance_map, gradient_map = self.scan_region_intensity_variation(scene_sequence)
        
        # 2. 提取反光特征并分类（过滤组织干扰）
        features, positions = self.extract_reflection_features(variance_map, gradient_map)
        with torch.no_grad():
            # 预测每个窗口是否为器械反光
            is_instrument = self.interference_classifier.predict(features) == 1
            confidence = self.interference_classifier.predict_proba(features)[:, 1]
        
        # 构建置信度图
        self.confidence_map = np.zeros((H, W), dtype=np.float32)
        for (i, j), conf, is_inst in zip(positions, confidence, is_instrument):
            if is_inst and conf > self.low_thresh:
                self.confidence_map[i, j] = conf
        
        # 3. 时间序列追踪
        for t in tqdm(range(T), desc="逆向投影分析"):
            # 提取当前帧特征
            frame = scene_sequence[t].astype(np.float32) / 255.0
            frame_tensor = torch.tensor(
                frame.transpose(2, 0, 1)[np.newaxis, ...],
                dtype=torch.float32
            ).to(next(self.inverse_renderer.parameters()).device)
            
            # 逆渲染预测3D位置
            with torch.no_grad():
                pred_3d = self.inverse_renderer(frame_tensor).cpu().numpy()[0]
            
            # 结合置信度过滤
            h, w = int(pred_3d[1]), int(pred_3d[0])  # 投影到2D坐标
            if 0 <= h < H and 0 <= w < W:
                conf = self.confidence_map[h, w]
                if conf > self.high_thresh:
                    estimated_positions.append(pred_3d)
                elif conf > self.low_thresh:
                    # 低置信度时进行平滑处理
                    if estimated_positions:
                        # 取前一帧位置和当前预测的加权平均
                        smoothed = 0.7 * estimated_positions[-1] + 0.3 * pred_3d
                        estimated_positions.append(smoothed)
                    else:
                        estimated_positions.append(pred_3d)
        
        self.estimated_positions = np.array(estimated_positions)
        return self.estimated_positions, np.mean(confidence[confidence > self.low_thresh])

    def train_model(self, X_train, y_train, frame_samples):
        """
        训练干扰分类器和逆渲染网络
        :param X_train: 特征训练集
        :param y_train: 标签（1:器械反光, 0:组织干扰）
        :param frame_samples: 用于训练逆渲染的帧样本
        """
        # 训练干扰分类器
        self.interference_classifier.fit(X_train, y_train)
        
        # 训练逆渲染网络
        optimizer = torch.optim.Adam(self.inverse_renderer.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        device = next(self.inverse_renderer.parameters()).device
        
        # 准备训练数据
        X_frames = torch.tensor(
            np.transpose(frame_samples['frames'], (0, 3, 1, 2)),
            dtype=torch.float32
        ).to(device) / 255.0
        y_positions = torch.tensor(
            frame_samples['positions'],
            dtype=torch.float32
        ).to(device)
        
        # 训练循环
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = self.inverse_renderer(X_frames)
            loss = criterion(outputs, y_positions)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"逆渲染网络训练 Epoch {epoch+1}/50, Loss: {loss.item():.6f}")

    def visualize_analysis(self, original_frame):
        """可视化分析结果：强度变化、置信度和预测轨迹"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 原始图像
        axes[0, 0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("原始手术场景")
        axes[0, 0].axis('off')
        
        # 2. 强度变化方差图
        variance_map, _ = self.scan_region_intensity_variation(
            np.expand_dims(original_frame, axis=0)
        )
        im = axes[0, 1].imshow(variance_map, cmap='jet')
        axes[0, 1].set_title("强度变化方差图（高亮反光区域）")
        axes[0, 1].axis('off')
        divider = make_axes_locatable(axes[0, 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # 3. 置信度图
        if self.confidence_map is not None:
            im = axes[1, 0].imshow(self.confidence_map, cmap='viridis')
            axes[1, 0].set_title(f"器械反光置信度（阈值: {self.high_thresh}）")
            axes[1, 0].axis('off')
            divider = make_axes_locatable(axes[1, 0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        
        # 4. 3D轨迹
        if len(self.estimated_positions) > 1:
            ax3d = fig.add_subplot(224, projection='3d')
            ax3d.plot(
                self.estimated_positions[:, 0],
                self.estimated_positions[:, 1],
                self.estimated_positions[:, 2],
                'r-', linewidth=2
            )
            ax3d.set_xlabel('X')
            ax3d.set_ylabel('Y')
            ax3d.set_zlabel('Z')
            ax3d.set_title("逆推的器械3D轨迹")
        
        plt.tight_layout()
        plt.show()


# ------------------------------
# 示例使用
# ------------------------------
if __name__ == "__main__":
    # 1. 生成模拟手术场景数据（含反光器械）
    def generate_surgical_scene(T=50, H=256, W=256):
        """生成包含反光器械的手术场景序列"""
        # 基础组织背景
        scene_sequence = np.zeros((T, H, W, 3), dtype=np.uint8)
        for t in range(T):
            # 模拟人体组织（带自然纹理）
            tissue = np.random.normal(150, 30, (H, W, 3)).astype(np.uint8)
            tissue = gaussian_filter(tissue, sigma=2)
            scene_sequence[t] = tissue
            
            # 模拟反光器械（随时间移动的光斑）
            x = int(H/2 + 50 * np.sin(t * 0.2))  # 正弦轨迹
            y = int(W/2 + 50 * np.cos(t * 0.2))
            radius = 5 + 3 * np.sin(t * 0.5)  # 反光强度变化
            
            # 绘制反光区域（高强度光斑）
            cv2.circle(
                scene_sequence[t], 
                (y, x), 
                int(radius), 
                (255, 255, 255),  # 白色反光
                -1
            )
            # 添加高光扩散
            cv2.circle(
                scene_sequence[t], 
                (y, x), 
                int(radius * 2), 
                (200, 200, 255),  # 暖色反光边缘
                -1,
                cv2.LINE_AA
            )
            scene_sequence[t] = np.clip(scene_sequence[t], 0, 255)
        
        # 生成真实轨迹（用于训练）
        true_positions = np.array([
            [y, x, 50 + 10 * np.sin(t * 0.3)]  # Z轴位置模拟
            for t, x, y in zip(range(T), 
                              [int(H/2 + 50 * np.sin(t * 0.2)) for t in range(T)],
                              [int(W/2 + 50 * np.cos(t * 0.2)) for t in range(T)])
        ])
        
        return scene_sequence, true_positions

    # 2. 生成训练数据
    scene_sequence, true_positions = generate_surgical_scene(T=50)
    variance_map, gradient_map = GlareInstrumentAnalyzer().scan_region_intensity_variation(scene_sequence)
    features, _ = GlareInstrumentAnalyzer().extract_reflection_features(variance_map, gradient_map)
    
    # 创建训练标签（靠近真实轨迹的为1，其他为0）
    y_train = np.zeros(len(features))
    for i, (x, y) in enumerate(true_positions[:, :2].astype(int)):
        # 简单标签规则：真实位置附近的特征标记为1
        dist = np.linalg.norm(features[:, :2] - np.array([x, y]), axis=1)
        y_train[dist < 20] = 1

    # 3. 初始化分析器
    analyzer = GlareInstrumentAnalyzer(
        low_threshold=0.25,
        high_threshold=0.65
    )

    # 4. 训练模型
    print("开始训练模型...")
    analyzer.train_model(
        X_train=features,
        y_train=y_train,
        frame_samples={
            'frames': scene_sequence[::5],  # 每隔5帧取一个样本
            'positions': true_positions[::5]
        }
    )

    # 5. 执行逆向投影分析
    print("\n执行反光器械逆向分析...")
    estimated_pos, confidence = analyzer.inverse_glare_analysis(scene_sequence)

    # 6. 输出结果
    print(f"\n分析完成！平均置信度: {confidence:.2f}")
    print(f"轨迹长度: {len(estimated_pos)} 帧")
    if len(estimated_pos) > 0 and len(true_positions) > 0:
        avg_error = np.mean(np.linalg.norm(estimated_pos - true_positions, axis=1))
        print(f"平均位置误差: {avg_error:.2f} 像素")

    # 7. 可视化分析结果
    analyzer.visualize_analysis(scene_sequence[25])  # 可视化第25帧
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2  # 基于opencv-python 4.11.0.86
from scipy.spatial.transform import Rotation as R  # 兼容scipy 1.13.1
from sklearn.model_selection import train_test_split  # 基于scikit-learn 1.5.1
import matplotlib.pyplot as plt  # 基于matplotlib 3.9.4
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  # 基于tqdm 4.67.1


class CubeProjectionAttention(nn.Module):
    """正方体投影+注意力机制的器械识别模型"""
    
    def __init__(self, proj_size=64, num_classes=5, 
                 low_threshold=0.2, high_threshold=0.7):
        """
        初始化模型
        :param proj_size: 每个面的投影尺寸
        :param num_classes: 器械类别数
        :param low_threshold: 低置信度阈值（过滤干扰）
        :param high_threshold: 高置信度阈值（确认有效特征）
        """
        super().__init__()
        self.proj_size = proj_size
        self.low_thresh = low_threshold
        self.high_thresh = high_threshold
        
        # 1. 六面投影特征提取器
        self.proj_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),  # 6个投影面作为输入通道
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # 2. 自注意力机制（突出重要投影区域）
        self.attention = nn.MultiheadAttention(
            embed_dim=128 * 8,  # 适配特征维度
            num_heads=4,
            batch_first=True
        )
        
        # 3. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
        
        # 干扰过滤模型
        self.interference_filter = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def normalize_to_cube(point_cloud, cube_size=1.0):
        """将点云归一化到单位正方体中"""
        # 找到点云边界
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
        
        # 平移到原点
        centered = point_cloud - min_coords
        
        # 缩放至单位正方体
        scale = cube_size / np.max(max_coords - min_coords)
        normalized = centered * scale
        
        return normalized

    def project_to_cube_faces(self, point_cloud):
        """
        将点云投影到正方体的6个面
        投影面: front(+z), back(-z), left(-x), right(+x), top(+y), bottom(-y)
        """
        # 确保点云已归一化到[0,1]范围
        normalized_pc = self.normalize_to_cube(point_cloud)
        
        # 初始化6个投影面
        projections = np.zeros((6, self.proj_size, self.proj_size), dtype=np.float32)
        
        # 1. Front face (+z, 投影到xy平面)
        z_values = normalized_pc[:, 2]
        front_idx = np.argsort(z_values)[-self.proj_size**2:]  # 取z最大的点
        for i, j in front_idx:
            x = int(normalized_pc[i, 0] * (self.proj_size - 1))
            y = int(normalized_pc[i, 1] * (self.proj_size - 1))
            projections[0, y, x] += 1
        
        # 2. Back face (-z, 投影到xy平面)
        back_idx = np.argsort(z_values)[:self.proj_size**2]  # 取z最小的点
        for i in back_idx:
            x = int(normalized_pc[i, 0] * (self.proj_size - 1))
            y = int(normalized_pc[i, 1] * (self.proj_size - 1))
            projections[1, y, x] += 1
        
        # 3. Left face (-x, 投影到yz平面)
        x_values = normalized_pc[:, 0]
        left_idx = np.argsort(x_values)[:self.proj_size**2]  # 取x最小的点
        for i in left_idx:
            y = int(normalized_pc[i, 1] * (self.proj_size - 1))
            z = int(normalized_pc[i, 2] * (self.proj_size - 1))
            projections[2, z, y] += 1
        
        # 4. Right face (+x, 投影到yz平面)
        right_idx = np.argsort(x_values)[-self.proj_size**2:]  # 取x最大的点
        for i in right_idx:
            y = int(normalized_pc[i, 1] * (self.proj_size - 1))
            z = int(normalized_pc[i, 2] * (self.proj_size - 1))
            projections[3, z, y] += 1
        
        # 5. Top face (+y, 投影到xz平面)
        y_values = normalized_pc[:, 1]
        top_idx = np.argsort(y_values)[-self.proj_size**2:]  # 取y最大的点
        for i in top_idx:
            x = int(normalized_pc[i, 0] * (self.proj_size - 1))
            z = int(normalized_pc[i, 2] * (self.proj_size - 1))
            projections[4, z, x] += 1
        
        # 6. Bottom face (-y, 投影到xz平面)
        bottom_idx = np.argsort(y_values)[:self.proj_size**2]  # 取y最小的点
        for i in bottom_idx:
            x = int(normalized_pc[i, 0] * (self.proj_size - 1))
            z = int(normalized_pc[i, 2] * (self.proj_size - 1))
            projections[5, z, x] += 1
        
        # 归一化每个投影面
        for i in range(6):
            if np.max(projections[i]) > 0:
                projections[i] /= np.max(projections[i])
        
        return projections

    def forward(self, point_clouds):
        """
        前向传播
        :param point_clouds: 输入点云列表 (B, N, 3)
        :return: 分类概率、注意力权重、干扰分数
        """
        B = len(point_clouds)
        device = next(self.parameters()).device
        
        # 1. 生成所有点云的六面投影
        proj_list = []
        for pc in point_clouds:
            proj = self.project_to_cube_faces(pc)
            proj_list.append(proj)
        
        # 转换为张量 (B, 6, H, W)
        proj_tensor = torch.tensor(
            np.stack(proj_list), 
            dtype=torch.float32
        ).to(device)
        
        # 2. 提取投影特征
        feat = self.proj_encoder(proj_tensor)  # (B, 128, 8, 8)
        
        # 3. 应用注意力机制
        feat_flat = feat.view(B, 128, -1).permute(0, 2, 1)  # (B, 64, 128)
        attn_output, attn_weights = self.attention(feat_flat, feat_flat, feat_flat)
        
        # 4. 分类与干扰检测
        global_feat = attn_output.reshape(B, -1)  # 展平特征
        class_probs = self.classifier(global_feat)
        interference_score = self.interference_filter(global_feat)  # 0-1，越高越可能是干扰
        
        return class_probs, attn_weights, interference_score

    def detect_instrument(self, point_cloud):
        """
        检测器械并过滤干扰
        :param point_cloud: 输入点云 (N, 3)
        :return: 类别、置信度、是否为干扰
        """
        self.eval()
        with torch.no_grad():
            # 转换为批处理格式
            pc_tensor = [point_cloud]
            class_probs, _, interference_score = self.forward(pc_tensor)
            
            # 获取最高置信度类别
            max_prob, class_idx = torch.max(class_probs[0], dim=0)
            max_prob = max_prob.item()
            class_idx = class_idx.item()
            
            # 判断是否为干扰
            is_interference = (interference_score[0].item() > self.high_thresh) or \
                             (max_prob < self.low_thresh)
        
        return class_idx, max_prob, is_interference

    def visualize_projections(self, point_cloud, title="六面投影特征"):
        """可视化六面投影结果"""
        projections = self.project_to_cube_faces(point_cloud)
        faces = ["Front (+z)", "Back (-z)", "Left (-x)", "Right (+x)", "Top (+y)", "Bottom (-y)"]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for i, ax in enumerate(axes.flatten()):
            im = ax.imshow(projections[i], cmap='viridis')
            ax.set_title(faces[i])
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


# ------------------------------
# 模型训练与示例使用
# ------------------------------
def generate_instrument_point_cloud(num_points=1000, instrument_type=0, add_noise=True):
    """生成模拟的手术器械点云数据"""
    # 基础形状（不同器械类型有不同基础形状）
    if instrument_type == 0:  # 手术刀（细长形）
        theta = np.linspace(0, np.pi, num_points)
        x = np.cos(theta) * 0.3
        y = np.sin(theta) * 0.05
        z = np.linspace(0, 1, num_points)
    elif instrument_type == 1:  # 钳子（分叉形）
        t = np.linspace(0, 1, num_points)
        x = np.where(t < 0.7, 0.05 * np.sin(t*10), 
                    0.1 * np.sin(t*10 + np.pi/2) * np.sign(np.sin(t*5)))
        y = np.where(t < 0.7, 0.05 * np.cos(t*10), 
                    0.1 * np.cos(t*10 + np.pi/2))
        z = t
    else:  # 镊子（对称形）
        t = np.linspace(0, 1, num_points)
        x = 0.05 * np.sin(t*20)
        y = np.where(t < 0.6, 0.05 * np.cos(t*20), 
                    0.05 * np.cos(t*20) * (2 - t*2))
        z = t
    
    point_cloud = np.column_stack([x, y, z])
    
    # 随机旋转（模拟任意摆放）
    rot = R.random()
    point_cloud = rot.apply(point_cloud)
    
    # 添加噪声和组织干扰
    if add_noise:
        # 高斯噪声
        point_cloud += np.random.normal(0, 0.02, point_cloud.shape)
        # 模拟组织干扰点
        tissue_points = np.random.uniform(-0.5, 0.5, (int(num_points*0.2), 3))
        point_cloud = np.vstack([point_cloud, tissue_points])
    
    return point_cloud


if __name__ == "__main__":
    # 1. 生成训练数据
    print("生成模拟训练数据...")
    num_samples = 200
    num_classes = 3
    point_clouds = []
    labels = []
    
    for i in tqdm(range(num_samples)):
        instr_type = np.random.randint(0, num_classes)
        pc = generate_instrument_point_cloud(
            num_points=1000,
            instrument_type=instr_type,
            add_noise=True
        )
        point_clouds.append(pc)
        labels.append(instr_type)
    
    # 划分训练集和测试集
    train_pc, test_pc, train_labels, test_labels = train_test_split(
        point_clouds, labels, test_size=0.2, random_state=42
    )
    
    # 2. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CubeProjectionAttention(
        proj_size=64,
        num_classes=num_classes,
        low_threshold=0.3,
        high_threshold=0.6
    ).to(device)
    
    # 3. 训练模型
    print("\n开始训练模型...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 30
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        
        for pc, label in zip(train_pc, train_labels):
            optimizer.zero_grad()
            # 转换为批处理格式
            inputs = [pc]
            outputs, _, _ = model(inputs)
            loss = criterion(outputs, torch.tensor([label], dtype=torch.long).to(device))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1).item()
            if pred == label:
                correct += 1
        
        # 计算准确率
        train_acc = correct / len(train_pc)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_pc):.4f}, Acc: {train_acc:.4f}")
    
    # 4. 测试模型
    print("\n测试模型性能...")
    model.eval()
    correct = 0
    interference_count = 0
    
    with torch.no_grad():
        for pc, label in zip(test_pc, test_labels):
            class_idx, prob, is_interference = model.detect_instrument(pc)
            if not is_interference:
                if class_idx == label:
                    correct += 1
            else:
                interference_count += 1
    
    test_acc = correct / (len(test_pc) - interference_count) if (len(test_pc) - interference_count) > 0 else 0
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"干扰识别率: {interference_count/len(test_pc):.4f}")
    
    # 5. 可视化示例
    print("\n可视化投影结果...")
    sample_pc = generate_instrument_point_cloud(instrument_type=1, add_noise=True)
    model.visualize_projections(sample_pc, title="手术钳六面投影特征")
    
    # 6. 单样本检测示例
    class_names = ["手术刀", "手术钳", "镊子"]
    class_idx, prob, is_interference = model.detect_instrument(sample_pc)
    if is_interference:
        print("\n检测结果: 检测到干扰（可能为人体组织）")
    else:
        print(f"\n检测结果: {class_names[class_idx]} (置信度: {prob:.2f})")
# python_backend/routes/analysis.py（文件末尾）
from utils.tools import get_current_timestamp, create_dir_if_not_exists
from .detection import detect_instruments  # 依赖识别模块

def track_instrument_trajectory(frame_sequence):
    """
    追踪器械轨迹（整合AdaptiveSphereTracker）
    :param frame_sequence: 连续帧数据（base64列表）
    :return: 轨迹数据
    """
    from ..utils.tools import base64_to_cv2
    from .detection import INSTRUMENT_CLASSES  # 从同目录的detection.py导入
    from .tracker import AdaptiveSphereTracker  # 从同目录的tracker.py导入追踪器

    # 初始化追踪器（以第一帧的第一个器械为起点）
    first_frame = base64_to_cv2(frame_sequence[0])
    first_detections = detect_instruments(first_frame)
    if not first_detections:
        return {'error': '第一帧未检测到器械'}
    
    initial_position = first_detections[0]['position']  # (x,y) -> 转为3D (x,y,0)
    initial_position_3d = [initial_position['x'], initial_position['y'], 0.0]
    tracker = AdaptiveSphereTracker(initial_position=initial_position_3d)

    # 逐帧追踪
    trajectory = []
    for frame_str in frame_sequence:
        frame = base64_to_cv2(frame_str)
        h, w = frame.shape[:2]
        
        # 提取场景特征（简化：用随机向量模拟，实际需替换为真实特征提取）
        def scene_features(x, y, z):
            return np.random.rand(128)  # 模拟128维特征向量
        
        # 单步追踪
        new_pos, conf, dist = tracker.track_step(scene_features)
        trajectory.append({
            'position': new_pos.tolist(),
            'confidence': float(conf),
            'timestamp': get_current_timestamp()
        })
    
    return {
        'trajectory': trajectory,
        'total_length': tracker.get_trajectory_length()
    }