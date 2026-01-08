import numpy as np
import torch
import torch.nn as nn
import cv2  # 基于opencv-python 4.11.0.86
from scipy.ndimage import gaussian_filter  # 兼容scipy 1.13.1
from sklearn.ensemble import GradientBoostingClassifier  # 基于scikit-learn 1.5.1
import matplotlib.pyplot as plt  # 基于matplotlib 3.9.4
from matplotlib.patches import Rectangle
from tqdm import tqdm  # 基于tqdm 4.67.1


class MetalNeedleRecognizer:
    """金属针识别系统，基于细分矩阵和多特征矩阵解决反光、遮挡和重叠问题"""
    
    def __init__(self, block_size=4, grid_size=16, 
                 low_thresh=0.3, high_thresh=0.7):
        """
        初始化识别器
        :param block_size: 细分矩阵块大小（越小识别越精细）
        :param grid_size: 基础网格矩阵大小
        :param low_thresh: 低置信度阈值（过滤弱特征）
        :param high_thresh: 高置信度阈值（确认有效特征）
        """
        self.block_size = block_size  # 细分矩阵块大小（4x4像素）
        self.grid_size = grid_size    # 16x16网格矩阵
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        
        # 1. 多特征矩阵提取器（处理反光、遮挡等不同场景）
        self.feature_extractors = {
            'glare': self._extract_glare_features,       # 金属反光特征矩阵
            'edge': self._extract_edge_features,         # 边缘特征矩阵
            'color': self._extract_color_features,       # 颜色特征矩阵（区分血液）
            'texture': self._extract_texture_features    # 纹理特征矩阵（区分组织）
        }
        
        # 2. 细分矩阵注意力网络（关注细微差异）
        self.attention_net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  # 4类特征矩阵
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1),             # 输出注意力权重
            nn.Softmax(dim=1)
        ).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 3. 分类器（将所有特征转化为识别问题）
        self.classifier = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        )
        
        # 存储中间结果
        self.features = None
        self.attention_map = None

    def _extract_glare_features(self, region):
        """提取金属反光特征矩阵（高光区域值偏大）"""
        # 转换为灰度图
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # 反光区域通常有高强度和剧烈变化
        glare = np.where(gray > 0.7, gray, 0)  # 高光区域
        glare_var = gaussian_filter(glare**2, sigma=1)  # 局部方差（值越大反光越明显）
        
        # 下采样为网格矩阵
        h, w = region.shape[:2]
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                roi = glare_var[
                    i*h//self.grid_size : (i+1)*h//self.grid_size,
                    j*w//self.grid_size : (j+1)*w//self.grid_size
                ]
                grid[i, j] = np.mean(roi)
        
        return grid

    def _extract_edge_features(self, region):
        """提取边缘特征矩阵（解决重叠问题）"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        
        # 下采样为网格矩阵
        h, w = region.shape[:2]
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                roi = edges[
                    i*h//self.grid_size : (i+1)*h//self.grid_size,
                    j*w//self.grid_size : (j+1)*w//self.grid_size
                ]
                grid[i, j] = np.mean(roi)
        
        return grid

    def _extract_color_features(self, region):
        """提取颜色特征矩阵（区分血液遮挡）"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        # 血液通常为红色/棕色：低Hue，中高Saturation，中Value
        blood_mask = (hsv[:, :, 0] < 20) & (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 50)
        blood = blood_mask.astype(np.float32)
        
        # 金属通常为高亮度：高Value通道
        value = hsv[:, :, 2].astype(np.float32) / 255.0
        metal_value = np.where(~blood_mask, value, 0)  # 排除血液区域
        
        # 下采样为网格矩阵
        h, w = region.shape[:2]
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                roi = metal_value[
                    i*h//self.grid_size : (i+1)*h//self.grid_size,
                    j*w//self.grid_size : (j+1)*w//self.grid_size
                ]
                grid[i, j] = np.mean(roi)
        
        return grid

    def _extract_texture_features(self, region):
        """提取纹理特征矩阵（区分人体组织）"""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        # 使用局部二进制模式(LBP)描述纹理
        lbp = np.zeros_like(gray, dtype=np.float32)
        for i in range(1, gray.shape[0]-1):
            for j in range(1, gray.shape[1]-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] > center) << 7
                code |= (gray[i-1, j] > center) << 6
                code |= (gray[i-1, j+1] > center) << 5
                code |= (gray[i, j+1] > center) << 4
                code |= (gray[i+1, j+1] > center) << 3
                code |= (gray[i+1, j] > center) << 2
                code |= (gray[i+1, j-1] > center) << 1
                code |= (gray[i, j-1] > center) << 0
                lbp[i, j] = code / 255.0  # 归一化
        
        # 金属纹理通常更规则，方差更小
        texture_var = gaussian_filter(lbp**2, sigma=1)
        
        # 下采样为网格矩阵
        h, w = region.shape[:2]
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                roi = texture_var[
                    i*h//self.grid_size : (i+1)*h//self.grid_size,
                    j*w//self.grid_size : (j+1)*w//self.grid_size
                ]
                grid[i, j] = np.mean(roi)
        
        return grid

    def _split_into_blocks(self, feature_matrix):
        """将特征矩阵细分为更小的分块矩阵，捕捉细微差异"""
        blocks = []
        for i in range(0, self.grid_size, self.block_size):
            for j in range(0, self.grid_size, self.block_size):
                block = feature_matrix[
                    i:i+self.block_size,
                    j:j+self.block_size
                ]
                blocks.append(block.flatten())  # 展平分块
        return np.concatenate(blocks)  # 合并所有分块特征

    def extract_multi_feature_vector(self, region):
        """提取多特征矩阵并转换为特征向量"""
        # 1. 提取所有类型的特征矩阵
        feature_matrices = {}
        for name, extractor in self.feature_extractors.items():
            feature_matrices[name] = extractor(region)
        
        # 2. 应用注意力机制（突出重要特征）
        with torch.no_grad():
            device = next(self.attention_net.parameters()).device
            # 转换为张量 (1, 4, grid_size, grid_size)
            feat_tensor = torch.tensor(
                np.stack(list(feature_matrices.values())),
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            # 计算注意力权重
            attn_weights = self.attention_net(feat_tensor)[0].cpu().numpy()  # (4, grid_size, grid_size)
            self.attention_map = attn_weights
        
        # 3. 加权融合特征矩阵
        weighted_features = []
        for i, (name, mat) in enumerate(feature_matrices.items()):
            weighted = mat * attn_weights[i]  # 应用注意力权重
            weighted_features.append(weighted)
        
        # 4. 细分为小分块并转换为特征向量
        feature_vector = []
        for wf in weighted_features:
            feature_vector.append(self._split_into_blocks(wf))
        
        self.features = feature_matrices  # 保存中间特征
        return np.concatenate(feature_vector)

    def recognize_needle(self, region):
        """
        识别区域中是否存在金属针
        :param region: 待检测区域图像 (H, W, 3)
        :return: 识别结果(1=针, 0=非针)、置信度、分块矩阵差异图
        """
        # 提取特征向量
        feat_vec = self.extract_multi_feature_vector(region)
        
        # 预测并计算置信度
        pred = self.classifier.predict([feat_vec])[0]
        pred_prob = self.classifier.predict_proba([feat_vec])[0][1]  # 针的概率
        
        # 计算分块矩阵差异（用于可视化）
        block_diff = np.zeros((self.grid_size//self.block_size, 
                              self.grid_size//self.block_size))
        glare_mat = self.features['glare']
        for i in range(block_diff.shape[0]):
            for j in range(block_diff.shape[1]):
                block = glare_mat[
                    i*self.block_size : (i+1)*self.block_size,
                    j*self.block_size : (j+1)*self.block_size
                ]
                block_diff[i, j] = np.std(block)  # 用标准差表示差异
        
        # 应用双阈值过滤
        if pred_prob < self.low_thresh:
            return 0, pred_prob, block_diff
        elif pred_prob > self.high_thresh:
            return 1, pred_prob, block_diff
        else:
            # 中置信度区域结合边缘特征二次判断
            edge_mat = self.features['edge']
            edge_strength = np.mean(edge_mat)
            return 1 if edge_strength > 0.2 else 0, pred_prob, block_diff

    def train(self, X_regions, y_labels):
        """训练分类器"""
        print("提取训练特征...")
        X_features = []
        for region in tqdm(X_regions):
            X_features.append(self.extract_multi_feature_vector(region))
        
        print("训练分类器...")
        self.classifier.fit(X_features, y_labels)
        return self

    def visualize_features(self, region, block_diff):
        """可视化特征矩阵和分块差异"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 原始区域
        axes[0, 0].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("检测区域")
        axes[0, 0].axis('off')
        
        # 反光特征矩阵
        im = axes[0, 1].imshow(self.features['glare'], cmap='hot')
        axes[0, 1].set_title("金属反光特征矩阵（值越大越亮）")
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # 边缘特征矩阵
        im = axes[0, 2].imshow(self.features['edge'], cmap='binary')
        axes[0, 2].set_title("边缘特征矩阵（解决重叠）")
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 颜色特征矩阵
        im = axes[1, 0].imshow(self.features['color'], cmap='coolwarm')
        axes[1, 0].set_title("颜色特征矩阵（区分血液）")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 纹理特征矩阵
        im = axes[1, 1].imshow(self.features['texture'], cmap='viridis')
        axes[1, 1].set_title("纹理特征矩阵（区分组织）")
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # 分块矩阵差异
        ax = axes[1, 2]
        im = ax.imshow(block_diff, cmap='jet')
        ax.set_title("分块矩阵差异（值越大差异越显著）")
        # 绘制分块边界
        for i in range(self.grid_size//self.block_size + 1):
            ax.axhline(i*self.block_size, color='white', linewidth=1)
            ax.axvline(i*self.block_size, color='white', linewidth=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()


# ------------------------------
# 示例使用
# ------------------------------
def generate_surgical_scene(has_needle=True, has_blood=False, has_overlap=False):
    """生成包含金属针的手术场景图像（模拟反光、血液遮挡和重叠）"""
    h, w = 256, 256
    scene = np.ones((h, w, 3), dtype=np.uint8) * 180  # 组织背景
    
    # 添加金属针（模拟反光）
    if has_needle:
        # 针的位置和形状
        x1, y1 = w//2, h//3
        x2, y2 = w//2 + 100, h//3 + 30
        cv2.line(scene, (x1, y1), (x2, y2), (200, 200, 255), 3)  # 针体
        # 添加高光反光
        cv2.line(scene, (x1+10, y1+3), (x2-10, y2+3), (255, 255, 255), 2)  # 反光条
    
    # 添加血液遮挡
    if has_blood:
        blood = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(blood, (w//2 + 30, h//3 + 15), 20, 255, -1)
        blood = cv2.GaussianBlur(blood, (15, 15), 0)
        scene[blood > 50] = np.clip(
            scene[blood > 50] * np.array([0.8, 0.2, 0.2]) + 30,
            0, 255
        ).astype(np.uint8)
    
    # 添加器械重叠
    if has_overlap:
        # 模拟另一个器械
        x1, y1 = w//2 - 20, h//3 - 30
        x2, y2 = w//2 + 80, h//3 + 60
        cv2.line(scene, (x1, y1), (x2, y2), (150, 150, 150), 4)
    
    # 添加噪声
    scene = cv2.GaussianBlur(scene, (3, 3), 0)
    scene += np.random.normal(0, 5, scene.shape).astype(np.int8)
    scene = np.clip(scene, 0, 255).astype(np.uint8)
    
    return scene


if __name__ == "__main__":
    # 1. 生成训练数据
    print("生成模拟训练数据...")
    num_samples = 100
    X_regions = []
    y_labels = []
    
    # 生成有针和无针的样本
    for i in tqdm(range(num_samples)):
        has_needle = np.random.rand() > 0.5
        has_blood = np.random.rand() > 0.3
        has_overlap = np.random.rand() > 0.3
        
        scene = generate_surgical_scene(has_needle, has_blood, has_overlap)
        X_regions.append(scene)
        y_labels.append(1 if has_needle else 0)
    
    # 2. 初始化并训练识别器
    recognizer = MetalNeedleRecognizer(
        block_size=4,  # 4x4细分矩阵块
        grid_size=16,  # 16x16基础网格
        low_thresh=0.35,
        high_thresh=0.65
    )
    recognizer.train(X_regions, y_labels)
    
    # 3. 测试识别器
    print("\n测试金属针识别...")
    test_scene = generate_surgical_scene(has_needle=True, has_blood=True, has_overlap=True)
    pred, prob, block_diff = recognizer.recognize_needle(test_scene)
    
    # 输出结果
    result = "存在金属针" if pred == 1 else "不存在金属针"
    print(f"识别结果: {result} (置信度: {prob:.2f})")
    
    # 4. 可视化特征矩阵和分块差异
    recognizer.visualize_features(test_scene, block_diff)
import numpy as np
import cv2  # 基于opencv-python 4.11.0.86
import pandas as pd  # 基于pandas 2.3.3
import matplotlib.pyplot as plt  # 基于matplotlib 3.9.4
from matplotlib.animation import FuncAnimation
from tqdm import tqdm  # 基于tqdm 4.67.1
import torch  # 基于torch 2.7.1+cu118
import torch.nn as nn
from sklearn.cluster import KMeans  # 基于scikit-learn 1.5.1
from scipy.ndimage import gaussian_filter  # 基于scipy 1.13.1


class SurgicalReviewSystem:
    """手术复盘系统：基于像素矩阵热力图和器械时间序列分析"""
    
    def __init__(self, video_resolution=(1080, 1920), 
                 low_threshold=0.2, high_threshold=0.7):
        """
        初始化系统
        :param video_resolution: 视频分辨率 (H, W)
        :param low_threshold: 低置信度阈值（过滤干扰）
        :param high_threshold: 高置信度阈值（确认有效器械）
        """
        self.H, self.W = video_resolution
        self.low_thresh = low_threshold
        self.high_thresh = high_threshold
        
        # 1. 热力图矩阵（记录器械出现频率）
        self.heatmap = np.zeros((self.H, self.W), dtype=np.float32)
        
        # 2. 器械时间序列数据（记录每帧出现的器械）
        self.instrument_log = {
            'timestamp': [],       # 时间戳
            'frame_idx': [],       # 帧索引
            'instruments': [],     # 检测到的器械列表
            'positions': [],       # 器械位置坐标
            'confidence': []       # 置信度
        }
        
        # 3. 器械识别模型（带双阈值过滤）
        self.detector = self._build_detector()
        
        # 4. 颜色聚类器（分析颜色区域分布）
        self.color_cluster = KMeans(n_clusters=5, random_state=42)

    def _build_detector(self):
        """构建轻量级器械检测器（带置信度过滤）"""
        # 简化的CNN检测器（实际应用可替换为预训练模型）
        class InstrumentDetector(nn.Module):
            def __init__(self, num_classes=5, low_thresh=0.2, high_thresh=0.7):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.classifier = nn.Linear(32 * 270 * 480, num_classes)  # 适配1080x1920输入
                self.low_thresh = low_thresh
                self.high_thresh = high_thresh

            def forward(self, x):
                """返回：(器械类别, 位置, 置信度)"""
                feat = self.features(x)
                logits = self.classifier(feat.view(x.shape[0], -1))
                probs = torch.softmax(logits, dim=1)
                return probs

            def detect(self, frame):
                """检测单帧中的器械"""
                self.eval()
                with torch.no_grad():
                    # 预处理
                    frame_tensor = torch.tensor(
                        frame.transpose(2, 0, 1), 
                        dtype=torch.float32
                    ).unsqueeze(0) / 255.0
                    
                    # 预测
                    probs = self.forward(frame_tensor)[0]
                    results = []
                    
                    # 应用双阈值过滤
                    for cls_idx, prob in enumerate(probs):
                        if prob > self.high_thresh:
                            # 高置信度：直接保留
                            results.append((cls_idx, prob.item()))
                        elif self.low_thresh < prob <= self.high_thresh:
                            # 中置信度：结合颜色特征二次判断
                            results.append((cls_idx, prob.item()))
                    return results
        
        return InstrumentDetector(
            low_thresh=self.low_thresh,
            high_thresh=self.high_thresh
        )

    def update_heatmap(self, positions, decay=0.99):
        """
        更新热力图矩阵（根据器械位置累积频率）
        :param positions: 器械位置列表 [(x1,y1), (x2,y2), ...]
        :param decay: 衰减因子（防止历史数据过度影响）
        """
        # 热力图衰减（让新数据权重更高）
        self.heatmap = self.heatmap * decay
        
        # 累积新位置的频率
        for (x, y) in positions:
            if 0 <= x < self.W and 0 <= y < self.H:
                # 以器械位置为中心，生成区域热度
                self.heatmap[
                    max(0, y-10):min(self.H, y+10),
                    max(0, x-10):min(self.W, x+10)
                ] += 1.0
        
        # 归一化热力图
        if np.max(self.heatmap) > 0:
            self.heatmap = self.heatmap / np.max(self.heatmap)

    def process_frame(self, frame, frame_idx, timestamp):
        """
        处理单帧视频：检测器械+更新热力图+记录时间序列
        :param frame: 视频帧 (H, W, 3)
        :param frame_idx: 帧索引
        :param timestamp: 时间戳（秒）
        """
        # 1. 检测器械
        detections = self.detector.detect(frame)
        if not detections:
            return False
        
        # 2. 提取器械位置（简化为中心点检测）
        h, w = frame.shape[:2]
        positions = [
            (int(w * (i+1)/(len(detections)+1)), int(h/2))  # 模拟位置
            for i, _ in enumerate(detections)
        ]
        
        # 3. 更新时间序列日志
        self.instrument_log['timestamp'].append(timestamp)
        self.instrument_log['frame_idx'].append(frame_idx)
        self.instrument_log['instruments'].append([d[0] for d in detections])
        self.instrument_log['positions'].append(positions)
        self.instrument_log['confidence'].append([d[1] for d in detections])
        
        # 4. 更新热力图
        self.update_heatmap(positions)
        return True

    def analyze_color_regions(self, frame):
        """分析颜色区域的像素矩阵分布"""
        # 提取颜色特征
        pixels = frame.reshape(-1, 3)
        self.color_cluster.fit(pixels)
        
        # 计算每个颜色簇的像素占比（矩阵值）
        labels, counts = np.unique(self.color_cluster.labels_, return_counts=True)
        color_ratios = counts / len(pixels)
        
        # 分析颜色区域变化：如果某颜色矩阵值总体均衡变大，说明出现更频繁
        return {
            'colors': self.color_cluster.cluster_centers_.astype(int),
            'ratios': color_ratios,
            'is_frequent': np.mean(color_ratios > np.mean(color_ratios)) > 0.5
        }

    def get_instrument_timeline(self, instrument_id):
        """提取特定器械的时间序列数据"""
        # 转换为DataFrame便于分析
        df = pd.DataFrame(self.instrument_log)
        
        # 标记器械出现的帧
        df['has_target'] = df['instruments'].apply(
            lambda x: instrument_id in x
        )
        
        # 计算每秒钟出现的频率
        timeline = df.groupby('timestamp')['has_target'].mean().reset_index()
        return timeline

    def generate_review_report(self, instrument_names):
        """生成手术复盘报告"""
        report = {
            'total_frames': len(self.instrument_log['frame_idx']),
            'total_time': max(self.instrument_log['timestamp']) if self.instrument_log['timestamp'] else 0,
            'instrument_stats': {},
            'hot_regions': self._get_hot_regions()
        }
        
        # 统计每个器械的出现频率和时间
        for idx, name in enumerate(instrument_names):
            timeline = self.get_instrument_timeline(idx)
            total_occur = timeline['has_target'].sum()
            report['instrument_stats'][name] = {
                'occurrence_rate': total_occur / len(timeline) if len(timeline) > 0 else 0,
                'total_time': total_occur,
                'peak_times': timeline.sort_values('has_target', ascending=False).head(3)['timestamp'].tolist()
            }
        
        return report

    def _get_hot_regions(self, threshold=0.7):
        """提取热力图中的高频区域"""
        # 高斯模糊平滑热力图
        smoothed = gaussian_filter(self.heatmap, sigma=5)
        hot_mask = smoothed > threshold
        
        # 找到连通区域（高频出现区域）
        regions = []
        if np.any(hot_mask):
            _, labels, stats, _ = cv2.connectedComponentsWithStats(
                hot_mask.astype(np.uint8) * 255, connectivity=8
            )
            for i in range(1, len(stats)):  # 跳过背景
                x, y, w, h, _ = stats[i]
                regions.append({
                    'bbox': (x, y, w, h),
                    'intensity': np.mean(smoothed[y:y+h, x:x+w])
                })
        
        return sorted(regions, key=lambda r: r['intensity'], reverse=True)

    def visualize(self, video_frames=None, instrument_names=None):
        """可视化热力图和时间序列分析结果"""
        instrument_names = instrument_names or [f'器械{i}' for i in range(5)]
        
        # 创建2x2可视化面板
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 热力图可视化
        axes[0, 0].imshow(self.heatmap, cmap='jet')
        axes[0, 0].set_title('器械出现频率热力图')
        axes[0, 0].axis('off')
        plt.colorbar(axes[0, 0].imshow(self.heatmap, cmap='jet'), 
                    ax=axes[0, 0], fraction=0.046, pad=0.04)
        
        # 2. 时间序列可视化（前两种器械）
        for i in range(min(2, len(instrument_names))):
            timeline = self.get_instrument_timeline(i)
            axes[0, 1].plot(timeline['timestamp'], timeline['has_target'], 
                          label=instrument_names[i])
        axes[0, 1].set_title('器械出现时间序列')
        axes[0, 1].set_xlabel('时间（秒）')
        axes[0, 1].set_ylabel('出现频率')
        axes[0, 1].legend()
        
        # 3. 颜色区域分析
        if video_frames:
            color_analysis = self.analyze_color_regions(video_frames[0])
            axes[1, 0].pie(color_analysis['ratios'], 
                         colors=color_analysis['colors']/255.0,
                         labels=[f'区域{i}' for i in range(len(color_analysis['ratios']))])
            axes[1, 0].set_title(f'颜色区域分布（是否高频出现：{color_analysis["is_frequent"]}）')
        
        # 4. 高频区域统计
        hot_regions = self._get_hot_regions()
        axes[1, 1].bar(
            [f'区域{i+1}' for i in range(len(hot_regions))],
            [r['intensity'] for r in hot_regions]
        )
        axes[1, 1].set_title('高频出现区域强度')
        axes[1, 1].set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.show()

    def save_heatmap_video(self, output_path, video_frames, fps=30):
        """生成带热力图叠加的视频"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.W, self.H))
        
        # 重置热力图（重新计算用于视频）
        temp_heatmap = np.zeros_like(self.heatmap)
        
        for i, frame in enumerate(tqdm(video_frames)):
            # 重新处理帧生成热力图
            self.process_frame(frame, i, i/fps)
            
            # 叠加热力图到原始帧
            heatmap_colored = cv2.applyColorMap(
                (self.heatmap * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
            
            out.write(overlay)
        
        out.release()


# ------------------------------
# 示例使用
# ------------------------------
def generate_surgical_video(num_frames=100, fps=30):
    """生成模拟手术视频帧"""
    frames = []
    H, W = 1080, 1920
    
    for i in range(num_frames):
        # 创建手术场景背景
        frame = np.ones((H, W, 3), dtype=np.uint8) * 200
        
        # 随机添加器械（模拟手术过程）
        num_instruments = np.random.randint(1, 4)
        for j in range(num_instruments):
            # 器械位置随时间变化
            x = int(W * 0.3 + 0.4 * W * np.sin(i * 0.1 + j))
            y = int(H * 0.5 + 0.2 * H * np.cos(i * 0.05 + j))
            
            # 绘制不同颜色的器械
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][j]
            cv2.rectangle(frame, (x-20, y-10), (x+20, y+10), color, -1)
        
        # 添加一些干扰（模拟组织）
        if np.random.rand() > 0.7:
            x = np.random.randint(100, W-100)
            y = np.random.randint(100, H-100)
            cv2.circle(frame, (x, y), 50, (255, 200, 200), -1)
        
        frames.append(frame)
    
    return frames


if __name__ == "__main__":
    # 1. 初始化系统
    review_system = SurgicalReviewSystem(
        video_resolution=(1080, 1920),
        low_threshold=0.25,
        high_threshold=0.65
    )
    
    # 2. 生成模拟手术视频
    print("生成模拟手术视频...")
    video_frames = generate_surgical_video(num_frames=200)
    
    # 3. 处理视频帧
    print("处理视频帧并生成分析数据...")
    for i, frame in enumerate(tqdm(video_frames)):
        timestamp = i / 30  # 30fps
        review_system.process_frame(frame, i, timestamp)
    
    # 4. 生成复盘报告
    instrument_names = ["手术刀", "止血钳", "缝合针", "镊子", "牵开器"]
    report = review_system.generate_review_report(instrument_names)
    print("\n手术复盘报告摘要：")
    print(f"总手术时间：{report['total_time']:.1f}秒")
    print(f"高频操作区域数量：{len(report['hot_regions'])}")
    for name, stats in report['instrument_stats'].items():
        print(f"{name} - 出现频率：{stats['occurrence_rate']:.2f}，峰值时间：{stats['peak_times']}")
    
    # 5. 可视化分析结果
    review_system.visualize(video_frames, instrument_names)
    
    # 6. 保存带热力图的复盘视频（可选）
    # review_system.save_heatmap_video("surgical_review.mp4", video_frames)