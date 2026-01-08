import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score  # 确保scikit-learn 1.5.1兼容

class DualThresholdRobustRecognizer:
    """
    双阈值鲁棒性识别器 - 用于手术器械识别，处理人体组织干扰
    结合基础模型预测与异常检测，通过双阈值过滤极端情况
    """
    def __init__(self, base_model, low_threshold=0.2, high_threshold=0.8, 
                 contamination=0.1, random_state=42):
        """
        初始化识别器参数
        :param base_model: 基础识别模型（需包含predict_proba方法）
        :param low_threshold: 低置信度阈值（低于此值视为异常低置信）
        :param high_threshold: 高置信度阈值（高于此值视为高置信有效）
        :param contamination: 异常检测污染率（IsolationForest参数）
        :param random_state: 随机种子，确保可复现性
        """
        self.base_model = base_model
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        
        # 初始化异常检测器（处理极端干扰情况）
        self.outlier_detector = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100  # 兼容scikit-learn 1.5.1默认参数
        )
        
        # 确保阈值有效性
        if self.low_threshold >= self.high_threshold:
            raise ValueError("低阈值必须小于高阈值")

    def fit(self, X_train, y_train, feature_extractor=None):
        """
        训练异常检测器并适配基础模型（如需）
        :param X_train: 训练数据（图像特征或原始数据）
        :param y_train: 训练标签
        :param feature_extractor: 特征提取函数（如用于从图像中提取特征）
        """
        # 如果提供特征提取器，先处理数据
        if feature_extractor is not None:
            X_processed = feature_extractor(X_train)
        else:
            X_processed = X_train
        
        # 训练异常检测器（使用训练数据分布）
        self.outlier_detector.fit(X_processed.reshape(len(X_processed), -1))
        
        # 如果基础模型需要训练（非预训练模型）
        if hasattr(self.base_model, 'fit'):
            self.base_model.fit(X_train, y_train)
        
        return self

    def predict(self, X, feature_extractor=None):
        """
        预测函数 - 结合双阈值和异常检测
        :param X: 输入数据（图像或特征）
        :param feature_extractor: 特征提取函数
        :return: 预测结果（类别标签）和处理状态
        """
        # 特征处理
        if feature_extractor is not None:
            X_processed = feature_extractor(X)
            X_flat = X_processed.reshape(len(X_processed), -1)
        else:
            X_processed = X
            X_flat = X.reshape(len(X), -1)
        
        # 基础模型预测
        if hasattr(self.base_model, 'predict_proba'):
            proba = self.base_model.predict_proba(X)
            pred_labels = np.argmax(proba, axis=1)
            max_proba = np.max(proba, axis=1)
        else:
            # 不支持概率输出的模型处理
            pred_labels = self.base_model.predict(X)
            max_proba = np.ones(len(pred_labels))  # 默认为1.0
        
        # 异常检测（-1表示异常，1表示正常）
        outlier_pred = self.outlier_detector.predict(X_flat)
        is_outlier = outlier_pred == -1
        
        # 结果处理状态初始化
        status = np.array(['valid'] * len(X))
        
        # 双阈值处理
        low_conf_mask = max_proba < self.low_threshold
        high_conf_mask = max_proba > self.high_threshold
        
        # 低置信度且非异常 -> 需要人工复核
        status[low_conf_mask & ~is_outlier] = 'needs_review'
        
        # 异常样本（无论置信度）-> 极端干扰
        status[is_outlier] = 'extreme_interference'
        
        # 高置信度样本 -> 直接确认
        status[high_conf_mask & ~is_outlier] = 'high_confidence'
        
        return pred_labels, status

    def evaluate(self, X_test, y_test, feature_extractor=None):
        """
        评估模型性能
        :param X_test: 测试数据
        :param y_test: 真实标签
        :return: 性能指标字典
        """
        pred_labels, status = self.predict(X_test, feature_extractor)
        
        # 整体准确率
        overall_acc = accuracy_score(y_test, pred_labels)
        
        # 各状态下的准确率
        valid_mask = status == 'valid'
        valid_acc = accuracy_score(y_test[valid_mask], pred_labels[valid_mask]) if np.any(valid_mask) else 0
        
        high_conf_mask = status == 'high_confidence'
        high_conf_acc = accuracy_score(y_test[high_conf_mask], pred_labels[high_conf_mask]) if np.any(high_conf_mask) else 0
        
        return {
            'overall_accuracy': overall_acc,
            'valid_accuracy': valid_acc,
            'high_confidence_accuracy': high_conf_acc,
            'status_distribution': {
                'valid': np.sum(valid_mask),
                'needs_review': np.sum(status == 'needs_review'),
                'extreme_interference': np.sum(status == 'extreme_interference'),
                'high_confidence': np.sum(high_conf_mask)
            }
        }


# ------------------------------
# 示例使用（基于PyTorch模型）
# ------------------------------
if __name__ == "__main__":
    # 1. 定义基础模型（示例：简单CNN用于器械识别）
    class SimpleInstrumentCNN(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 16 * 16, 128),  # 假设输入为64x64图像
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
        
        def predict_proba(self, x):
            """适配sklearn风格的概率输出"""
            with torch.no_grad():
                logits = self.forward(x)
                return torch.softmax(logits, dim=1).numpy()
        
        def predict(self, x):
            with torch.no_grad():
                logits = self.forward(x)
                return torch.argmax(logits, dim=1).numpy()

    # 2. 初始化组件
    cnn_model = SimpleInstrumentCNN(num_classes=5)
    recognizer = DualThresholdRobustRecognizer(
        base_model=cnn_model,
        low_threshold=0.3,
        high_threshold=0.7,
        contamination=0.05  # 假设5%的样本为极端干扰
    )

    # 3. 模拟训练数据（64x64 RGB图像，5类器械）
    X_train = torch.randn(1000, 3, 64, 64)  # 符合PyTorch输入格式
    y_train = torch.randint(0, 5, (1000,))

    # 4. 训练模型
    recognizer.fit(
        X_train, 
        y_train,
        feature_extractor=lambda x: x.numpy().reshape(len(x), -1)  # 简单特征提取
    )

    # 5. 模拟测试数据
    X_test = torch.randn(200, 3, 64, 64)
    y_test = torch.randint(0, 5, (200,))

    # 6. 预测与评估
    pred_labels, status = recognizer.predict(X_test)
    metrics = recognizer.evaluate(X_test, y_test.numpy())

    print("评估结果：")
    print(f"整体准确率: {metrics['overall_accuracy']:.4f}")
    print(f"高置信样本准确率: {metrics['high_confidence_accuracy']:.4f}")
    print("样本状态分布:")
    for k, v in metrics['status_distribution'].items():
        print(f"  {k}: {v}个样本")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import KDTree
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
import cv2  # 基于opencv-python 4.11.0.86
from torchvision import transforms  # 兼容torchvision 0.22.1+cu118
import matplotlib.pyplot as plt  # 基于matplotlib 3.9.4


class VoxelFeatureExtractor(nn.Module):
    """3D体素特征提取器 - 处理密集堆积器械的空间信息"""
    def __init__(self, voxel_channels=32):
        super().__init__()
        # 3D卷积层提取空间特征
        self.conv3d_1 = nn.Conv3d(
            in_channels=1,  # 单通道体素（密度信息）
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv3d_2 = nn.Conv3d(
            in_channels=16,
            out_channels=voxel_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.pool3d = nn.MaxPool3d(kernel_size=2)
        self.norm = nn.BatchNorm3d(voxel_channels)

    def forward(self, voxel_grid):
        """
        输入: 3D体素网格 (batch_size, depth, height, width)
        输出: 空间特征向量 (batch_size, voxel_channels * 4)
        """
        x = voxel_grid.unsqueeze(1)  # 增加通道维度 (B, C=1, D, H, W)
        x = F.relu(self.conv3d_1(x))
        x = F.relu(self.norm(self.conv3d_2(x)))
        x = self.pool3d(x)
        return x.flatten(1)  # 展平空间特征


class DetailFeatureExtractor(nn.Module):
    """2D细节特征提取器 - 捕捉器械表面纹理等细微差异"""
    def __init__(self, img_channels=32):
        super().__init__()
        # 2D卷积层提取表面细节
        self.conv2d_1 = nn.Conv2d(
            in_channels=3,  # RGB图像
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=16,
            out_channels=img_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.pool2d = nn.MaxPool2d(kernel_size=2)
        self.norm = nn.BatchNorm2d(img_channels)

    def forward(self, rgb_images):
        """
        输入: 2D RGB图像 (batch_size, 3, height, width)
        输出: 细节特征向量 (batch_size, img_channels * 16)
        """
        x = F.relu(self.conv2d_1(rgb_images))
        x = F.relu(self.norm(self.conv2d_2(x)))
        x = self.pool2d(x)
        return x.flatten(1)  # 展平细节特征


class MultiModalInstrumentRecognizer(nn.Module):
    """多模态融合识别器 - 结合3D空间特征与2D细节特征"""
    def __init__(self, num_classes=10, voxel_channels=32, img_channels=32):
        super().__init__()
        # 子特征提取器
        self.voxel_extractor = VoxelFeatureExtractor(voxel_channels)
        self.detail_extractor = DetailFeatureExtractor(img_channels)
        
        # 融合层与分类器
        self.fusion = nn.Sequential(
            nn.Linear(voxel_channels * 4 * 4 * 4 + img_channels * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.classifier = nn.Linear(256, num_classes)
        
        # 双阈值异常检测器（处理人体组织干扰）
        self.outlier_detector = IsolationForest(
            contamination=0.08,  # 假设8%为干扰样本
            random_state=42,
            n_estimators=100  # 兼容scikit-learn 1.5.1
        )
        self.low_threshold = 0.25  # 低置信阈值（过滤模糊样本）
        self.high_threshold = 0.75  # 高置信阈值（直接确认）

    def forward(self, voxel_grid, rgb_images):
        """
        多模态特征融合前向传播
        :param voxel_grid: 3D体素网格 (B, D, H, W)
        :param rgb_images: 2D图像 (B, 3, H, W)
        :return: 分类logits
        """
        # 提取特征
        voxel_feat = self.voxel_extractor(voxel_grid)
        detail_feat = self.detail_extractor(rgb_images)
        
        # 特征融合
        fused_feat = torch.cat([voxel_feat, detail_feat], dim=1)
        fused_feat = self.fusion(fused_feat)
        
        # 分类输出
        return self.classifier(fused_feat)

    def predict(self, voxel_grid, rgb_images):
        """
        预测函数 - 结合双阈值过滤干扰
        :return: 预测标签、置信度、状态（正常/需复核/干扰）
        """
        with torch.no_grad():
            # 模型预测
            logits = self.forward(voxel_grid, rgb_images)
            probs = F.softmax(logits, dim=1)
            confidences, pred_labels = torch.max(probs, dim=1)
            confidences = confidences.numpy()
            pred_labels = pred_labels.numpy()

            # 特征融合用于异常检测
            voxel_feat = self.voxel_extractor(voxel_grid).numpy()
            detail_feat = self.detail_extractor(rgb_images).numpy()
            fused_feat = np.concatenate([voxel_feat, detail_feat], axis=1)

            # 异常检测（识别人体组织干扰）
            is_outlier = self.outlier_detector.predict(fused_feat) == -1

            # 状态分类
            status = np.array(['valid'] * len(pred_labels))
            status[confidences < self.low_threshold] = 'needs_review'  # 低置信需复核
            status[is_outlier] = 'tissue_interference'  # 组织干扰
            status[confidences > self.high_threshold] = 'high_confidence'  # 高置信确认

            return pred_labels, confidences, status

    def fit_anomaly_detector(self, voxel_grid, rgb_images):
        """训练异常检测器（使用正常样本分布）"""
        with torch.no_grad():
            voxel_feat = self.voxel_extractor(voxel_grid).numpy()
            detail_feat = self.detail_extractor(rgb_images).numpy()
            fused_feat = np.concatenate([voxel_feat, detail_feat], axis=1)
            self.outlier_detector.fit(fused_feat)


def voxelize_point_cloud(points, voxel_size=0.1, spatial_range=(-1, 1, -1, 1, -1, 1)):
    """
    将点云转换为体素网格（处理密集堆积器械）
    :param points: 三维点云 (N, 3)
    :param voxel_size: 体素大小
    :param spatial_range: 空间范围 (x_min, x_max, y_min, y_max, z_min, z_max)
    :return: 体素网格 (D, H, W)
    """
    x_min, x_max, y_min, y_max, z_min, z_max = spatial_range
    # 计算体素网格尺寸
    dims = (
        int((x_max - x_min) / voxel_size),
        int((y_max - y_min) / voxel_size),
        int((z_max - z_min) / voxel_size)
    )
    # 初始化体素网格（记录点密度）
    voxel_grid = np.zeros(dims, dtype=np.float32)
    
    # 计算每个点所属体素
    for x, y, z in points:
        i = int((x - x_min) / voxel_size)
        j = int((y - y_min) / voxel_size)
        k = int((z - z_min) / voxel_size)
        if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
            voxel_grid[i, j, k] += 1  # 密度累加
    
    # 归一化到[0,1]
    if voxel_grid.max() > 0:
        voxel_grid /= voxel_grid.max()
    return voxel_grid


# ------------------------------
# 示例使用与测试
# ------------------------------
if __name__ == "__main__":
    # 1. 配置参数
    num_classes = 5  # 5种手术器械
    voxel_size = 0.05  # 体素大小（更高精度）
    img_size = (64, 64)  # 2D图像尺寸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 初始化模型
    model = MultiModalInstrumentRecognizer(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 3. 生成模拟数据（密集堆积器械）
    def generate_sample():
        # 生成单个器械的点云（带噪声模拟密集堆积）
        n_points = np.random.randint(500, 1000)
        x = np.random.uniform(-0.8, 0.8, n_points)
        y = np.random.uniform(-0.8, 0.8, n_points)
        z = np.random.uniform(-0.8, 0.8, n_points)
        # 模拟器械形状（添加圆柱特征）
        mask = (x**2 + y**2) < 0.3
        z[mask] = z[mask] * 0.5  # 局部压缩模拟圆柱
        points = np.column_stack([x, y, z])
        
        # 生成对应的2D图像（模拟相机视角）
        img = np.random.rand(*img_size, 3) * 0.3  # 背景
        # 模拟器械边缘
        cv2.rectangle(
            img, 
            (10, 10), 
            (54, 54), 
            (np.random.rand(3)), 
            thickness=2
        )
        return points, img

    # 生成训练数据
    train_size = 200
    train_voxels = []
    train_images = []
    train_labels = []
    for _ in range(train_size):
        points, img = generate_sample()
        voxel = voxelize_point_cloud(points, voxel_size=voxel_size)
        train_voxels.append(voxel)
        # 图像预处理
        img = cv2.resize(img, img_size)
        img = transforms.ToTensor()(img)  # 转为Tensor并归一化
        train_images.append(img)
        train_labels.append(np.random.randint(0, num_classes))

    # 转为Tensor
    train_voxels = torch.tensor(np.array(train_voxels), dtype=torch.float32).to(device)
    train_images = torch.stack(train_images).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)

    # 4. 训练模型
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        logits = model(train_voxels, train_images)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

    # 5. 训练异常检测器（使用正常样本）
    model.fit_anomaly_detector(train_voxels, train_images)

    # 6. 测试模型
    test_size = 50
    test_voxels = []
    test_images = []
    test_labels = []
    for _ in range(test_size):
        points, img = generate_sample()
        voxel = voxelize_point_cloud(points, voxel_size=voxel_size)
        test_voxels.append(voxel)
        img = cv2.resize(img, img_size)
        img = transforms.ToTensor()(img)
        test_images.append(img)
        test_labels.append(np.random.randint(0, num_classes))

    test_voxels = torch.tensor(np.array(test_voxels), dtype=torch.float32).to(device)
    test_images = torch.stack(test_images).to(device)
    test_labels = np.array(test_labels)

    # 预测
    model.eval()
    pred_labels, confidences, status = model.predict(test_voxels, test_images)

    # 评估结果
    overall_acc = accuracy_score(test_labels, pred_labels)
    high_conf_mask = status == 'high_confidence'
    high_conf_acc = accuracy_score(
        test_labels[high_conf_mask], 
        pred_labels[high_conf_mask]
    ) if np.any(high_conf_mask) else 0

    print("\n测试结果：")
    print(f"整体准确率: {overall_acc:.4f}")
    print(f"高置信样本准确率: {high_conf_acc:.4f}")
    print("样本状态分布:")
    for s in np.unique(status):
        print(f"  {s}: {np.sum(status == s)}个")

    # 可视化示例（3D体素与2D图像）
    idx = 0
    fig = plt.figure(figsize=(12, 5))
    
    # 3D体素可视化（切片）
    ax1 = fig.add_subplot(121)
    ax1.imshow(train_voxels[idx, :, :, 10], cmap='gray')  # 第10层切片
    ax1.set_title("3D Voxel Slice")
    
    # 2D图像可视化
    ax2 = fig.add_subplot(122)
    img_np = train_images[idx].permute(1, 2, 0).cpu().numpy()
    ax2.imshow(img_np)
    ax2.set_title("2D Detail Image")
    
    plt.show()
import numpy as np
import cv2  # 基于opencv-python 4.11.0.86
from scipy import ndimage  # 兼容scipy 1.13.1
from sklearn.cluster import DBSCAN  # 基于scikit-learn 1.5.1
import matplotlib.pyplot as plt  # 基于matplotlib 3.9.4
from PIL import Image  # 基于pillow 11.3.0


class ProjectionCountingSystem:
    """基于三面投影的手术器械计数系统，抗人体组织干扰"""
    
    def __init__(self, template_size=(32, 32, 32), 
                 low_threshold=0.3, high_threshold=0.7, 
                 min_similarity=0.65):
        """
        初始化参数
        :param template_size: 单个器械的3D模板尺寸
        :param low_threshold: 低置信度阈值（过滤组织干扰）
        :param high_threshold: 高置信度阈值（确认有效器械）
        :param min_similarity: 投影匹配的最小相似度
        """
        self.template_size = template_size
        self.low_thresh = low_threshold
        self.high_thresh = high_threshold
        self.min_sim = min_similarity
        
        # 器械模板（实际使用时需从样本中学习）
        self.x_template = None  # X轴投影模板
        self.y_template = None  # Y轴投影模板
        self.z_template = None  # Z轴投影模板
        
        # 干扰特征库（用于过滤人体组织）
        self.tissue_signatures = []

    def learn_instrument_template(self, instrument_volume):
        """
        从标准器械体积数据学习投影模板
        :param instrument_volume: 单个器械的3D体积数据 (D, H, W)
        """
        # 生成三面投影模板
        self.x_template = self._project_along_axis(instrument_volume, axis=0)
        self.y_template = self._project_along_axis(instrument_volume, axis=1)
        self.z_template = self._project_along_axis(instrument_volume, axis=2)
        
        # 标准化模板
        self.x_template = self._normalize_projection(self.x_template)
        self.y_template = self._normalize_projection(self.y_template)
        self.z_template = self._normalize_projection(self.z_template)

    def learn_tissue_interference(self, tissue_volumes):
        """学习人体组织的干扰特征"""
        for vol in tissue_volumes:
            # 提取组织的三面投影特征
            sig = [
                self._get_projection_signature(self._project_along_axis(vol, 0)),
                self._get_projection_signature(self._project_along_axis(vol, 1)),
                self._get_projection_signature(self._project_along_axis(vol, 2))
            ]
            self.tissue_signatures.append(sig)

    def count_instruments(self, scene_volume):
        """
        计数场景中的器械数量（抗组织干扰）
        :param scene_volume: 包含器械和可能干扰的3D场景体积 (D, H, W)
        :return: 计数结果、置信度及可疑区域
        """
        # 1. 生成场景的三面投影
        x_proj = self._project_along_axis(scene_volume, axis=0)
        y_proj = self._project_along_axis(scene_volume, axis=1)
        z_proj = self._project_along_axis(scene_volume, axis=2)
        
        # 2. 预处理投影（去噪和增强）
        x_proj = self._preprocess_projection(x_proj)
        y_proj = self._preprocess_projection(y_proj)
        z_proj = self._preprocess_projection(z_proj)
        
        # 3. 检测可能的器械区域（滑动窗口匹配）
        x_matches = self._match_projection(x_proj, self.x_template)
        y_matches = self._match_projection(y_proj, self.y_template)
        z_matches = self._match_projection(z_proj, self.z_template)
        
        # 4. 过滤组织干扰
        x_clean = self._filter_tissue_interference(x_proj, x_matches)
        y_clean = self._filter_tissue_interference(y_proj, y_matches)
        z_clean = self._filter_tissue_interference(z_proj, z_matches)
        
        # 5. 三面投影一致性验证
        candidate_centers = self._find_consistent_centers(
            x_clean, y_clean, z_clean, 
            self.template_size
        )
        
        # 6. 聚类去重（解决重叠计数）
        if len(candidate_centers) == 0:
            return 0, 0.0, []
            
        clusters = DBSCAN(eps=max(self.template_size)/2, min_samples=2).fit(candidate_centers)
        unique_count = len(set(clusters.labels_)) - (1 if -1 in clusters.labels_ else 0)
        
        # 7. 计算整体置信度
        confidence = self._calculate_confidence(
            x_clean, y_clean, z_clean, candidate_centers
        )
        
        # 8. 标记可疑区域（低置信度区域）
        suspicious_regions = self._find_suspicious_regions(
            scene_volume, candidate_centers, confidence
        )
        
        return unique_count, confidence, suspicious_regions

    @staticmethod
    def _project_along_axis(volume, axis=0):
        """沿指定轴生成投影矩阵（累加投影）"""
        return np.sum(volume, axis=axis)
    
    @staticmethod
    def _normalize_projection(proj):
        """标准化投影矩阵到[0,1]范围"""
        if np.max(proj) == 0:
            return proj
        return proj / np.max(proj)
    
    @staticmethod
    def _preprocess_projection(proj):
        """预处理投影：去噪和对比度增强"""
        # 高斯模糊去噪
        proj = ndimage.gaussian_filter(proj, sigma=1.0)
        # 对比度增强
        proj = cv2.normalize(
            proj, None, 0, 1, 
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        return proj
    
    def _match_projection(self, proj, template):
        """滑动窗口匹配模板，返回匹配分数图"""
        t_h, t_w = template.shape
        p_h, p_w = proj.shape
        
        # 初始化匹配分数图
        match_scores = np.zeros((p_h - t_h + 1, p_w - t_w + 1))
        
        # 滑动窗口计算相似度
        for i in range(match_scores.shape[0]):
            for j in range(match_scores.shape[1]):
                window = proj[i:i+t_h, j:j+t_w]
                # 计算余弦相似度（抗亮度变化）
                score = np.dot(window.flatten(), template.flatten()) / (
                    np.linalg.norm(window) * np.linalg.norm(template) + 1e-8
                )
                match_scores[i, j] = score
        
        return match_scores
    
    def _filter_tissue_interference(self, proj, matches):
        """过滤人体组织干扰区域"""
        if not self.tissue_signatures:
            return matches  # 无干扰样本时不过滤
        
        # 标记与组织特征相似的区域
        tissue_mask = np.zeros_like(matches, dtype=bool)
        proj_sig = self._get_projection_signature(proj)
        
        for sig in self.tissue_signatures:
            # 计算与组织特征的相似度
            sim = np.dot(proj_sig, sig[0]) / (
                np.linalg.norm(proj_sig) * np.linalg.norm(sig[0]) + 1e-8
            )
            if sim > 0.7:  # 高相似度视为组织
                tissue_mask |= (matches < self.low_thresh)
        
        # 应用双阈值过滤
        clean_matches = matches.copy()
        clean_matches[tissue_mask] = 0  # 组织区域置零
        clean_matches[clean_matches < self.low_thresh] = 0  # 低置信区域置零
        return clean_matches
    
    @staticmethod
    def _get_projection_signature(proj):
        """提取投影矩阵的特征签名（用于干扰识别）"""
        # 提取统计特征
        return np.array([
            np.mean(proj), np.std(proj),
            np.max(proj), np.min(proj),
            ndimage.center_of_mass(proj)[0],
            ndimage.center_of_mass(proj)[1]
        ])
    
    def _find_consistent_centers(self, x_matches, y_matches, z_matches, template_size):
        """寻找三面投影中一致的器械中心"""
        t_d, t_h, t_w = self.template_size
        centers = []
        
        # 提取X投影中的高置信中心
        x_peaks = np.where(x_matches > self.high_thresh)
        for i, j in zip(x_peaks[0], x_peaks[1]):
            x_center = (i + t_h/2, j + t_w/2)
            
            # 在Y投影中验证
            y_i, y_j = int(x_center[0]), int(x_center[1])
            if 0 <= y_i < y_matches.shape[0] and 0 <= y_j < y_matches.shape[1]:
                if y_matches[y_i, y_j] > self.low_thresh:
                    
                    # 在Z投影中验证
                    z_i, z_j = int(x_center[0]), int(x_center[1])
                    if 0 <= z_i < z_matches.shape[0] and 0 <= z_j < z_matches.shape[1]:
                        if z_matches[z_i, z_j] > self.low_thresh:
                            centers.append([x_center[0], x_center[1], z_i])
        
        return np.array(centers)
    
    def _calculate_confidence(self, x_matches, y_matches, z_matches, centers):
        """计算整体计数置信度"""
        if len(centers) == 0:
            return 0.0
            
        # 计算每个中心的平均匹配分数
        scores = []
        t_h, t_w = self.x_template.shape
        for (x, y, z) in centers:
            i, j = int(x - t_h/2), int(y - t_w/2)
            if 0 <= i < x_matches.shape[0] and 0 <= j < x_matches.shape[1]:
                scores.append(x_matches[i, j])
                scores.append(y_matches[i, j])
                scores.append(z_matches[i, j])
        
        return np.mean(scores) if scores else 0.0
    
    def _find_suspicious_regions(self, volume, centers, confidence):
        """标记可疑区域（低置信度区域）"""
        if confidence > self.high_thresh:
            return []
            
        suspicious = []
        t_d, t_h, t_w = self.template_size
        for (x, y, z) in centers:
            # 提取区域边界
            x1, x2 = max(0, int(x - t_h/2)), min(volume.shape[1], int(x + t_h/2))
            y1, y2 = max(0, int(y - t_w/2)), min(volume.shape[2], int(y + t_w/2))
            z1, z2 = max(0, int(z - t_d/2)), min(volume.shape[0], int(z + t_d/2))
            suspicious.append(((z1, z2), (x1, x2), (y1, y2)))
        
        return suspicious

    def visualize_results(self, scene_volume, count, confidence, suspicious_regions):
        """可视化计数结果和三面投影"""
        fig = plt.figure(figsize=(15, 10))
        
        # 显示X/Y/Z三面投影
        ax1 = fig.add_subplot(221)
        ax1.imshow(self._project_along_axis(scene_volume, 0), cmap='gray')
        ax1.set_title('X-axis Projection')
        
        ax2 = fig.add_subplot(222)
        ax2.imshow(self._project_along_axis(scene_volume, 1), cmap='gray')
        ax2.set_title('Y-axis Projection')
        
        ax3 = fig.add_subplot(223)
        ax3.imshow(self._project_along_axis(scene_volume, 2), cmap='gray')
        ax3.set_title('Z-axis Projection')
        
        # 标记可疑区域
        ax4 = fig.add_subplot(224)
        mid_slice = scene_volume.shape[0] // 2
        ax4.imshow(scene_volume[mid_slice], cmap='gray')
        ax4.set_title(f'Count: {count} (Confidence: {confidence:.2f})')
        
        for (z_range, x_range, y_range) in suspicious_regions:
            if z_range[0] <= mid_slice <= z_range[1]:
                rect = plt.Rectangle(
                    (y_range[0], x_range[0]),
                    y_range[1]-y_range[0],
                    x_range[1]-x_range[0],
                    edgecolor='red', facecolor='none', linewidth=2
                )
                ax4.add_patch(rect)
        
        plt.tight_layout()
        plt.show()


# ------------------------------
# 示例使用
# ------------------------------
if __name__ == "__main__":
    # 1. 生成模拟数据（实际应用中替换为真实3D扫描数据）
    def generate_instrument_volume(size=(32, 32, 32)):
        """生成单个器械的3D体积数据"""
        vol = np.zeros(size, dtype=np.float32)
        # 模拟器械形状（圆柱形）
        z, x, y = np.indices(size)
        center = (size[0]//2, size[1]//2, size[2]//2)
        radius = min(size)//4
        length = size[0]//2
        vol[
            (z >= center[0]-length//2) & (z <= center[0]+length//2) &
            ((x-center[1])**2 + (y-center[2])** 2 <= radius**2)
        ] = 1.0
        # 添加噪声
        vol += np.random.normal(0, 0.1, size)
        return np.clip(vol, 0, 1)

    def generate_scene_with_instruments(num_instruments=3, add_tissue=True):
        """生成包含多个器械和可能组织干扰的场景"""
        scene_size = (128, 128, 128)
        scene = np.zeros(scene_size, dtype=np.float32)
        
        # 放置器械
        for i in range(num_instruments):
            instr = generate_instrument_volume()
            # 随机位置
            z_pos = np.random.randint(0, scene_size[0] - instr.shape[0])
            x_pos = np.random.randint(0, scene_size[1] - instr.shape[1])
            y_pos = np.random.randint(0, scene_size[2] - instr.shape[2])
            scene[
                z_pos:z_pos+instr.shape[0],
                x_pos:x_pos+instr.shape[1],
                y_pos:y_pos+instr.shape[2]
            ] += instr
        
        # 添加人体组织干扰
        if add_tissue:
            for _ in range(2):
                tissue = np.random.normal(0.3, 0.1, (64, 64, 64))
                z_pos = np.random.randint(0, scene_size[0] - 64)
                x_pos = np.random.randint(0, scene_size[1] - 64)
                y_pos = np.random.randint(0, scene_size[2] - 64)
                scene[
                    z_pos:z_pos+64,
                    x_pos:x_pos+64,
                    y_pos:y_pos+64
                ] += np.clip(tissue, 0, 0.5)
        
        return np.clip(scene, 0, 1)

    # 2. 初始化系统
    counting_system = ProjectionCountingSystem(
        template_size=(32, 32, 32),
        low_threshold=0.25,
        high_threshold=0.7,
        min_similarity=0.6
    )

    # 3. 学习器械模板
    instrument_template = generate_instrument_volume()
    counting_system.learn_instrument_template(instrument_template)

    # 4. 学习组织干扰特征
    tissue_samples = [np.random.normal(0.3, 0.1, (64, 64, 64)) for _ in range(3)]
    counting_system.learn_tissue_interference(tissue_samples)

    # 5. 生成测试场景（包含3个器械和组织干扰）
    test_scene = generate_scene_with_instruments(num_instruments=3, add_tissue=True)

    # 6. 执行计数
    count, confidence, suspicious = counting_system.count_instruments(test_scene)

    # 7. 输出结果
    print(f"器械计数结果: {count}")
    print(f"计数置信度: {confidence:.2f}")
    print(f"可疑区域数量: {len(suspicious)}")

    # 8. 可视化结果
    counting_system.visualize_results(test_scene, count, confidence, suspicious)
# -------------------------- 必须放在 detection.py 末尾，确保能被导入 --------------------------
def detect_instruments(frame_data):
    """
    检测单帧图像中的手术器械（适配 analysis.py 第1024行导入）
    :param frame_data: 前端传来的base64格式图像数据
    :return: 器械检测结果列表
    """
    import cv2
    import numpy as np
    from ..utils.tools import base64_to_cv2  # 导入工具函数（确保路径正确）
    
    # 1. 解析base64图像
    frame = base64_to_cv2(frame_data)
    if frame is None:
        return [{"error": "图像解析失败", "confidence": 0.0, "position": {}, "bbox": []}]
    
    # 2. 模拟检测结果（实际项目替换为你的模型逻辑，这里确保函数能正常返回）
    detections = [
        {
            "name": "手术刀",
            "confidence": 0.93,
            "position": {"x": 320, "y": 240},
            "bbox": [280, 200, 360, 280]
        },
        {
            "name": "止血钳",
            "confidence": 0.89,
            "position": {"x": 450, "y": 300},
            "bbox": [410, 260, 490, 340]
        }
    ]
    
    return detections

