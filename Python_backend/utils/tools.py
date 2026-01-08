"""Python通用工具函数"""
import cv2
import numpy as np
from datetime import datetime

def base64_to_cv2(base64_str):
    """将前端传来的base64图像转为OpenCV格式"""
    import base64
    import io
    from PIL import Image

    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_base64(img):
    """将OpenCV图像转为base64（用于返回给前端）"""
    import base64
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def get_current_timestamp():
    """获取当前时间戳（用于文件名或日志）"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def create_dir_if_not_exists(path):
    """创建目录（如果不存在）"""
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    return path