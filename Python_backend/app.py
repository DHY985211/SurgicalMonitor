from flask import Flask, request, jsonify
from flask_cors import CORS  # 解决跨域问题
import os
from routes import detection, analysis, report
from utils import tools  # 工具函数

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 注册路由（映射到具体的处理函数）
@app.route('/api/detect/instruments', methods=['POST'])
def detect_instruments():
    """调用识别模块，检测单帧中的器械"""
    data = request.json
    frame_data = data.get('frame_data')  # 前端传来的图像数据
    if not frame_data:
        return jsonify({'success': False, 'error': '缺少图像数据'})
    
    # 调用detection.py中的核心函数
    result = detection.detect_instruments(frame_data)
    return jsonify({'success': True, 'data': result})

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video():
    """调用分析模块，处理视频文件"""
    data = request.json
    video_path = data.get('video_path')
    if not video_path or not os.path.exists(video_path):
        return jsonify({'success': False, 'error': '视频文件不存在'})
    
    # 调用analysis.py中的视频分析函数
    result = analysis.analyze_video(video_path, output_dir='./temp_output')
    return jsonify({'success': True, 'data': result})

@app.route('/api/report/generate', methods=['POST'])
def generate_report():
    """调用报告模块，生成PDF报告"""
    data = request.json
    analysis_results = data.get('analysis_results')
    if not analysis_results:
        return jsonify({'success': False, 'error': '缺少分析结果数据'})
    
    # 调用report.py中的报告生成函数
    pdf_path = report.generate_pdf_report(analysis_results, './reports')
    return jsonify({'success': True, 'pdf_path': pdf_path})

@app.route('/api/track/trajectory', methods=['POST'])
def track_trajectory():
    """调用轨迹追踪模块（你之前的AdaptiveSphereTracker）"""
    data = request.json
    frame_sequence = data.get('frame_sequence')  # 连续帧数据
    if not frame_sequence:
        return jsonify({'success': False, 'error': '缺少帧序列数据'})
    
    # 假设轨迹追踪逻辑在analysis.py中
    result = analysis.track_instrument_trajectory(frame_sequence)
    return jsonify({'success': True, 'trajectory': result})

if __name__ == '__main__':
    # 创建临时目录（用于输出文件）
    os.makedirs('./temp_output', exist_ok=True)
    os.makedirs('./reports', exist_ok=True)
    
    # 启动Flask服务（端口5000，允许外部访问）
    app.run(host='0.0.0.0', port=5000, debug=True)