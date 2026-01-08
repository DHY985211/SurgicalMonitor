const express = require('express');
const router = express.Router();
const axios = require('axios');  // 用于调用Python服务

// Python服务地址（Flask运行在5000端口）
const PYTHON_SERVICE_URL = 'http://localhost:5000';

/**
 * 前端调用的分析接口：转发到Python服务
 */
router.post('/detect-instruments', async (req, res) => {
    try {
        // 调用Python的器械检测接口
        const response = await axios.post(
            `${PYTHON_SERVICE_URL}/api/detect/instruments`,
            req.body  // 传递前端的图像数据
        );
        res.json(response.data);
    } catch (err) {
        res.json({
            success: false,
            error: `调用Python服务失败：${err.message}`
        });
    }
});

router.post('/analyze-video', async (req, res) => {
    try {
        const response = await axios.post(
            `${PYTHON_SERVICE_URL}/api/analyze/video`,
            req.body  // 传递视频路径
        );
        res.json(response.data);
    } catch (err) {
        res.json({ success: false, error: err.message });
    }
});

router.post('/track-trajectory', async (req, res) => {
    try {
        const response = await axios.post(
            `${PYTHON_SERVICE_URL}/api/track/trajectory`,
            req.body  // 传递帧序列
        );
        res.json(response.data);
    } catch (err) {
        res.json({ success: false, error: err.message });
    }
});

module.exports = router;