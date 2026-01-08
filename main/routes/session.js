const express = require('express');
const router = express.Router();
const db = require('../utils/db');  // 本地存储工具

// 创建会话
router.post('/create', (req, res) => {
    const { sessionName, doctorName } = req.body;
    if (!sessionName || !doctorName) {
        return res.json({ success: false, error: '缺少会话名称或医生姓名' });
    }

    const sessionId = db.createSession({
        name: sessionName,
        doctor: doctorName,
        startTime: new Date().toISOString(),
        status: 'active'
    });

    res.json({ success: true, sessionId });
});

// 获取会话列表
router.get('/list', (req, res) => {
    const sessions = db.getSessions();
    res.json({ success: true, sessions });
});

module.exports = router;