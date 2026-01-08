const express = require('express');
const router = express.Router();

// 简化的登录逻辑（实际项目需添加密码验证）
router.post('/login', (req, res) => {
    const { username } = req.body;
    if (!username) {
        return res.json({ success: false, error: '请输入用户名' });
    }

    // 模拟登录成功（实际需验证密码）
    res.json({
        success: true,
        token: `token_${Date.now()}`,  // 生成临时token
        user: { username }
    });
});

module.exports = router;