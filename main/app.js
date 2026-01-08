const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const path = require('path');

// 初始化Express应用
const app = express();
app.use(cors());  // 允许跨域
app.use(bodyParser.json({ limit: '50mb' }));  // 处理大文件（如视频帧）
app.use(bodyParser.urlencoded({ extended: true }));

// 导入路由
const authRoutes = require('./routes/auth');
const sessionRoutes = require('./routes/session');
const analysisRoutes = require('./routes/analysis');


// 注册路由
app.use('/api/auth', authRoutes);
app.use('/api/session', sessionRoutes);
app.use('/api/analysis', analysisRoutes);  // 分析相关接口（转发到Python）

// 测试接口
app.get('/api/test', (req, res) => {
    res.json({ message: 'Node.js后端运行正常' });
});

// 启动服务（端口3000）
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Node.js后端启动成功，端口：${PORT}`);
    console.log(`Python服务地址：http://localhost:5000`);
});

module.exports = app;