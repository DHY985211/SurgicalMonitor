const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

// 保持窗口引用
let mainWindow;

function createWindow() {
  // 创建浏览器窗口
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true, // 允许Node.js集成
      contextIsolation: false, // 关闭上下文隔离
      webSecurity: false, // 关闭跨域限制
      enableRemoteModule: true // 允许远程模块
    },
    icon: path.join(__dirname, 'frontend/icon.ico') // 可选：放个图标文件
  });

  // 加载前端页面
  mainWindow.loadFile(path.join(__dirname, 'frontend/index.html'));

  // 打开开发者工具（可选）
  // mainWindow.webContents.openDevTools();

  // 启动后端服务
  startBackends();

  // 窗口关闭时触发
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// 启动Python和Node.js后端
function startBackends() {
  // 修复环境变量，添加System32路径
  const fixedEnv = {
    ...process.env,
    PATH: `${process.env.PATH};C:\\WINDOWS\\System32;${process.env.SystemRoot}\\system32`
  };

  // 启动Python后端
  try {
    const pythonProcess = spawn('D:\\Anaconda3\\envs\\cellpose_finally\\python.exe', 
      ['app.py'], 
      { 
        cwd: path.join(__dirname, 'Python_backend'), 
        shell: true,
        env: fixedEnv, // 使用修复后的环境变量
        windowsHide: true // 隐藏cmd窗口
      }
    );
    
    pythonProcess.stdout.on('data', (data) => {
      console.log('Python后端:', data.toString().trim());
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error('Python后端错误:', data.toString().trim());
    });

    pythonProcess.on('error', (err) => {
      console.warn('Python后端启动警告:', err.message);
    });
  } catch (err) {
    console.error('启动Python后端失败:', err);
  }

  // 启动Node.js后端
  try {
    const nodeProcess = spawn('node', 
      ['app.js'], 
      { 
        cwd: path.join(__dirname, 'main'), 
        shell: true,
        env: fixedEnv, // 使用修复后的环境变量
        windowsHide: true // 隐藏cmd窗口
      }
    );
    
    nodeProcess.stdout.on('data', (data) => {
      console.log('Node.js后端:', data.toString().trim());
    });
    
    nodeProcess.stderr.on('data', (data) => {
      console.error('Node.js后端错误:', data.toString().trim());
    });

    nodeProcess.on('error', (err) => {
      console.warn('Node.js后端启动警告:', err.message);
    });
  } catch (err) {
    console.error('启动Node.js后端失败:', err);
  }
}

// 应用准备就绪时创建窗口
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// 所有窗口关闭时退出应用
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// 捕获未处理的异常，避免弹窗
process.on('uncaughtException', (err) => {
  console.error('未捕获异常:', err);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('未处理的Promise拒绝:', reason);
});