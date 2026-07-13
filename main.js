const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const isMac = process.platform === 'darwin';
let backendProcess = null;

function resolvePythonExecutable() {
  const projectRoot = __dirname;
  const candidates = process.platform === 'win32'
    ? [
        path.join(projectRoot, '.venv', 'Scripts', 'python.exe'),
        path.join(projectRoot, 'venv', 'Scripts', 'python.exe'),
        'python'
      ]
    : [
        path.join(projectRoot, '.venv', 'bin', 'python'),
        path.join(projectRoot, 'venv', 'bin', 'python'),
        'python3',
        'python'
      ];

  for (const candidate of candidates) {
    if (candidate.includes(path.sep)) {
      if (fs.existsSync(candidate)) return candidate;
    } else {
      return candidate;
    }
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

function startPythonBackend() {
  const projectRoot = __dirname;
  const backendScript = path.join(projectRoot, 'backend', 'plate_runtime_backend.py');

  if (!fs.existsSync(backendScript)) {
    console.error('[Backend] plate_runtime_backend.py not found:', backendScript);
    return;
  }

  if (backendProcess && !backendProcess.killed) return;

  const pythonExe = resolvePythonExecutable();
  console.log('[Backend] Starting:', pythonExe, backendScript);

  backendProcess = spawn(pythonExe, [backendScript], {
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONIOENCODING: 'utf-8',
      TRAFFIC_BACKEND_PORT: process.env.TRAFFIC_BACKEND_PORT || '8765'
    },
    windowsHide: false
  });

  backendProcess.stdout.on('data', (data) => {
    console.log(`[Backend stdout] ${data.toString('utf8')}`);
  });

  backendProcess.stderr.on('data', (data) => {
    console.error(`[Backend stderr] ${data.toString('utf8')}`);
  });

  backendProcess.on('exit', (code, signal) => {
    console.log(`[Backend] exited code=${code} signal=${signal}`);
    backendProcess = null;
  });
}

function stopPythonBackend() {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
    backendProcess = null;
  }
}

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1500,
    height: 960,
    minWidth: 1280,
    minHeight: 780,
    backgroundColor: '#07111f',
    show: false,
    title: '智慧交通视觉感知系统',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'frontend', 'dashboard.html'));

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });
}

app.whenReady().then(() => {
  startPythonBackend();

  ipcMain.handle('app:get-runtime-info', () => ({
    appName: '智慧交通视觉感知系统',
    version: app.getVersion(),
    platform: process.platform,
    mode: 'electron-python-yolo-ocr',
    backendUrl: 'http://127.0.0.1:8765'
  }));

  ipcMain.handle('dialog:select-video-file', async () => {
    const result = await dialog.showOpenDialog({
      title: '请选择要检测的本地视频',
      properties: ['openFile'],
      filters: [
        { name: 'Video Files', extensions: ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'm4v'] },
        { name: 'All Files', extensions: ['*'] }
      ]
    });
    if (result.canceled || !result.filePaths.length) return null;
    return result.filePaths[0];
  });

  ipcMain.handle('backend:restart', () => {
    stopPythonBackend();
    startPythonBackend();
    return true;
  });

  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('before-quit', () => {
  stopPythonBackend();
});

app.on('window-all-closed', () => {
  if (!isMac) app.quit();
});


