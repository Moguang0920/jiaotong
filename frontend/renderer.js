const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const BACKEND_URL = 'http://127.0.0.1:8765';

const state = {
  tick: 0,
  backendReady: false,
  selectedVideoPath: '',
  plateCount: 0,
  vehicleCount: 0,
  parkingAlerts: 0,
  anomalyCount: 0,
  fps: 0,
  latency: 0,
  plates: [],
  events: [],
  tracks: [],
  devices: [],
  models: [],
  trend: [],
  lastMessage: '等待后端启动与模型预加载。'
};

function nowTime() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false });
}

function formatPlate(text) {
  if (!text) return '识别中';
  if (/^[\u4e00-\u9fa5][A-Z]/.test(text) && text.length >= 3) {
    return `${text.slice(0, 2)}·${text.slice(2)}`;
  }
  return text;
}

function initData() {
  state.events = [
    { title: '系统待命', message: '请先确认 best(1).onnx 位于项目根目录，然后等待 ONNX 与 OCR 模型预加载完成。', level: '低', className: 'low', time: nowTime(), id: 'init-1' }
  ];
  state.tracks = [];
  state.devices = [
    { name: '田浩阳手机采集端', type: 'IP Webcam / Tailscale', ip: '100.70.11.30', status: '待连接', className: 'training' },
    { name: '李俊发分析节点', type: 'Electron + Python Backend', ip: '127.0.0.1:8765', status: '启动中', className: 'training' },
    { name: 'ONNX车牌检测模型', type: 'best(1).onnx / ONNXRuntime', ip: 'project root', status: '加载中', className: 'training' },
    { name: '车牌识别模型', type: 'PaddleOCR TextRecognition Tiny', ip: 'PP-OCRv6_tiny_rec', status: '加载中', className: 'training' }
  ];
  state.models = [
    { name: 'best(1).onnx', desc: 'YOLO ONNX 车牌框检测模型 · ONNXRuntime 推理定位车牌 ROI', score: '加载中', active: true },
    { name: 'PaddleOCR TextRecognition Tiny', desc: '只做车牌字符识别，不再做文字检测', score: '加载中', active: true },
    { name: '多帧投票纠错', desc: '省份票 + 主体票 + 置信度权重，修正 E4682Y / RE4682Y / 康E4682Y 等波动', score: '启用', active: true },
    { name: 'OpenCV VideoCapture', desc: '支持本地视频和 IP Webcam 实时视频流', score: '待启动', active: false }
  ];
  state.trend = Array.from({ length: 12 }, (_, i) => ({ label: `${i + 1}m`, plates: 0, congestion: 0, parking: 0, anomaly: 0 }));
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    }
  });
  const text = await res.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (_) { data = { raw: text }; }
  if (!res.ok) {
    throw new Error(data.detail || data.message || text || `HTTP ${res.status}`);
  }
  return data;
}

function pushEvent(title, message, level = '低', className = 'low') {
  state.events.unshift({ title, message, level, className, time: nowTime(), id: `${Date.now()}-${Math.random()}` });
  state.events = state.events.slice(0, 8);
  if (className === 'high') state.anomalyCount += 1;
}

function setWorkflowRunning(isRunning, sourceText = '') {
  const img = $('#liveFeed');
  const mock = $('#mockRoadScene');
  if (img && mock) {
    if (isRunning) {
      img.classList.remove('hidden');
      mock.classList.add('hidden');
      // 加时间戳，避免浏览器缓存旧 MJPEG 连接。
      img.src = `${BACKEND_URL}/api/stream.mjpg?t=${Date.now()}`;
    } else {
      img.classList.add('hidden');
      mock.classList.remove('hidden');
      img.removeAttribute('src');
    }
  }
  $('#liveModeLabel').textContent = isRunning ? 'LIVE' : 'READY';
  $('#streamAddress').textContent = sourceText || state.lastMessage || '等待启动检测工作流';
}

async function refreshBackendStatus() {
  const started = performance.now();
  try {
    const data = await fetchJson(`${BACKEND_URL}/api/latest`);
    state.latency = Math.round(performance.now() - started);
    applyBackendData(data);
    state.backendReady = true;
  } catch (error) {
    state.backendReady = false;
    state.lastMessage = `后端未连接：${error.message}`;
    state.fps = 0;
    state.models = state.models.map(m => ({ ...m, score: m.name.includes('投票') ? '启用' : '未连接', active: m.name.includes('投票') }));
    state.devices = state.devices.map(d => d.name.includes('分析节点') ? { ...d, status: '未连接', className: 'offline' } : d);
  }
  renderAll();
}

function applyBackendData(data) {
  const status = data.status || {};
  const result = data.result || {};
  state.lastMessage = result.message || status.message || '后端运行中';
  state.fps = Number(status.display_fps || status.capture_fps || 0);
  const running = Boolean(status.running);
  const source = status.source || '';

  const yoloReady = Boolean(status.yolo_ready);
  const ocrReady = Boolean(status.ocr_ready);
  const modelsReady = Boolean(status.models_ready);

  state.devices = [
    { name: '田浩阳手机采集端', type: 'IP Webcam / Tailscale', ip: $('#cameraUrlInput')?.value || '100.70.11.30', status: running && status.source_type === 'camera' ? '在线检测中' : '待连接', className: running && status.source_type === 'camera' ? '' : 'training' },
    { name: '李俊发分析节点', type: 'Electron + Python Backend', ip: BACKEND_URL.replace('http://', ''), status: status.backend || 'unknown', className: status.backend === 'ready' ? '' : 'training' },
    { name: 'ONNX车牌检测模型', type: 'best(1).onnx / ONNXRuntime', ip: status.model_path || 'project root', status: yoloReady ? `已加载 ${status.yolo_provider || ''}` : '未就绪', className: yoloReady ? '' : 'offline' },
    { name: '车牌识别模型', type: 'PaddleOCR TextRecognition Tiny', ip: 'PP-OCRv6_tiny_rec', status: ocrReady ? `已加载 ${status.ocr_model || ''}` : '未就绪', className: ocrReady ? '' : 'offline' }
  ];

  state.models = [
    { name: 'best(1).onnx', desc: `ONNXRuntime 车牌检测 · ${status.yolo_provider || 'provider未知'} · YOLO ${status.yolo_ms || 0}ms`, score: yoloReady ? '已加载' : '未就绪', active: yoloReady },
    { name: 'PaddleOCR TextRecognition Tiny', desc: `轻量 OCR · 每 track 每秒最多补充 1 次 · ${status.ocr_ms || 0}ms`, score: ocrReady ? '已加载' : '未就绪', active: ocrReady },
    { name: '多帧投票纠错', desc: '省份票 + 主体票 + 置信度权重，修正单帧波动', score: '启用', active: true },
    { name: 'OpenCV VideoCapture', desc: '本地视频 / IP Webcam 实时流', score: running ? '运行中' : '待启动', active: running }
  ];

  const plates = result.plates || [];
  const tracks = result.tracks || [];
  const detections = result.detections || [];
  state.vehicleCount = detections.length;

  if (plates.length) {
    plates.forEach((p) => {
      const plateText = p.stable_text || p.plate_text || p.raw_text || '';
      const key = `${p.track_id}-${plateText}-${Math.round((p.stable_score || p.ocr_confidence || 0) * 1000)}`;
      const exists = state.plates.some(old => old.key === key);
      if (!exists && plateText) {
        state.plates.unshift({
          key,
          time: nowTime(),
          plate: formatPlate(plateText),
          whitelist: true,
          decision: p.stable ? '稳定识别' : '候选识别',
          confidence: Math.round((p.stable_score || p.ocr_confidence || 0) * 100),
          id: key
        });
      }
    });
    state.plates = state.plates.slice(0, 10);
    state.plateCount = Math.max(state.plateCount, state.plates.length);
  }

  state.tracks = tracks.map(t => ({
    id: t.track_id,
    plate: formatPlate(t.label),
    zone: t.stable ? '稳定车牌结果' : '候选车牌结果',
    seconds: t.votes || 1,
    status: t.stable ? '已稳定' : '投票中'
  }));

  if (running) {
    setWorkflowRunning(true, source || state.lastMessage);
  } else if (!source) {
    setWorkflowRunning(false, state.lastMessage);
  }

  if (plates.length) {
    const stableCount = plates.filter(p => p.stable).length;
    if (stableCount) pushEvent('车牌稳定识别', `多帧投票已稳定输出 ${stableCount} 个车牌结果`, '低', 'low');
  }

  if (state.tick % 4 === 0) {
    state.trend.push({
      label: nowTime().slice(3),
      plates: Math.min(100, state.plates.length * 10),
      congestion: Math.min(100, state.vehicleCount * 18),
      parking: state.parkingAlerts,
      anomaly: state.anomalyCount
    });
    state.trend = state.trend.slice(-12);
  }

  $('.topbar-actions .status-pill')?.classList.toggle('success', modelsReady);
}

async function chooseVideoFile() {
  try {
    const filePath = await window.trafficDesk?.selectVideoFile?.();
    if (!filePath) return;
    state.selectedVideoPath = filePath;
    $('#videoFilePath').value = filePath;
    pushEvent('已选择本地视频', filePath, '低', 'low');
    renderAll();
  } catch (error) {
    pushEvent('选择视频失败', error.message, '高', 'high');
  }
}

async function startVideoWorkflow() {
  try {
    const path = $('#videoFilePath').value || state.selectedVideoPath;
    if (!path) {
      pushEvent('缺少视频文件', '请先点击“选择视频”，再启动本地视频全流程检测。', '高', 'high');
      return;
    }
    await fetchJson(`${BACKEND_URL}/api/start/video`, {
      method: 'POST',
      body: JSON.stringify({ path })
    });
    state.plates = [];
    state.tracks = [];
    state.plateCount = 0;
    pushEvent('本地视频工作流启动', '开始执行：视频帧 → ONNX 检测车牌 → OCR 异步补票 → 多帧投票。', '低', 'low');
    setWorkflowRunning(true, path);
  } catch (error) {
    pushEvent('启动本地视频失败', error.message, '高', 'high');
  }
}

async function startCameraWorkflow() {
  try {
    const url = $('#cameraUrlInput').value.trim();
    if (!url) {
      pushEvent('缺少视频流地址', '请填写 IP Webcam 的 /video 地址。', '高', 'high');
      return;
    }
    await fetchJson(`${BACKEND_URL}/api/start/camera`, {
      method: 'POST',
      body: JSON.stringify({ url })
    });
    state.plates = [];
    state.tracks = [];
    state.plateCount = 0;
    pushEvent('手机实时检测启动', `正在连接手机视频流：${url}`, '低', 'low');
    setWorkflowRunning(true, url);
  } catch (error) {
    pushEvent('连接手机视频流失败', error.message, '高', 'high');
  }
}

async function stopWorkflow() {
  try {
    await fetchJson(`${BACKEND_URL}/api/stop`, { method: 'POST', body: JSON.stringify({}) });
    pushEvent('工作流已停止', '已停止当前视频检测任务。', '低', 'low');
    setWorkflowRunning(false, '已停止当前工作流');
  } catch (error) {
    pushEvent('停止失败', error.message, '高', 'high');
  }
}

async function restartBackend() {
  try {
    await window.trafficDesk?.restartBackend?.();
    pushEvent('后端重启中', '正在重新启动 Python 后端并预加载模型。', '低', 'low');
    setWorkflowRunning(false, '后端重启中，请稍候...');
  } catch (error) {
    pushEvent('后端重启失败', error.message, '高', 'high');
  }
}

function renderKpis() {
  $('#plateCount').textContent = state.plateCount.toLocaleString('zh-CN');
  $('#vehicleCount').textContent = state.vehicleCount;
  $('#parkingAlerts').textContent = state.parkingAlerts;
  $('#anomalyCount').textContent = state.anomalyCount;
  $('#fpsBadge').textContent = `FPS ${Number(state.fps || 0).toFixed(1)}`;
  $('#latencyBadge').textContent = `Latency ${state.latency || 0}ms`;
}

function renderPlateTable() {
  $('#plateTable').innerHTML = state.plates.length ? state.plates.map(item => {
    const decisionClass = item.decision === '稳定识别' ? 'ok' : 'wait';
    return `
      <tr>
        <td>${item.time}</td>
        <td><strong>${item.plate}</strong></td>
        <td>${item.confidence}%</td>
        <td><span class="tag ok">本地识别</span></td>
        <td><span class="tag ${decisionClass}">${item.decision}</span></td>
      </tr>
    `;
  }).join('') : '<tr><td colspan="5">等待识别结果。启动本地视频或手机实时流后，这里会显示车牌候选与稳定结果。</td></tr>';
}

function renderEvents() {
  $('#eventFeed').innerHTML = state.events.map(item => `
    <div class="event-card ${item.className === 'high' ? '' : item.className}">
      <div class="event-head">
        <div class="event-title">${item.title}</div>
        <span class="event-time">${item.time}</span>
      </div>
      <div class="event-meta">等级：${item.level} · ${item.message}</div>
    </div>
  `).join('');
}

function renderTracks() {
  $('#trackingList').innerHTML = state.tracks.length ? state.tracks.map(item => {
    const cls = item.status === '已稳定' ? '' : 'warn';
    return `
      <div class="track-card">
        <div>
          <div class="track-title">${item.id} · ${item.plate}</div>
          <div class="track-meta">${item.zone} · 状态：${item.status}</div>
        </div>
        <div class="countdown ${cls}">${item.seconds}票</div>
      </div>
    `;
  }).join('') : '<div class="track-card"><div><div class="track-title">暂无车牌轨迹</div><div class="track-meta">检测到车牌框后会自动维护 track_id 并进行多帧投票。</div></div><div class="countdown">0票</div></div>';
}

function renderDevices() {
  $('#deviceGrid').innerHTML = state.devices.map(item => `
    <div class="device-card">
      <div>
        <div class="device-title">${item.name}</div>
        <div class="device-meta">${item.type} · ${item.ip}</div>
      </div>
      <span class="device-state ${item.className}">${item.status}</span>
    </div>
  `).join('');
}

function renderModels() {
  $('#modelList').innerHTML = state.models.map(item => `
    <div class="model-card ${item.active ? 'active' : ''}">
      <div>
        <div class="model-title">${item.name}</div>
        <div class="model-meta">${item.desc}</div>
      </div>
      <div class="model-score">${item.score}</div>
    </div>
  `).join('');
}

function updateResourceBars() {
  const cpu = state.backendReady ? Math.min(95, 20 + state.vehicleCount * 5) : 0;
  const gpu = state.backendReady ? (state.models.some(m => String(m.desc).includes('CUDA')) ? 45 : 8) : 0;
  const mem = state.backendReady ? 55 : 0;
  const stream = state.fps > 0 ? 95 : 35;
  $('#cpuVal').textContent = `${cpu}%`;
  $('#gpuVal').textContent = `${gpu}%`;
  $('#memVal').textContent = `${mem}%`;
  $('#streamVal').textContent = state.fps > 0 ? '稳定' : '待连接';
  $('#cpuBar').style.width = `${cpu}%`;
  $('#gpuBar').style.width = `${gpu}%`;
  $('#memBar').style.width = `${mem}%`;
  $('#streamBar').style.width = `${stream}%`;
}

function drawHeatmap() {
  const canvas = $('#heatmapCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, '#09162a');
  bg.addColorStop(1, '#0d2136');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let x = 20; x < w; x += 46) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x - 80, h);
    ctx.stroke();
  }
  for (let y = 30; y < h; y += 50) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y + 40);
    ctx.stroke();
  }

  const intensity = Math.min(1, state.vehicleCount / 6);
  const points = [
    { x: 150, y: 100, r: 70 + intensity * 80, alpha: 0.18 + intensity * 0.38 },
    { x: 330, y: 170, r: 50 + intensity * 90, alpha: 0.12 + intensity * 0.32 }
  ];
  points.forEach((p) => {
    const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r);
    grd.addColorStop(0, `rgba(255,95,122,${p.alpha})`);
    grd.addColorStop(0.45, `rgba(255,209,102,${p.alpha * 0.55})`);
    grd.addColorStop(1, 'rgba(255,95,122,0)');
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = 'rgba(238,246,255,.88)';
  ctx.font = '700 12px system-ui, Microsoft YaHei';
  ctx.fillText(`当前检测车牌框：${state.vehicleCount}`, 42, 42);
  ctx.fillText(`稳定车牌结果：${state.plates.filter(p => p.decision === '稳定识别').length}`, 42, 64);

  const level = $('#congestionLevel');
  if (!level) return;
  if (state.vehicleCount >= 5) {
    level.textContent = '车牌密集';
    level.className = 'severity high';
  } else if (state.vehicleCount >= 2) {
    level.textContent = '中等密度';
    level.className = 'severity medium';
  } else {
    level.textContent = '低密度';
    level.className = 'severity low';
  }
}

function drawTrend() {
  const canvas = $('#trendCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const padding = { left: 52, right: 26, top: 26, bottom: 38 };
  const chartW = w - padding.left - padding.right;
  const chartH = h - padding.top - padding.bottom;
  ctx.fillStyle = '#071326';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,.08)';
  for (let i = 0; i <= 5; i++) {
    const y = padding.top + chartH * (i / 5);
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(w - padding.right, y);
    ctx.stroke();
  }
  const max = 100;
  const series = [
    { key: 'plates', label: '车牌结果', color: 'rgba(72,216,255,1)' },
    { key: 'congestion', label: '检测密度', color: 'rgba(255,209,102,1)' }
  ];
  series.forEach((s) => {
    ctx.beginPath();
    state.trend.forEach((d, idx) => {
      const x = padding.left + (chartW / Math.max(1, state.trend.length - 1)) * idx;
      const raw = d[s.key];
      const y = padding.top + chartH - (Math.min(raw, max) / max) * chartH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.6;
    ctx.stroke();
  });
  ctx.font = '12px system-ui, Microsoft YaHei';
  let legendX = padding.left;
  series.forEach((s) => {
    ctx.fillStyle = s.color;
    ctx.beginPath();
    ctx.arc(legendX, 16, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(220,236,255,.86)';
    ctx.fillText(s.label, legendX + 9, 20);
    legendX += 100;
  });
}

function updateSection(section) {
  $$('.nav-item').forEach(btn => btn.classList.toggle('active', btn.dataset.section === section));
  $$('.panel').forEach(panel => {
    const supported = panel.dataset.panel.split(' ');
    panel.classList.toggle('hidden', !supported.includes(section));
  });
}

function renderAll() {
  renderKpis();
  renderPlateTable();
  renderEvents();
  renderTracks();
  renderDevices();
  renderModels();
  updateResourceBars();
  drawHeatmap();
  drawTrend();
}

function bindEvents() {
  $$('.nav-item').forEach(btn => {
    btn.addEventListener('click', () => updateSection(btn.dataset.section));
  });
  $('#chooseVideoBtn')?.addEventListener('click', chooseVideoFile);
  $('#startVideoBtn')?.addEventListener('click', startVideoWorkflow);
  $('#startCameraBtn')?.addEventListener('click', startCameraWorkflow);
  $('#stopWorkflowBtn')?.addEventListener('click', stopWorkflow);
  $('#restartBackendBtn')?.addEventListener('click', restartBackend);
  $('#resetMockBtn')?.addEventListener('click', () => {
    state.plates = [];
    state.tracks = [];
    state.plateCount = 0;
    state.vehicleCount = 0;
    pushEvent('界面数据已清空', '仅清空前端表格，不影响后端模型。', '低', 'low');
    renderAll();
  });
}

async function showRuntimeInfo() {
  if (!window.trafficDesk) return;
  try {
    const info = await window.trafficDesk.getRuntimeInfo();
    document.title = `${info.appName} · v${info.version}`;
  } catch (error) {
    console.warn('Failed to read runtime info', error);
  }
}

window.addEventListener('resize', () => {
  drawHeatmap();
  drawTrend();
});

initData();
bindEvents();
renderAll();
showRuntimeInfo();
refreshBackendStatus();
setInterval(() => {
  state.tick += 1;
  refreshBackendStatus();
}, 700);
