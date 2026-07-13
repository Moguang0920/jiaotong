const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const BACKEND_URL = 'http://127.0.0.1:8765';

const DETECTOR_UI = {
  plate: {
    title: '车牌实时识别画面',
    description: 'best(1).onnx 检测车牌区域，OCR 通过短时关联与多次投票输出稳定车牌。',
    hud: 'Plate OCR：异步投票',
    action: '车牌识别'
  },
  vehicle: {
    title: '车辆与拥堵热力实时画面',
    description: 'hearmap.onnx 检测当前车辆位置，右侧热力图只根据当前帧车辆数量实时更新。',
    hud: 'Vehicle：实时密度',
    action: '拥堵分析'
  },
  stop: {
    title: '禁停区域车辆跟踪画面',
    description: 'stop.onnx 检测 car / carNumber / noParking，车辆真正停止后开始计时告警。',
    hud: 'Stop Track：停车计时',
    action: '禁停检测'
  },
  normal: {
    title: '道路异常实时检测画面',
    description: 'normal.onnx 与手动道路 ROI 联动；圈选 ROI 时直接复用当前视频帧，不重新连接手机。',
    hud: 'Road ROI：异常检测',
    action: '道路异常检测'
  }
};

const SECTION_CONFIG = {
  monitor: {
    eyebrow: 'Integrated Traffic Vision',
    title: '综合实时检测',
    description: '手机视频只连接一次；车牌识别、拥堵热力、禁停跟踪和道路异常通过模型切换在同一页面完成。'
  },
  system: {
    eyebrow: 'Runtime Monitoring',
    title: '系统监控',
    description: '集中查看 CPU、GPU、内存、视频传输、模型推理吞吐和后端运行状态。'
  },
  devices: {
    eyebrow: 'Devices & Video Sources',
    title: '设备与视频源',
    description: '管理手机采集端、Tailscale 地址、分析节点和设备在线状态。'
  },
  models: {
    eyebrow: 'Model Management',
    title: '模型管理',
    description: '查看四个检测模型、OCR 模型、ONNX Provider、加载状态和当前运行模型。'
  },
  whitelist: {
    eyebrow: 'Vehicle Whitelist',
    title: '白名单管理',
    description: '维护允许通行的车辆信息；该数据会在车牌识别时用于放行或拦截决策。'
  },
  history: {
    eyebrow: 'History & Statistics',
    title: '历史统计',
    description: '查询车牌识别记录，并查看车辆密度、违停告警和道路异常的趋势。'
  },
  users: {
    eyebrow: 'Users & Permissions',
    title: '用户权限',
    description: '管理管理员、操作员和查看员账户，以及登录和操作审计信息。'
  },
  settings: {
    eyebrow: 'System Configuration',
    title: '系统配置',
    description: '管理视频同步方式、检测阈值、Provider 参数与系统审计配置。'
  }
};

const state = {
  adminSummary: null,
  currentSection: 'monitor',
  currentUser: null,
  tick: 0,
  backendReady: false,
  workflowRunning: false,
  streamUiActive: false,
  modelSwitching: false,
  selectedVideoPath: '',
  detectorModel: 'plate',
  plateCount: 0,
  vehicleCount: 0,
  parkingAlerts: 0,
  anomalyCount: 0,
  fps: 0,
  latency: 0,
  plates: [],
  plateVotes: [],
  events: [],
  tracks: [],
  devices: [],
  models: [],
  trend: [],
  perf: {},
  currentBoxes: [],
  frameWidth: 0,
  frameHeight: 0,
  lastOverlayFrameId: 0,
  roadMap: { roads: [], assignments: [], heat: [] },
  roadAssignments: [],
  roadHeat: [],
  parkingMonitor: { active: [], alerts: [], zones: [], history: [] },
  parkingHistory: [],
  parkingAlertKeys: new Set(),
  normalRoi: {
    confirmed: false,
    sourceType: 'file',
    source: '',
    points: [],
    draftPoints: [],
    frameWidth: 0,
    frameHeight: 0,
    imageDataUrl: '',
    previewImage: null,
    drawRect: null
  },
  normalLane: {
    enabled: false,
    status: 'disabled',
    roi: [],
    candidate_lines: [],
    lane_lines: [],
    stable_lane_count: 0,
    edge_count: 0,
    processing_ms: 0,
    message: '仅 normal.onnx 模式启用。'
  },
  normalRoadAnalysis: {
    enabled: false,
    roi_ready: false,
    normal_vehicle_count: 0,
    stable_lane_count: 0,
    message: '选择 normal.onnx 后配置道路 ROI。'
  },
  heatField: null,
  heatFieldCols: 72,
  heatFieldRows: 44,
  heatLastUpdate: 0,
  lastMessage: '等待后端启动与模型预加载。'
};

function nowTime() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false });
}

function formatPlate(text) {
  const fixed = fixDemoPlateText(text);
  return fixed || '识别中';
}

function fixDemoPlateText(text) {
  if (!text) return '';

  const t = String(text)
    .trim()
    .toUpperCase()
    .replace(/[^0-9A-Z\u4e00-\u9fa5]/g, '');

  // 未稳定阶段后端只返回 6 位主体号；此时不在前端提前补“京”。
  if (/^[A-Z][A-Z0-9]{5}$/.test(t)) return t;

  // 只有后端投票稳定后返回的“京 + 6 位主体”才作为完整车牌显示。
  if (/^京[A-Z][A-Z0-9]{5}$/.test(t)) return t;

  // 其他长度、其他中文前缀或格式异常结果直接丢弃。
  return '';
}


function getSelectedDetectorModel() {
  return $('#detectorModelSelect')?.value || state.detectorModel || 'plate';
}

function pickBoxes(result) {
  const plates = Array.isArray(result.plates) ? result.plates : [];
  const detections = Array.isArray(result.detections) ? result.detections : [];
  if (plates.length) return plates;
  return detections;
}

function detectorModelLabel(model) {
  if (model === 'vehicle') return '车辆检测 hearmap.onnx';
  if (model === 'stop') return '禁停区域 stop.onnx';
  if (model === 'normal') return '正常道路检测 normal.onnx';
  return '车牌检测 best(1).onnx';
}

function applyDetectorPanelVisibility(model = state.detectorModel) {
  const key = DETECTOR_UI[model] ? model : 'plate';
  const monitorActive = state.currentSection === 'monitor';

  $$('[data-detector-view]').forEach(element => {
    if (!monitorActive) {
      element.classList.remove('detector-hidden');
      return;
    }
    const supported = String(element.dataset.detectorView || '')
      .split(/\s+/)
      .filter(Boolean);
    const visible = supported.includes('all') || supported.includes(key);
    element.classList.toggle('detector-hidden', !visible);
  });

  $$('[data-detector-kpi]').forEach(element => {
    if (!monitorActive) {
      element.classList.remove('detector-hidden');
      return;
    }
    element.classList.toggle('detector-hidden', element.dataset.detectorKpi !== key);
  });
}

function applyDetectorUi(model) {
  const key = DETECTOR_UI[model] ? model : 'plate';
  const config = DETECTOR_UI[key];
  state.detectorModel = key;
  document.body.dataset.detectorModel = key;
  applyDetectorPanelVisibility(key);

  const videoTitle = $('#videoPanelTitle');
  const videoDescription = $('#videoPanelDescription');
  const runtimeHud = $('#runtimeModelHud');
  const liveFeed = $('#liveFeed');
  if (videoTitle) videoTitle.textContent = config.title;
  if (videoDescription) videoDescription.textContent = config.description;
  if (runtimeHud) runtimeHud.textContent = config.hud;
  if (liveFeed) liveFeed.alt = `${config.action}实时视频画面`;

  const startVideoButton = $('#startVideoBtn');
  const startCameraButton = $('#startCameraBtn');
  if (startVideoButton) startVideoButton.textContent = `启动本地${config.action}`;
  if (startCameraButton) startCameraButton.textContent = `连接手机并启动${config.action}`;

  const status = $('#modelSwitchStatus');
  if (status && !state.modelSwitching) {
    status.textContent = `${detectorModelLabel(key)} · 切换不重连视频`;
  }
}

async function switchDetectorModel(nextModel) {
  const model = DETECTOR_UI[nextModel] ? nextModel : 'plate';
  const previous = state.detectorModel;
  if (state.modelSwitching || model === previous) {
    applyDetectorUi(model);
    updateNormalRoiControls();
    return;
  }

  state.modelSwitching = true;
  state.detectorModel = model;
  const modelSelect = $('#detectorModelSelect');
  if (modelSelect) modelSelect.disabled = true;
  applyDetectorUi(model);
  const status = $('#modelSwitchStatus');
  if (status) {
    status.textContent = state.workflowRunning
      ? `视频保持连接，正在加载 ${detectorModelLabel(model)}...`
      : `正在加载 ${detectorModelLabel(model)}...`;
    status.classList.add('switching');
  }

  try {
    const modeValue = $('#syncModeSelect')?.value || 'handheld';
    const mode = modeValue === 'fixed' ? 'fixed' : 'handheld';
    const patch = buildConfigForMode(mode);
    patch.detector_model = model;
    const data = await fetchJson(`${BACKEND_URL}/api/config`, {
      method: 'POST',
      body: JSON.stringify(patch)
    });
    state.detectorModel = data.detector_model || model;

    // 已经圈过道路 ROI 时，切换到 normal 后直接下发，不启动新的 VideoCapture。
    if (model === 'normal' && state.normalRoi.confirmed && state.normalRoi.points.length >= 3) {
      await configureNormalRoiOnBackend();
    }

    pushEvent(
      '检测模型已切换',
      `${detectorModelLabel(model)} 已启用；${data.running ? '当前手机/本地视频连接保持不变。' : '等待启动视频源。'}`,
      '低',
      'low'
    );
  } catch (error) {
    state.detectorModel = previous;
    const select = $('#detectorModelSelect');
    if (select) select.value = previous;
    applyDetectorUi(previous);
    pushEvent('模型切换失败', error.message, '高', 'high');
  } finally {
    state.modelSwitching = false;
    if (modelSelect) modelSelect.disabled = false;
    if (status) status.classList.remove('switching');
    applyDetectorUi(state.detectorModel);
    updateNormalRoiControls();
    renderNormalRoadRuntime();
    drawDetectionOverlay();
  }
}

function initData() {
  state.events = [
    { title: '系统待命', message: '请先确认 best(1).onnx / hearmap.onnx / stop.onnx / normal.onnx 位于项目根目录，然后等待 ONNX 与 OCR 模型预加载完成。', level: '低', className: 'low', time: nowTime(), id: 'init-1' }
  ];
  state.tracks = [];
  state.devices = [
    { name: '田浩阳手机采集端', type: 'IP Webcam / Tailscale', ip: '100.70.11.30', status: '待连接', className: 'training' },
    { name: '李俊发分析节点', type: 'Electron + Python Backend', ip: '127.0.0.1:8765', status: '启动中', className: 'training' },
    { name: 'ONNX检测模型', type: 'best(1).onnx / hearmap.onnx / stop.onnx / normal.onnx', ip: 'project root', status: '加载中', className: 'training' },
    { name: '车牌识别模型', type: 'PaddleOCR TextRecognition Tiny', ip: 'PP-OCRv6_tiny_rec', status: '加载中', className: 'training' }
  ];
  state.models = [
    { name: 'best(1).onnx / hearmap.onnx / stop.onnx / normal.onnx', desc: '可在前端选择车牌、车辆、禁停区域、正常区域检测模型；normal 模式会启用手动道路 ROI、连续车道线绘制，并把 ROI 内车辆标记为正常车辆', score: '加载中', active: true },
    { name: 'PaddleOCR TextRecognition Tiny', desc: '只做车牌字符识别，不再做文字检测', score: '加载中', active: true },
    { name: '多帧投票纠错', desc: '短时车牌关联 + 六位主体格式过滤 + 整串/逐字符投票，稳定后统一补京', score: '启用', active: true },
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


async function refreshAdminSummary() {
  try { state.adminSummary = await fetchJson(`${BACKEND_URL}/api/admin/summary`); }
  catch (error) { state.adminSummary = { error: error.message, users: [], devices: [], models: [], whitelist: [], configs: [], logs: [] }; }
}
function renderAdminPanels() {
  const data = state.adminSummary || {};
  const auth = $('#authPanel');
  if (auth) auth.innerHTML = (data.users || []).length ? (data.users || []).map(u => `<div class="admin-row"><div><b>${u.display_name || u.username}</b><span>${u.username} · ${u.role}${u.last_login ? ' · 最近登录 ' + u.last_login : ''}</span></div><em>${u.role}</em></div>`).join('') : '<div class="admin-empty">暂无用户数据。默认账号：admin / admin123</div>';
  const whitelist = $('#whitelistPanel');
  if (whitelist) whitelist.innerHTML = (data.whitelist || []).length ? (data.whitelist || []).map(w => `<div class="admin-row"><div><b>${w.plate_no}</b><span>${w.owner || '未填写车主'} · ${w.note || '允许通行'}</span></div><em>${w.allow ? '允许' : '禁止'}</em></div>`).join('') : '<div class="admin-empty">暂无白名单。可录入答辩演示车辆。</div>';
  const cfg = $('#configPanel');
  if (cfg) cfg.innerHTML = `<h3>关键参数</h3>` + (data.configs || []).map(c => `<div class="admin-row"><div><b>${c.config_key}</b><span>${c.description || ''}</span></div><em>${c.config_value}</em></div>`).join('');
  const logs = $('#operationLogPanel');
  if (logs) logs.innerHTML = `<h3>操作审计</h3>` + ((data.logs || []).length ? (data.logs || []).map(l => `<div class="admin-row"><div><b>${l.action}</b><span>${l.username || 'system'} · ${l.created_at || ''}</span></div><em>${l.detail || ''}</em></div>`).join('') : '<div class="admin-empty">暂无操作日志。</div>');
  const maturity = $('#moduleMaturityPanel');
  if (maturity) {
    const modules = [['用户与权限','注册/登录/角色/审计已接入 SQLite'],['视频源与设备','本地视频、手机流、设备登记与状态表已接入'],['模型管理','best/hearmap/stop/normal 多模型与类别统一已接入'],['业务告警','禁停停车后 3 秒告警、违停历史持久化'],['数据记录','白名单、操作日志、违停事件、统计接口已接入'],['前端展示','实时视频、Canvas 框、热力图、侧边栏告警']];
    maturity.innerHTML = `<h3>一级模块完成度</h3>` + modules.map(([a,b]) => `<div class="admin-row"><div><b>${a}</b><span>${b}</span></div><em>已补齐</em></div>`).join('');
  }
}
async function loginSystem() {
  try {
    const data = await fetchJson(`${BACKEND_URL}/api/auth/login`, { method: 'POST', body: JSON.stringify({ username: $('#authUsername')?.value || '', password: $('#authPassword')?.value || '' }) });
    state.currentUser = data.user; pushEvent('用户登录', `${data.user.display_name || data.user.username} 已登录，角色 ${data.user.role}`, '低', 'low'); await refreshAdminSummary(); renderAll();
  } catch (e) { alert(`登录失败：${e.message}`); }
}
async function registerSystemUser() {
  try { await fetchJson(`${BACKEND_URL}/api/auth/register`, { method: 'POST', body: JSON.stringify({ username: $('#authUsername')?.value || '', password: $('#authPassword')?.value || '', role: $('#authRole')?.value || 'operator', display_name: $('#authUsername')?.value || '' }) }); await refreshAdminSummary(); renderAll(); }
  catch (e) { alert(`注册失败：${e.message}`); }
}
async function addWhitelistPlate() {
  const plate = $('#whitePlate')?.value || ''; if (!plate.trim()) { alert('请填写车牌号'); return; }
  try { await fetchJson(`${BACKEND_URL}/api/whitelist`, { method: 'POST', body: JSON.stringify({ plate_no: plate, owner: $('#whiteOwner')?.value || '', allow: true, note: '前端白名单管理录入' }) }); $('#whitePlate').value = ''; await refreshAdminSummary(); renderAll(); }
  catch (e) { alert(`保存白名单失败：${e.message}`); }
}

function pushEvent(title, message, level = '低', className = 'low') {
  state.events.unshift({ title, message, level, className, time: nowTime(), id: `${Date.now()}-${Math.random()}` });
  state.events = state.events.slice(0, 8);
  if (className === 'high') state.anomalyCount += 1;
}

function setWorkflowRunning(isRunning, sourceText = '') {
  const img = $('#liveFeed');
  const mock = $('#mockRoadScene');
  const overlay = $('#detectionOverlay');
  state.workflowRunning = Boolean(isRunning);

  if (img && mock) {
    if (isRunning) {
      img.classList.remove('hidden');
      overlay?.classList.remove('hidden');
      mock.classList.add('hidden');

      // 只有从“未连接”进入“已连接”时才创建一次 MJPEG 请求。
      // 状态轮询和模型切换均不会重写 src，避免浏览器反复断开/重连后端视频流。
      if (!state.streamUiActive || !img.getAttribute('src')) {
        img.src = `${BACKEND_URL}/api/stream.mjpg?t=${Date.now()}`;
        state.streamUiActive = true;
      }
    } else {
      img.classList.add('hidden');
      overlay?.classList.add('hidden');
      clearDetectionOverlay();
      mock.classList.remove('hidden');
      img.removeAttribute('src');
      state.streamUiActive = false;
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
  state.perf = data.perf || {};
  const backendDetectorModel = status.detector_model || result.detector_model || state.detectorModel || 'plate';
  if (!state.modelSwitching) {
    state.detectorModel = backendDetectorModel;
    const modelSelect = $('#detectorModelSelect');
    if (modelSelect && modelSelect.value !== state.detectorModel) {
      modelSelect.value = state.detectorModel;
    }
    applyDetectorUi(state.detectorModel);
  }
  state.currentBoxes = pickBoxes(result);
  state.frameWidth = Number(result.frame_width || status.frame_width || state.frameWidth || 0);
  state.frameHeight = Number(result.frame_height || status.frame_height || state.frameHeight || 0);
  state.lastOverlayFrameId = Number(result.frame_id || state.lastOverlayFrameId || 0);
  state.roadMap = result.road_map || state.roadMap || { roads: [], assignments: [], heat: [] };
  state.roadAssignments = Array.isArray(result.road_assignments) ? result.road_assignments : (state.roadMap.assignments || []);
  state.roadHeat = Array.isArray(result.road_heat) ? result.road_heat : (state.roadMap.heat || []);
  state.parkingMonitor = result.parking_monitor || { active: [], alerts: [], zones: [], history: [] };
  state.normalLane = result.normal_lane || state.normalLane || {};
  state.normalRoadAnalysis = result.normal_road_analysis || state.normalRoadAnalysis || {};
  state.parkingHistory = Array.isArray(state.parkingMonitor.history) ? state.parkingMonitor.history : [];
  state.parkingAlerts = Number(result.parking_alert_count || (state.parkingMonitor.alerts || []).length || status.parking_alerts || 0);
  state.lastMessage = result.message || status.message || '后端运行中';
  state.fps = Number(status.display_fps || status.capture_fps || 0);
  const running = Boolean(status.running);
  state.workflowRunning = running;
  const source = status.source || '';

  const yoloReady = Boolean(status.yolo_ready);
  const ocrReady = Boolean(status.ocr_ready);
  const modelsReady = Boolean(status.models_ready);

  state.devices = [
    { name: '田浩阳手机采集端', type: 'IP Webcam / Tailscale', ip: $('#cameraUrlInput')?.value || '100.70.11.30', status: running && status.source_type === 'camera' ? '在线检测中' : '待连接', className: running && status.source_type === 'camera' ? '' : 'training' },
    { name: '李俊发分析节点', type: 'Electron + Python Backend', ip: BACKEND_URL.replace('http://', ''), status: status.backend || 'unknown', className: status.backend === 'ready' ? '' : 'training' },
    { name: 'ONNX检测模型', type: `${detectorModelLabel(status.detector_model || state.detectorModel)} / ONNXRuntime`, ip: status.model_path || 'project root', status: yoloReady ? `实际 ${status.yolo_provider || ''}` : '未就绪', className: yoloReady ? '' : 'offline' },
    { name: '车牌识别模型', type: 'PaddleOCR TextRecognition Tiny', ip: 'PP-OCRv6_tiny_rec', status: ocrReady ? `已加载 ${status.ocr_model || ''}` : '未就绪', className: ocrReady ? '' : 'offline' }
  ];

  state.models = [
    { name: detectorModelLabel(status.detector_model || state.detectorModel), desc: `实际 ${status.yolo_provider || 'provider未知'} · ${status.model_path || 'project root'} · YOLO ${status.yolo_ms || 0}ms`, score: yoloReady ? '已加载' : '未就绪', active: yoloReady },
    { name: 'PaddleOCR TextRecognition Tiny', desc: `轻量 OCR · 未稳定时最高约 4 次/秒，稳定后 2 秒复核 · ${status.ocr_ms || 0}ms`, score: ocrReady ? '已加载' : '未就绪', active: ocrReady },
    { name: '时序同步防拖影', desc: `模式：${status.sync_mode || 'auto'} · motion=${status.motion_score || 0} · YOLO实时框+OCR缓存复用`, score: '启用', active: true },
    { name: '性能监测诊断', desc: `pre ${status.yolo_pre_ms || 0}ms / infer ${status.yolo_infer_ms || 0}ms / post ${status.yolo_post_ms || 0}ms / encode ${status.encode_ms || 0}ms`, score: '监测中', active: true },
    { name: '多帧投票纠错', desc: '省份票 + 主体票 + 置信度权重，修正单帧波动', score: '启用', active: true },
    { name: 'OpenCV VideoCapture', desc: '本地视频 / IP Webcam 实时流', score: running ? '运行中' : '待启动', active: running }
  ];

  const plates = result.detector_model === 'vehicle' ? [] : (result.plates || []);
  const tracks = result.tracks || [];
  const detections = result.detections || [];

  if (String(result.detector_model || state.detectorModel).toLowerCase() === 'plate') {
    state.plateVotes = plates.map((p, index) => {
      const stable = Boolean(p.stable);
      const rawText = p.stable_text || p.plate_text || p.raw_text || '';
      const text = formatPlate(rawText);
      return {
        id: String(p.track_id || `PLATE-${p.track_num || index + 1}`),
        text,
        stable,
        votes: Number(p.votes || 0),
        confidence: Math.round(Number(p.stable_score || p.ocr_confidence || p.det_confidence || 0) * 100),
        pending: Boolean(p.ocr_pending),
        status: stable ? '投票稳定' : (text ? '候选投票中' : '等待 OCR')
      };
    }).filter(item => item.text || item.pending);
  } else {
    state.plateVotes = [];
  }
  state.vehicleCount = result.detector_model === 'vehicle' ? state.currentBoxes.length : detections.length;

  if (plates.length) {
    plates.forEach((p) => {
      const plateText = p.stable_text || p.plate_text || p.raw_text || '';
      if (!plateText) return;

      // 高频 OCR 会连续返回同一 track 的候选结果。
      // 表格按 track 更新同一行，而不是把每次候选都当成一辆新车插入。
      const trackKey = String(p.track_id || `PLATE-${p.track_num || plateText}`);
      const incomingStable = Boolean(p.stable);
      const entry = {
        key: trackKey,
        trackKey,
        time: nowTime(),
        plate: formatPlate(plateText),
        whitelist: true,
        decision: incomingStable ? '稳定识别' : '候选识别',
        confidence: Math.round((p.stable_score || p.ocr_confidence || 0) * 100),
        id: trackKey
      };

      const existingIndex = state.plates.findIndex(old => old.trackKey === trackKey || old.key === trackKey);
      if (existingIndex >= 0) {
        const existing = state.plates[existingIndex];
        // 稳定结果不能被后续低置信候选覆盖；候选阶段则持续展示最新结果。
        if (incomingStable || existing.decision !== '稳定识别') {
          state.plates[existingIndex] = { ...existing, ...entry };
        }
      } else {
        state.plates.unshift(entry);
      }
    });
    state.plates = state.plates.slice(0, 10);
    state.plateCount = Math.max(state.plateCount, state.plates.length);
  }

  const parkingActive = Array.isArray(state.parkingMonitor.active) ? state.parkingMonitor.active : [];
  if (parkingActive.length) {
    state.tracks = parkingActive.map(p => {
      const statusRaw = String(p.status || '').toLowerCase();
      const sourceLabel = p.source_label || (p.proxy_from === 'plate_number' ? '车牌代理' : '车辆框');
      const targetLabel = p.proxy_from === 'plate_number' ? '车牌代理车辆' : '车辆';
      const vehicleName = p.vehicle_name || p.associated_plate_text || p.track_id || `#${p.track_num || ''}`;
      const plateHint = p.associated_plate_track_id ? ` · 关联车牌=${p.associated_plate_track_id}` : '';
      let statusText = '禁停区内行驶';
      let zoneText = `${sourceLabel}进入禁停区，但仍在移动，暂不计时`;
      let secondsText = '0.0';
      if (p.alert) {
        statusText = '禁停告警';
        zoneText = `${p.zone_label || '禁停区域'} · ${sourceLabel}确认停车超过 ${Number(p.threshold_s || 3).toFixed(0)} 秒`;
        secondsText = Number(p.dwell_s || 0).toFixed(1);
      } else if (statusRaw === 'counting') {
        statusText = '停车计时';
        zoneText = `${p.zone_label || '禁停区域'} · ${sourceLabel}已停止，正在累计 3 秒阈值`;
        secondsText = Number(p.dwell_s || 0).toFixed(1);
      } else if (statusRaw === 'waiting_stable') {
        statusText = '静止确认';
        zoneText = `${p.zone_label || '禁停区域'} · 低速中，等待确认${targetLabel}真正停止`;
      } else if (statusRaw === 'lost_grace') {
        statusText = '短暂漏检';
        zoneText = `${p.zone_label || '禁停区域'} · 保留上一状态，防止单帧丢失闪烁`;
        secondsText = Number(p.dwell_s || 0).toFixed(1);
      }
      return {
        id: vehicleName,
        plate: targetLabel,
        zone: `${zoneText} · 证据=${sourceLabel}${plateHint} · 速度=${Number(p.speed_norm || 0).toFixed(4)}`,
        seconds: secondsText,
        status: statusText
      };
    });
  } else {
    const activeDetector = String(result.detector_model || state.detectorModel || '').toLowerCase();

    // 车牌识别模式只负责识别并展示车牌号，不生成任何可见的车辆/车牌跟踪卡片。
    // 后端仍可保留不可见的短时 OCR 关联，用于把同一块车牌的多次识别结果合并投票。
    if (activeDetector === 'plate') {
      state.tracks = [];
    } else {
      state.tracks = tracks.map(t => {
        const sem = String(t.semantic_type || '').toLowerCase();
        const isVehicle = sem === 'vehicle' || t.detector_model === 'vehicle' || state.detectorModel === 'vehicle';
        return {
          id: t.track_id,
          plate: isVehicle ? '车辆' : formatPlate(t.label),
          zone: isVehicle ? '车辆检测结果' : (t.stable ? '稳定车牌结果' : '候选车牌结果'),
          seconds: t.votes || 1,
          status: isVehicle ? '已检测' : (t.stable ? '已稳定' : '投票中')
        };
      });
    }
  }

  if (running) {
    setWorkflowRunning(true, source || state.lastMessage);
  } else if (!source) {
    setWorkflowRunning(false, state.lastMessage);
  }

  if (plates.length) {
    const stableCount = plates.filter(p => p.stable).length;
    if (stableCount) pushEvent('车牌稳定识别', `多帧投票已稳定输出 ${stableCount} 个车牌结果`, '低', 'low');
  }

  const parkingAlerts = Array.isArray(state.parkingMonitor.alerts) ? state.parkingMonitor.alerts : [];
  parkingAlerts.forEach((a) => {
    const key = a.event_id || `${a.track_id}-${Math.floor(Number(a.dwell_s || 0))}`;
    if (!state.parkingAlertKeys.has(key)) {
      state.parkingAlertKeys.add(key);
      pushEvent('禁停告警', `${a.vehicle_name || a.track_id || '车辆'} 已停止在 ${a.zone_label || '禁停区域'} ${Number(a.dwell_s || 0).toFixed(1)} 秒，超过 ${Number(a.threshold_s || 3).toFixed(0)} 秒阈值`, '高', 'high');
    }
  });

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


function buildConfigForMode(mode) {
  if (mode === 'fixed') {
    return {
      detector_model: getSelectedDetectorModel(),
      yolo_interval: 0.12,
      ocr_min_interval: 0.25,
      camera_track_ttl: 0.75,
      file_track_ttl: 1.8,
      camera_display_ttl: 0.65,
      file_display_ttl: 1.2,
      camera_max_frame_lag: 8,
      file_max_frame_lag: 25,
      camera_ocr_result_max_age: 1.4,
      file_ocr_result_max_age: 3.0,
      motion_reset_enabled: false,
      enable_tensorrt: false,
      trt_fp16_enable: true,
      trt_engine_cache_enable: true,
      perf_monitor_enabled: true,
      server_overlay: false,
      overlay_debug_text: false
    };
  }
  return {
    detector_model: getSelectedDetectorModel(),
    yolo_interval: 0.08,
    ocr_min_interval: 0.25,
    camera_track_ttl: 0.45,
    file_track_ttl: 1.6,
    camera_display_ttl: 0.35,
    file_display_ttl: 1.2,
    camera_max_frame_lag: 4,
    file_max_frame_lag: 25,
    camera_ocr_result_max_age: 0.85,
    file_ocr_result_max_age: 3.0,
    motion_reset_enabled: true,
    motion_reset_score: 22.0,
    motion_reset_min_interval: 0.75,
    enable_tensorrt: false,
    trt_fp16_enable: true,
    trt_engine_cache_enable: true,
    perf_monitor_enabled: true,
    server_overlay: false,
    overlay_debug_text: false
  };
}

async function applySyncModeConfig(mode) {
  const patch = buildConfigForMode(mode);
  await fetchJson(`${BACKEND_URL}/api/config`, {
    method: 'POST',
    body: JSON.stringify(patch)
  });
  pushEvent('检测配置已应用', `${detectorModelLabel(getSelectedDetectorModel())} · ${mode === 'fixed' ? '固定机位：保留更长 track。' : '手持防拖影：YOLO框实时显示。'}`, '低', 'low');
}

async function chooseVideoFile() {
  try {
    const filePath = await window.trafficDesk?.selectVideoFile?.();
    if (!filePath) return;
    const previousPath = state.selectedVideoPath;
    state.selectedVideoPath = filePath;
    $('#videoFilePath').value = filePath;
    if (previousPath !== filePath && state.normalRoi.sourceType === 'file') {
      invalidateNormalRoi('视频已更换，请重新选择 ROI');
    }
    pushEvent('已选择本地视频', filePath, '低', 'low');
    renderAll();
  } catch (error) {
    pushEvent('选择视频失败', error.message, '高', 'high');
  }
}


function isNormalDetectorSelected() {
  return getSelectedDetectorModel() === 'normal';
}

function normalRoiSourceValue(sourceType) {
  if (sourceType === 'camera') {
    return ($('#cameraUrlInput')?.value || '').trim();
  }
  return ($('#videoFilePath')?.value || state.selectedVideoPath || '').trim();
}

function invalidateNormalRoi(message = '尚未选择 ROI') {
  state.normalRoi.confirmed = false;
  state.normalRoi.source = '';
  state.normalRoi.points = [];
  state.normalRoi.draftPoints = [];
  state.normalRoi.frameWidth = 0;
  state.normalRoi.frameHeight = 0;
  state.normalRoi.imageDataUrl = '';
  state.normalRoi.previewImage = null;
  state.normalRoi.drawRect = null;
  const status = $('#normalRoiStatus');
  if (status) {
    status.textContent = message;
    status.classList.remove('ready');
  }
}

function updateNormalRoiControls() {
  const controls = $('#normalRoiControls');
  const normalSelected = isNormalDetectorSelected();
  controls?.classList.toggle('hidden', !normalSelected);

  const status = $('#normalRoiStatus');
  if (!status) return;
  if (!normalSelected) {
    status.textContent = '仅 normal.onnx 模式可用';
    status.classList.remove('ready');
  } else if (state.normalRoi.confirmed) {
    status.textContent = `ROI 已确认 · ${state.normalRoi.points.length} 个点`;
    status.classList.add('ready');
  } else {
    status.textContent = '尚未选择 ROI';
    status.classList.remove('ready');
  }
}

function closeNormalRoiModal() {
  const modal = $('#normalRoiModal');
  modal?.classList.add('hidden');
  modal?.setAttribute('aria-hidden', 'true');
}

function resizeNormalRoiCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const bw = Math.max(1, Math.round(width * dpr));
  const bh = Math.max(1, Math.round(height * dpr));
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height };
}

function drawNormalRoiSelector() {
  const canvas = $('#normalRoiCanvas');
  const image = state.normalRoi.previewImage;
  if (!canvas || !image) return;

  const { ctx, width, height } = resizeNormalRoiCanvas(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#050b14';
  ctx.fillRect(0, 0, width, height);

  const scale = Math.min(width / image.naturalWidth, height / image.naturalHeight);
  const drawW = image.naturalWidth * scale;
  const drawH = image.naturalHeight * scale;
  const drawX = (width - drawW) / 2;
  const drawY = (height - drawH) / 2;
  state.normalRoi.drawRect = { x: drawX, y: drawY, w: drawW, h: drawH };
  ctx.drawImage(image, drawX, drawY, drawW, drawH);

  const points = state.normalRoi.draftPoints || [];
  if (points.length) {
    ctx.save();
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = drawX + point[0] * drawW;
      const y = drawY + point[1] * drawH;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (points.length >= 3) ctx.closePath();
    ctx.fillStyle = 'rgba(250, 204, 21, .20)';
    if (points.length >= 3) ctx.fill();
    ctx.strokeStyle = '#fde047';
    ctx.lineWidth = 3;
    ctx.setLineDash(points.length >= 3 ? [] : [8, 6]);
    ctx.stroke();
    ctx.setLineDash([]);

    points.forEach((point, index) => {
      const x = drawX + point[0] * drawW;
      const y = drawY + point[1] * drawH;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fillStyle = '#fde047';
      ctx.fill();
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.font = '700 13px Microsoft YaHei, system-ui, sans-serif';
      ctx.fillText(String(index + 1), x + 10, y - 11);
    });
    ctx.restore();
  }

  const count = $('#normalRoiPointCount');
  if (count) count.textContent = `已选择 ${points.length} 个点`;
  const tip = $('#normalRoiTip');
  if (tip) tip.textContent = points.length >= 3
    ? '道路区域已闭合，可继续添加点或点击“确认道路区域”。'
    : '请沿道路边界依次点击，至少选择 3 个点。';
}

async function openNormalRoiSelector() {
  if (!isNormalDetectorSelected()) {
    alert('请先将检测模型切换为“正常道路检测 normal.onnx”。');
    return;
  }

  const sourceType = $('#normalRoiSourceSelect')?.value || 'file';
  const source = normalRoiSourceValue(sourceType);
  if (!source) {
    alert(sourceType === 'camera' ? '请先填写手机视频流地址。' : '请先选择本地视频文件。');
    return;
  }

  const status = $('#normalRoiStatus');
  if (status) {
    status.textContent = '正在读取预览帧...';
    status.classList.remove('ready');
  }

  try {
    const data = await fetchJson(`${BACKEND_URL}/api/normal/roi/preview`, {
      method: 'POST',
      body: JSON.stringify({ source, source_type: sourceType })
    });

    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error('ROI 预览图片加载失败'));
      image.src = data.image_data_url;
    });

    state.normalRoi.sourceType = sourceType;
    state.normalRoi.source = source;
    state.normalRoi.frameWidth = Number(data.frame_width || image.naturalWidth || 0);
    state.normalRoi.frameHeight = Number(data.frame_height || image.naturalHeight || 0);
    state.normalRoi.imageDataUrl = data.image_data_url;
    state.normalRoi.previewImage = image;
    state.normalRoi.draftPoints = [];

    const modal = $('#normalRoiModal');
    modal?.classList.remove('hidden');
    modal?.setAttribute('aria-hidden', 'false');
    requestAnimationFrame(drawNormalRoiSelector);
  } catch (error) {
    if (status) status.textContent = `预览失败：${error.message}`;
    pushEvent('正常道路 ROI 预览失败', error.message, '高', 'high');
  }
}

function onNormalRoiCanvasClick(event) {
  const canvas = $('#normalRoiCanvas');
  const drawRect = state.normalRoi.drawRect;
  if (!canvas || !drawRect) return;

  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  if (
    x < drawRect.x || x > drawRect.x + drawRect.w ||
    y < drawRect.y || y > drawRect.y + drawRect.h
  ) return;

  const nx = Math.max(0, Math.min(1, (x - drawRect.x) / drawRect.w));
  const ny = Math.max(0, Math.min(1, (y - drawRect.y) / drawRect.h));
  state.normalRoi.draftPoints.push([Number(nx.toFixed(7)), Number(ny.toFixed(7))]);
  drawNormalRoiSelector();
}

function undoNormalRoiPoint() {
  state.normalRoi.draftPoints.pop();
  drawNormalRoiSelector();
}

function resetNormalRoiPoints() {
  state.normalRoi.draftPoints = [];
  drawNormalRoiSelector();
}

async function configureNormalRoiOnBackend() {
  if (!state.normalRoi.confirmed || state.normalRoi.points.length < 3) return null;
  return fetchJson(`${BACKEND_URL}/api/normal/roi/configure`, {
    method: 'POST',
    body: JSON.stringify({ points: state.normalRoi.points })
  });
}

async function confirmNormalRoi() {
  const points = state.normalRoi.draftPoints || [];
  if (points.length < 3) {
    alert('正常道路 ROI 至少需要选择 3 个点。');
    return;
  }
  state.normalRoi.points = points.map(point => [...point]);
  state.normalRoi.confirmed = true;
  closeNormalRoiModal();
  updateNormalRoiControls();

  try {
    if (state.workflowRunning) {
      await configureNormalRoiOnBackend();
      pushEvent('道路区域已更新', `已选择 ${points.length} 个顶点；直接复用当前视频连接，下一帧开始道路分析。`, '低', 'low');
    } else {
      pushEvent('道路区域已确认', `已选择 ${points.length} 个顶点，启动视频后会用于 normal.onnx 检测。`, '低', 'low');
    }
  } catch (error) {
    pushEvent('道路 ROI 下发失败', error.message, '高', 'high');
  }
}

function requireNormalRoiForSource(sourceType, source) {
  if (!isNormalDetectorSelected()) return [];
  if (!state.normalRoi.confirmed || state.normalRoi.points.length < 3) {
    throw new Error('正常道路模式需要先点击“选择正常道路区域”，圈选至少 3 个点。');
  }
  if (state.normalRoi.sourceType !== sourceType || state.normalRoi.source !== source) {
    throw new Error('当前 ROI 与即将启动的视频来源不一致，请重新选择正常道路区域。');
  }
  return state.normalRoi.points.map(point => [...point]);
}

async function startVideoWorkflow() {
  try {
    const path = $('#videoFilePath').value || state.selectedVideoPath;
    if (!path) {
      pushEvent('缺少视频文件', '请先点击“选择视频”，再启动本地视频全流程检测。', '高', 'high');
      return;
    }
    const normalRoi = requireNormalRoiForSource('file', path);
    const mode = $('#syncModeSelect')?.value || 'fixed';
    await applySyncModeConfig(mode === 'handheld' ? 'fixed' : mode);
    await fetchJson(`${BACKEND_URL}/api/start/video`, {
      method: 'POST',
      body: JSON.stringify({ path, normal_roi: normalRoi })
    });
    state.plates = [];
    state.tracks = [];
    state.plateCount = 0;
    pushEvent('本地视频工作流启动', `当前模型：${detectorModelLabel(getSelectedDetectorModel())}。开始执行视频帧 → ONNX 检测 → 前端实时画框。`, '低', 'low');
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
    const normalRoi = requireNormalRoiForSource('camera', url);
    const mode = $('#syncModeSelect')?.value || 'handheld';
    await applySyncModeConfig(mode);
    await fetchJson(`${BACKEND_URL}/api/start/camera`, {
      method: 'POST',
      body: JSON.stringify({ url, normal_roi: normalRoi })
    });
    state.plates = [];
    state.tracks = [];
    state.plateCount = 0;
    pushEvent('手机实时检测启动', `当前模型：${detectorModelLabel(getSelectedDetectorModel())}，正在连接手机视频流：${url}`, '低', 'low');
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


function clearDetectionOverlay() {
  const canvas = $('#detectionOverlay');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (ctx) ctx.clearRect(0, 0, canvas.width || 0, canvas.height || 0);
}

function resizeOverlayCanvas(canvas, stage) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(stage.clientWidth));
  const h = Math.max(1, Math.round(stage.clientHeight));
  const bw = Math.round(w * dpr);
  const bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) {
    canvas.width = bw;
    canvas.height = bh;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function getImageContainRect(stageW, stageH, srcW, srcH) {
  if (!srcW || !srcH) return { x: 0, y: 0, w: stageW, h: stageH, scale: 1 };
  const scale = Math.min(stageW / srcW, stageH / srcH);
  const w = srcW * scale;
  const h = srcH * scale;
  return { x: (stageW - w) / 2, y: (stageH - h) / 2, w, h, scale };
}


function drawNormalRoadOverlay(ctx, rect) {
  if (!isNormalDetectorSelected()) return;
  const lane = state.normalLane || {};
  const roi = Array.isArray(lane.roi) ? lane.roi : [];
  const laneLines = Array.isArray(lane.lane_lines) ? lane.lane_lines : [];
  const candidateLines = Array.isArray(lane.candidate_lines) ? lane.candidate_lines : [];
  const mapX = (x) => rect.x + Number(x || 0) * rect.scale;
  const mapY = (y) => rect.y + Number(y || 0) * rect.scale;

  if (roi.length >= 3) {
    ctx.save();
    ctx.beginPath();
    roi.forEach((point, index) => {
      const x = mapX(Array.isArray(point) ? point[0] : point?.x);
      const y = mapY(Array.isArray(point) ? point[1] : point?.y);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = 'rgba(250, 204, 21, .08)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(250, 204, 21, .95)';
    ctx.lineWidth = 2.2;
    ctx.setLineDash([9, 6]);
    ctx.stroke();
    ctx.restore();
  }

  // 候选线段仅作淡色调试底图，最终稳定车道线使用粗线覆盖。
  ctx.save();
  ctx.strokeStyle = 'rgba(56, 189, 248, .30)';
  ctx.lineWidth = 1.2;
  candidateLines.slice(0, 120).forEach(line => {
    ctx.beginPath();
    ctx.moveTo(mapX(line.x1), mapY(line.y1));
    ctx.lineTo(mapX(line.x2), mapY(line.y2));
    ctx.stroke();
  });
  ctx.restore();

  const colors = ['#39ff88', '#38bdf8', '#f472d0', '#fde047', '#a78bfa', '#22d3ee'];
  laneLines.forEach((line, index) => {
    const color = colors[index % colors.length];
    const x1 = mapX(line.x1);
    const y1 = mapY(line.y1);
    const x2 = mapX(line.x2);
    const y2 = mapY(line.y2);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = line.matched === false ? 4 : 7;
    ctx.lineCap = 'round';
    ctx.shadowColor = color;
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = color;
    ctx.font = '700 12px Microsoft YaHei, system-ui, sans-serif';
    ctx.fillText(`L${line.track_id || index + 1}`, x1 + 6, Math.max(8, y1 + 6));
    ctx.restore();
  });
}

function drawDetectionOverlay() {
  const canvas = $('#detectionOverlay');
  const stage = $('#videoStage');
  const img = $('#liveFeed');

  if (!canvas || !stage || !img || img.classList.contains('hidden')) {
    clearDetectionOverlay();
    return;
  }

  const { ctx, w: stageW, h: stageH } = resizeOverlayCanvas(canvas, stage);
  ctx.clearRect(0, 0, stageW, stageH);

  const boxes = state.currentBoxes || [];
  const srcW = state.frameWidth || img.naturalWidth || 0;
  const srcH = state.frameHeight || img.naturalHeight || 0;

  if (!srcW || !srcH) return;

  const rect = getImageContainRect(stageW, stageH, srcW, srcH);
  drawNormalRoadOverlay(ctx, rect);

  ctx.save();
  ctx.font = '700 13px Microsoft YaHei, system-ui, sans-serif';
  ctx.textBaseline = 'top';

  for (const box of boxes) {
    const bb = box.bbox || [];
    if (bb.length < 4) continue;

    const [x1, y1, x2, y2] = bb.map(Number);

    const px = rect.x + x1 * rect.scale;
    const py = rect.y + y1 * rect.scale;
    const pw = Math.max(2, (x2 - x1) * rect.scale);
    const ph = Math.max(2, (y2 - y1) * rect.scale);

    const sem = String(box.semantic_type || '').toLowerCase();

    const rawName = String(
      box.class_name ||
      box.raw_class_name ||
      box.class_display_name ||
      box.display_label ||
      ''
    ).replace(/[\s_-]/g, '').toLowerCase();

    const isVehicleBox =
      sem === 'vehicle' ||
      rawName === 'car' ||
      rawName === 'vehicle' ||
      rawName === 'bus' ||
      rawName === 'truck' ||
      box.detector_model === 'vehicle';

    const isPlateLikeBox =
      sem === 'plate_number' ||
      rawName === 'carnumber' ||
      rawName === 'licenseplate' ||
      rawName === 'plate';

    const isStopBox =
      sem === 'no_parking' ||
      rawName === 'noparking' ||
      rawName === 'stop' ||
      rawName.includes('noparking');

    const isNormalBox =
      sem === 'normal_zone' ||
      rawName === 'normal' ||
      rawName === 'normalzone';

    const stable = Boolean(box.stable || box.stable_text);

    const isNormalVehicle = box.normal_road_status === 'normal_vehicle';

    const color = isStopBox
      ? '#fb7185'
      : isNormalVehicle
        ? '#22c55e'
        : isNormalBox
          ? '#34d399'
          : isVehicleBox
            ? '#38bdf8'
          : isPlateLikeBox
            ? '#facc15'
            : stable
              ? '#22c55e'
              : '#facc15';

    const ocrText =
      box.stable_text ||
      box.plate_text ||
      box.raw_text ||
      box.text ||
      '';

    const text = ocrText || box.label || '';

    const rawClassName = String(
      box.class_display_name ||
      box.class_name ||
      box.display_label ||
      box.model_class_name ||
      ''
    ).trim();

    const genericName = isStopBox
      ? '禁停区域'
      : isNormalVehicle
        ? '正常车辆'
        : isNormalBox
          ? '正常区域'
          : isVehicleBox
            ? '车辆'
          : isPlateLikeBox
            ? '车牌区域'
            : '';

    const clsName = rawClassName || genericName || '目标';

    const scoreText = Number(
      box.det_confidence ||
      box.confidence ||
      0
    ).toFixed(2);

    const trackText = `#${box.track_num || box.track_id || ''}`;

    let label = '';

    if (isNormalVehicle) {
      label = `${trackText} 正常车辆 YOLO ${scoreText}`;
    } else if (isVehicleBox || isStopBox || isNormalBox) {
      label = `${trackText} ${clsName} YOLO ${scoreText}`;
    } else if (isPlateLikeBox) {
      const fixedPlateText = fixDemoPlateText(ocrText);

      // 车牌识别模式只显示车牌号，不显示 #1、PLATE-001、票数或跟踪状态。
      label = fixedPlateText || '车牌识别中';
    } else {
      const fixedPlateText = fixDemoPlateText(text);
      const ocrScore = Number(box.ocr_confidence || box.stable_score || 0);

      label = fixedPlateText
        ? `${fixedPlateText} ${ocrScore ? ocrScore.toFixed(2) : ''}`
        : `${trackText} ${clsName} YOLO ${scoreText}`;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = stable ? 2.4 : 2;
    ctx.shadowColor = stable
      ? 'rgba(34,197,94,.35)'
      : 'rgba(250,204,21,.30)';
    ctx.shadowBlur = 8;
    ctx.strokeRect(px, py, pw, ph);
    ctx.shadowBlur = 0;

    const tx = Math.max(6, px);
    const ty = Math.max(6, py - 24);
    const metrics = ctx.measureText(label);
    const labelW = Math.min(stageW - tx - 6, metrics.width + 12);

    ctx.fillStyle = 'rgba(5, 12, 23, .82)';
    ctx.fillRect(tx, ty, labelW, 20);

    ctx.strokeStyle = color;
    ctx.strokeRect(tx, ty, labelW, 20);

    ctx.fillStyle = color;
    ctx.fillText(label, tx + 6, ty + 3, Math.max(40, labelW - 10));
  }

  ctx.restore();
}
async function refreshRealtimeOverlay() {
  const img = $('#liveFeed');
  if (!img || img.classList.contains('hidden')) return;
  try {
    const data = await fetchJson(`${BACKEND_URL}/api/latest`);
    const result = data.result || {};
    const status = data.status || {};
    if (!state.modelSwitching) {
      state.detectorModel = status.detector_model || result.detector_model || state.detectorModel;
      applyDetectorUi(state.detectorModel);
    }
    state.currentBoxes = pickBoxes(result);
    state.vehicleCount = (state.detectorModel === 'vehicle') ? state.currentBoxes.length : ((result.detections || result.plates || []).length);
    state.frameWidth = Number(result.frame_width || status.frame_width || state.frameWidth || 0);
    state.frameHeight = Number(result.frame_height || status.frame_height || state.frameHeight || 0);
    state.lastOverlayFrameId = Number(result.frame_id || state.lastOverlayFrameId || 0);
    state.normalLane = result.normal_lane || state.normalLane || {};
    state.normalRoadAnalysis = result.normal_road_analysis || state.normalRoadAnalysis || {};
    drawDetectionOverlay();
    renderNormalRoadRuntime();
  } catch (_) {
    // 高频 overlay 拉取失败不打断主状态轮询。
  }
}


function renderNormalRoadRuntime() {
  const target = $('#normalRoadRuntime');
  if (!target) return;

  const lane = state.normalLane || {};
  const analysis = state.normalRoadAnalysis || {};
  const normalSelected = isNormalDetectorSelected();

  if (!normalSelected) {
    target.innerHTML = `
      <div class="normal-road-runtime-title">正常道路模式状态</div>
      <div class="normal-road-runtime-body">当前检测模型不是 normal.onnx，车道线算法完全停用，不占用其他模式资源。</div>
    `;
    target.classList.remove('active', 'warning');
    return;
  }

  const roiReady = Boolean(analysis.roi_ready || (Array.isArray(lane.roi) && lane.roi.length >= 3) || state.normalRoi.confirmed);
  const stableCount = Number(lane.stable_lane_count || analysis.stable_lane_count || 0);
  const vehicleCount = Number(analysis.normal_vehicle_count || 0);
  const unclassifiedCount = Number(analysis.inside_unclassified_count || 0);
  const processingMs = Number(lane.processing_ms || 0).toFixed(1);

  target.classList.toggle('active', roiReady && stableCount > 0);
  target.classList.toggle('warning', roiReady && stableCount === 0);
  target.innerHTML = `
    <div class="normal-road-runtime-title">正常道路模式状态</div>
    <div class="normal-road-runtime-grid">
      <div><span>道路 ROI</span><b>${roiReady ? '已确认' : '未选择'}</b></div>
      <div><span>稳定车道线</span><b>${stableCount}</b></div>
      <div><span>ROI 内正常车辆</span><b>${vehicleCount}</b></div>
      <div><span>其他模型目标</span><b>${unclassifiedCount}</b></div>
      <div><span>OpenCV 耗时</span><b>${processingMs} ms</b></div>
    </div>
    <div class="normal-road-runtime-body">${analysis.message || lane.message || '等待正常道路检测结果。'}</div>
  `;
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

function renderPlateVotes() {
  const target = $('#plateVoteList');
  if (!target) return;

  if (!state.plateVotes.length) {
    target.innerHTML = '<div class="plate-vote-empty">当前没有有效车牌候选。长度或格式不符合“1 位字母 + 5 位字母/数字”的 OCR 结果不会进入投票。</div>';
    return;
  }

  target.innerHTML = state.plateVotes.map(item => {
    const confidence = Math.max(0, Math.min(100, Number(item.confidence || 0)));
    const statusClass = item.stable ? 'stable' : 'candidate';
    const displayedText = item.text || '识别中';
    return `
      <div class="plate-vote-card ${statusClass}">
        <div class="plate-vote-main">
          <div class="plate-vote-number">${displayedText}</div>
          <div class="plate-vote-meta">${item.status} · 有效票 ${item.votes} · 置信度 ${confidence}%</div>
        </div>
        <div class="plate-vote-state">${item.stable ? '已稳定' : (item.pending ? 'OCR 中' : '投票中')}</div>
      </div>
    `;
  }).join('');
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
    const isAlert = item.status === '禁停告警';
    const isCounting = item.status === '停车计时';
    const cls = isAlert ? 'danger' : (isCounting || item.status === '静止确认' ? 'warn' : '');
    const unit = isAlert || isCounting ? '秒' : '';
    const value = isAlert || isCounting ? item.seconds : item.status;
    return `
      <div class="track-card ${isAlert ? 'danger-card' : ''}">
        <div>
          <div class="track-title">${item.id} · ${item.plate}</div>
          <div class="track-meta">${item.zone} · 状态：${item.status}</div>
        </div>
        <div class="countdown ${cls}">${value}${unit}</div>
      </div>
    `;
  }).join('') : (() => {
    const summary = (state.parkingMonitor && state.parkingMonitor.summary) || {};
    const zones = Number(summary.zone_count || (state.parkingMonitor.zones || []).length || 0);
    const cars = Number(summary.car_count || 0);
    const plates = Number(summary.plate_number_count || 0);
    const proxies = Number(summary.plate_proxy_count || 0);
    const hint = zones || cars || plates
      ? `禁停区 ${zones} 个 · car ${cars} 个 · carNumber ${plates} 个 · 车牌代理 ${proxies} 个。若 car 漏检，会用 carNumber 作为禁停侧边栏代理证据。`
      : 'stop.onnx 会先检测 car、carNumber 与 noParking；车辆进入禁停区后，只有真正停止才开始 3 秒计时，移动经过不计时。';
    return `<div class="track-card"><div><div class="track-title">暂无正在计时的禁停车辆</div><div class="track-meta">${hint}</div></div><div class="countdown">0</div></div>`;
  })();
}

function renderParkingMiniPanel() {
  const el = $('#parkingMiniPanel');
  if (!el) return;
  const summary = (state.parkingMonitor && state.parkingMonitor.summary) || {};
  const active = Array.isArray(state.parkingMonitor.active) ? state.parkingMonitor.active : [];
  const alerts = Array.isArray(state.parkingMonitor.alerts) ? state.parkingMonitor.alerts : [];
  const zones = Number(summary.zone_count || (state.parkingMonitor.zones || []).length || 0);
  const cars = Number(summary.car_count || 0);
  const plates = Number(summary.plate_number_count || 0);
  const proxies = Number(summary.plate_proxy_count || 0);

  let body = '';
  if (active.length) {
    body = active.slice(0, 3).map(p => {
      const st = String(p.status || '').toLowerCase();
      const cls = p.alert ? 'danger' : (st === 'counting' ? 'warn' : '');
      const name = p.vehicle_name || p.associated_plate_text || p.track_id || `#${p.track_num || ''}`;
      const src = p.source_label || (p.proxy_from === 'plate_number' ? '车牌代理' : '车辆框');
      const value = p.alert || st === 'counting' ? `${Number(p.dwell_s || 0).toFixed(1)}s / ${Number(p.threshold_s || 3).toFixed(0)}s` : (st === 'moving_in_zone' ? '移动中，不计时' : '静止确认');
      return `<div class="parking-mini-row ${cls}"><span>${name} · ${src}</span><b>${value}</b></div>`;
    }).join('');
  } else {
    body = `<div class="parking-mini-empty">禁停区 ${zones} · car ${cars} · carNumber ${plates} · 代理 ${proxies}<br>没有 car 时会用 carNumber 作为禁停计时代理证据。</div>`;
  }
  el.innerHTML = `<div class="parking-mini-title">禁停跟踪状态 <b>${alerts.length ? '告警 ' + alerts.length : '阈值 3s'}</b></div><div class="parking-mini-body">${body}</div>`;
}


function renderParkingHistoryPanel() {
  const el = $('#parkingHistoryPanel');
  if (!el) return;
  const history = Array.isArray(state.parkingHistory) ? state.parkingHistory : [];
  if (!history.length) {
    el.innerHTML = `
      <div class="parking-history-title">违停历史记录 <b>0</b></div>
      <div class="parking-history-empty">暂无已确认违停。车辆停车超过阈值后，会永久保留在这里。</div>
    `;
    return;
  }
  const rows = history.slice(0, 8).map(h => {
    const vehicle = h.vehicle_name || h.associated_plate_text || h.track_id || '未知车辆';
    const zone = h.zone_label || '禁停区域';
    const tm = h.created_time || '--:--:--';
    const dwell = Number(h.dwell_s || 0).toFixed(1);
    const plate = h.associated_plate_track_id ? `<em>关联车牌 ${h.associated_plate_track_id}</em>` : '';
    return `<div class="parking-history-row">
      <div><strong>${vehicle}</strong><span>${tm} · ${zone} · 停留 ${dwell}s ${plate}</span></div>
      <b>${h.event_id || 'PARK'}</b>
    </div>`;
  }).join('');
  el.innerHTML = `<div class="parking-history-title">违停历史记录 <b>${history.length}</b></div><div class="parking-history-list">${rows}</div>`;
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

function getStageStat(name) {
  const stage = state.perf?.stage_ms || {};
  return stage[name] || { avg: 0, p95: 0, last: 0, max: 0 };
}

function fmtMs(v) {
  const n = Number(v || 0);
  return n >= 100 ? `${n.toFixed(0)}ms` : `${n.toFixed(1)}ms`;
}

function renderPerfMonitoring() {
  const grid = $('#perfGrid');
  if (!grid) return;
  const items = [
    ['读取', getStageStat('capture_read_ms')],
    ['快照', getStageStat('frame_snapshot_ms')],
    ['YOLO前处理', getStageStat('yolo_pre_ms')],
    ['YOLO推理', getStageStat('yolo_infer_ms')],
    ['YOLO后处理', getStageStat('yolo_post_ms')],
    ['跟踪更新', getStageStat('track_update_ms')],
    ['OCR识别', getStageStat('ocr_recognize_ms')],
    ['画框叠加', getStageStat('overlay_ms')],
    ['JPEG编码', getStageStat('jpeg_encode_ms')]
  ];
  grid.innerHTML = items.map(([name, st]) => `
    <div class="perf-item">
      <span>${name}</span>
      <b>${fmtMs(st.avg)}</b>
      <em>p95 ${fmtMs(st.p95)} · last ${fmtMs(st.last)}</em>
    </div>
  `).join('');

  const advice = $('#perfAdvice');
  const tips = state.perf?.bottlenecks || ['等待连续运行 30 秒后生成瓶颈判断。'];
  const counters = state.perf?.counters || {};
  const queue = state.perf?.queue || {};
  const frame = state.perf?.frame || {};
  if (advice) {
    advice.innerHTML = `
      <div><b>诊断建议：</b>${tips.map(t => `<span>${t}</span>`).join('')}</div>
      <div class="perf-counters">OCR队列 ${queue.ocr_size || 0} · Pending ${queue.ocr_pending_tracks || 0} · OCR完成 ${counters.ocr_done || 0} · OCR丢弃 ${counters.ocr_queue_drop || 0} · 帧龄 ${fmtMs(frame.frame_age_ms || 0)}</div>
    `;
  }
}

function updateResourceBars() {
  const cpu = state.backendReady ? Math.min(95, 20 + state.vehicleCount * 5) : 0;
  const gpu = state.backendReady ? (state.models.some(m => String(m.desc).includes('TensorRT') || String(m.desc).includes('CUDA')) ? 55 : 8) : 0;
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


function heatColor(score, alpha = 0.9) {
  const s = Math.max(0, Math.min(1, Number(score || 0)));
  if (s >= 0.78) return `rgba(255,62,96,${alpha})`;
  if (s >= 0.52) return `rgba(255,166,77,${alpha})`;
  if (s >= 0.28) return `rgba(255,229,102,${alpha})`;
  if (s >= 0.10) return `rgba(72,216,255,${alpha})`;
  return `rgba(70,102,140,${alpha * 0.55})`;
}

function fitFrameToCanvas(w, h) {
  const fw = Number(state.frameWidth || 0);
  const fh = Number(state.frameHeight || 0);
  const pad = 18;
  let areaW = w - pad * 2;
  let areaH = h - pad * 2;
  let areaX = pad;
  let areaY = pad;
  if (fw > 0 && fh > 0) {
    const frameRatio = fw / fh;
    const areaRatio = areaW / areaH;
    if (frameRatio > areaRatio) {
      const fittedH = areaW / frameRatio;
      areaY = pad + (areaH - fittedH) / 2;
      areaH = fittedH;
    } else {
      const fittedW = areaH * frameRatio;
      areaX = pad + (areaW - fittedW) / 2;
      areaW = fittedW;
    }
  }
  return { x: areaX, y: areaY, w: areaW, h: areaH };
}

function drawRoundedRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.closePath();
}

function isHeatmapVehicleBox(box) {
  const sem = String(box?.semantic_type || '').toLowerCase();
  const raw = String(box?.class_name || box?.raw_class_name || box?.class_display_name || '').replace(/[\s_-]/g, '').toLowerCase();
  if (sem) return sem === 'vehicle';
  return raw === 'car' || raw === 'vehicle' || raw === 'bus' || raw === 'truck';
}

function getHeatmapPoints() {
  const roadMap = state.roadMap || {};
  const points = Array.isArray(roadMap.points) ? roadMap.points.filter(p => String(p.semantic_type || 'vehicle') === 'vehicle') : [];
  if (points.length) return points;
  const assignments = Array.isArray(state.roadAssignments) ? state.roadAssignments.filter(p => String(p.semantic_type || 'vehicle') === 'vehicle') : [];
  if (assignments.length) return assignments;
  return (state.currentBoxes || []).filter(isHeatmapVehicleBox).map((b, idx) => {
    const box = b.bbox || [];
    const fw = Number(state.frameWidth || 1);
    const fh = Number(state.frameHeight || 1);
    const x1 = Number(box[0] || 0);
    const y1 = Number(box[1] || 0);
    const x2 = Number(box[2] || 0);
    const y2 = Number(box[3] || 0);
    return {
      track_id: b.track_id || `VEH-${idx + 1}`,
      track_num: b.track_num || idx + 1,
      confidence: Number(b.det_confidence || b.confidence || 0.5),
      anchor_x_norm: ((x1 + x2) / 2) / fw,
      anchor_y_norm: y2 / fh,
      center_x_norm: ((x1 + x2) / 2) / fw,
      center_y_norm: ((y1 + y2) / 2) / fh,
      bbox_w_norm: Math.max(0.02, (x2 - x1) / fw),
      bbox_h_norm: Math.max(0.02, (y2 - y1) / fh),
      density_score: 0.45,
      semantic_type: 'vehicle',
    };
  });
}

function ensureHeatField() {
  const n = state.heatFieldCols * state.heatFieldRows;
  if (!state.heatField || state.heatField.length !== n) {
    state.heatField = new Float32Array(n);
    state.heatLastUpdate = performance.now();
  }
  return state.heatField;
}

function resetHeatFieldIfNeeded(points) {
  const field = ensureHeatField();
  // 每次绘制都先清空：热力图只反映当前检测到的车辆，不保留上一帧热点。
  field.fill(0);
}

function updateHeatField(points) {
  resetHeatFieldIfNeeded(points);
  const field = ensureHeatField();
  const cols = state.heatFieldCols;
  const rows = state.heatFieldRows;
  state.heatLastUpdate = performance.now();

  points.forEach((p) => {
    const xNorm = Math.max(0, Math.min(1, Number(p.anchor_x_norm ?? p.center_x_norm ?? 0.5)));
    const yNorm = Math.max(0, Math.min(1, Number(p.anchor_y_norm ?? p.center_y_norm ?? 0.5)));
    const weight = Math.max(0.25, Math.min(1.2, Number(p.density_score || p.weight || p.confidence || 0.5)));
    const rx = Math.max(3, Math.min(10, Math.round((Number(p.radius_norm || 0.07)) * cols * 1.25)));
    const ry = Math.max(3, Math.min(9, Math.round((Number(p.radius_norm || 0.07)) * rows * 1.25)));
    const cx = Math.round(xNorm * (cols - 1));
    const cy = Math.round(yNorm * (rows - 1));
    for (let yy = Math.max(0, cy - ry); yy <= Math.min(rows - 1, cy + ry); yy += 1) {
      for (let xx = Math.max(0, cx - rx); xx <= Math.min(cols - 1, cx + rx); xx += 1) {
        const dx = (xx - cx) / Math.max(1, rx);
        const dy = (yy - cy) / Math.max(1, ry);
        const g = Math.exp(-(dx * dx * 2.4 + dy * dy * 2.0));
        field[yy * cols + xx] = Math.min(1.65, field[yy * cols + xx] + g * weight * 0.22);
      }
    }
  });
  return field;
}

function sampleHeatField(field, nx, ny) {
  const cols = state.heatFieldCols;
  const rows = state.heatFieldRows;
  const x = Math.max(0, Math.min(cols - 1, nx * (cols - 1)));
  const y = Math.max(0, Math.min(rows - 1, ny * (rows - 1)));
  const x0 = Math.floor(x), x1 = Math.min(cols - 1, x0 + 1);
  const y0 = Math.floor(y), y1 = Math.min(rows - 1, y0 + 1);
  const tx = x - x0, ty = y - y0;
  const v00 = field[y0 * cols + x0] || 0;
  const v10 = field[y0 * cols + x1] || 0;
  const v01 = field[y1 * cols + x0] || 0;
  const v11 = field[y1 * cols + x1] || 0;
  return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
}

function drawHeatSurface(ctx, area, field) {
  const cols = state.heatFieldCols;
  const rows = state.heatFieldRows;
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  const cellW = area.w / cols;
  const cellH = area.h / rows;
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const v = Math.max(0, Math.min(1, field[y * cols + x] || 0));
      if (v < 0.045) continue;
      ctx.fillStyle = heatColor(v, Math.min(0.72, 0.10 + v * 0.72));
      ctx.fillRect(area.x + x * cellW - cellW * 0.15, area.y + y * cellH - cellH * 0.15, cellW * 1.35, cellH * 1.35);
    }
  }
  ctx.restore();
}

function drawHeatContours(ctx, area, field) {
  const levels = [0.18, 0.34, 0.52, 0.72];
  ctx.save();
  levels.forEach((level, idx) => {
    ctx.strokeStyle = idx >= 2 ? 'rgba(255,218,92,.45)' : 'rgba(72,216,255,.22)';
    ctx.lineWidth = idx >= 2 ? 1.3 : 0.9;
    ctx.setLineDash(idx % 2 ? [4, 5] : []);
    for (let j = 0; j <= 8; j += 1) {
      ctx.beginPath();
      let started = false;
      for (let i = 0; i <= 140; i += 1) {
        const nx = i / 140;
        const ny = j / 8;
        const v = sampleHeatField(field, nx, ny);
        if (v >= level) {
          const px = area.x + nx * area.w;
          const py = area.y + ny * area.h;
          if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
        } else if (started) {
          ctx.stroke();
          started = false;
          ctx.beginPath();
        }
      }
      if (started) ctx.stroke();
    }
  });
  ctx.setLineDash([]);
  ctx.restore();
}

function drawAdvancedHeatmapBackground(ctx, area) {
  const grad = ctx.createLinearGradient(area.x, area.y, area.x + area.w, area.y + area.h);
  grad.addColorStop(0, '#04101f');
  grad.addColorStop(0.45, '#071a31');
  grad.addColorStop(1, '#102a46');
  ctx.fillStyle = grad;
  ctx.fillRect(area.x, area.y, area.w, area.h);

  // 透视网格：强调空间投影，不代表道路线识别。
  ctx.save();
  ctx.strokeStyle = 'rgba(148,189,255,0.075)';
  ctx.lineWidth = 1;
  for (let i = -1; i <= 10; i += 1) {
    const x = area.x + area.w * (i / 9);
    ctx.beginPath();
    ctx.moveTo(x + area.w * 0.06, area.y);
    ctx.lineTo(x - area.w * 0.13, area.y + area.h);
    ctx.stroke();
  }
  for (let i = 0; i <= 7; i += 1) {
    const y = area.y + area.h * (i / 7);
    ctx.beginPath();
    ctx.moveTo(area.x, y);
    ctx.lineTo(area.x + area.w, y);
    ctx.stroke();
  }
  ctx.restore();
}

function drawHeatmap() {
  const canvas = $('#heatmapCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const bg = ctx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, '#05101f');
  bg.addColorStop(0.55, '#0b1d34');
  bg.addColorStop(1, '#0e2947');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);

  const area = fitFrameToCanvas(w, h);
  ctx.save();
  drawRoundedRect(ctx, area.x, area.y, area.w, area.h, 22);
  ctx.clip();
  drawAdvancedHeatmapBackground(ctx, area);

  const points = getHeatmapPoints();
  const field = updateHeatField(points);

  // 1. 当前帧热力场：车辆消失后热点立即清除。
  drawHeatSurface(ctx, area, field);

  // 2. 高斯核云团：保留车辆检测框的连续空间关系，视觉更像真实热区。
  ctx.globalCompositeOperation = 'lighter';
  points.forEach((p) => {
    const xNorm = Math.max(0, Math.min(1, Number(p.anchor_x_norm ?? p.center_x_norm ?? 0.5)));
    const yNorm = Math.max(0, Math.min(1, Number(p.anchor_y_norm ?? p.center_y_norm ?? 0.5)));
    const x = area.x + xNorm * area.w;
    const y = area.y + yNorm * area.h;
    const density = Math.max(0.18, Math.min(1, Number(p.density_score || p.weight || p.confidence || 0.45)));
    const bboxScale = Math.max(Number(p.bbox_w_norm || 0.04) * area.w, Number(p.bbox_h_norm || 0.04) * area.h);
    const radius = Math.max(48, Math.min(118, bboxScale * 2.05 + 28 + density * 28));
    const g = ctx.createRadialGradient(x, y, 1, x, y, radius);
    g.addColorStop(0, `rgba(255,255,235,${0.50 + density * 0.36})`);
    g.addColorStop(0.14, `rgba(255,231,95,${0.44 + density * 0.32})`);
    g.addColorStop(0.36, `rgba(255,84,112,${0.26 + density * 0.38})`);
    g.addColorStop(0.66, `rgba(70,220,255,${0.10 + density * 0.14})`);
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.ellipse(x, y, radius * 0.72, radius, 0, 0, Math.PI * 2);
    ctx.fill();
  });

  // 3. 等密度线/热区边界：让热力图看起来更像分析结果，而不是几个普通光圈。
  ctx.globalCompositeOperation = 'source-over';
  drawHeatContours(ctx, area, field);

  // 4. 车辆点位层。
  ctx.font = '800 12px Microsoft YaHei, system-ui, sans-serif';
  points.forEach((p) => {
    const xNorm = Math.max(0, Math.min(1, Number(p.anchor_x_norm ?? p.center_x_norm ?? 0.5)));
    const yNorm = Math.max(0, Math.min(1, Number(p.anchor_y_norm ?? p.center_y_norm ?? 0.5)));
    const x = area.x + xNorm * area.w;
    const y = area.y + yNorm * area.h;
    const density = Math.max(0.18, Math.min(1, Number(p.density_score || p.weight || p.confidence || 0.45)));
    const color = heatColor(density, 1);
    const idText = `#${p.track_num || String(p.track_id || '').replace(/\D+/g, '') || '?'}`;

    ctx.shadowColor = color;
    ctx.shadowBlur = 18;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 7.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(255,255,255,0.95)';
    ctx.lineWidth = 1.7;
    ctx.stroke();

    const label = `${idText} 车辆`;
    const tw = ctx.measureText(label).width + 14;
    const lx = Math.max(area.x + 4, Math.min(area.x + area.w - tw - 4, x + 10));
    const ly = Math.max(area.y + 8, Math.min(area.y + area.h - 24, y - 15));
    ctx.fillStyle = 'rgba(3,9,18,0.76)';
    drawRoundedRect(ctx, lx, ly, tw, 22, 6);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.fillText(label, lx + 7, ly + 15);
  });

  ctx.restore();

  // 外框与标题
  ctx.strokeStyle = 'rgba(125,180,255,0.26)';
  ctx.lineWidth = 1.1;
  drawRoundedRect(ctx, area.x, area.y, area.w, area.h, 22);
  ctx.stroke();

  const roadMap = state.roadMap || {};
  const summary = roadMap.summary || {};
  const vehicleCount = Number(summary.vehicle_count || points.length || state.vehicleCount || 0);
  const densityLabel = summary.density_label || (vehicleCount ? '实时密度' : '等待车辆模型');
  ctx.fillStyle = 'rgba(238,246,255,.95)';
  ctx.font = '800 13px Microsoft YaHei, system-ui, sans-serif';
  ctx.fillText('高级空间热力图', 18, h - 50);
  ctx.font = '12px Microsoft YaHei, system-ui, sans-serif';
  ctx.fillStyle = 'rgba(142,164,192,.95)';
  ctx.fillText(`车辆目标：${vehicleCount} · ${densityLabel} · 高斯扩散/时间衰减/热点聚类`, 18, h - 29);

  renderRoadStats();

  const level = $('#congestionLevel');
  if (!level) return;
  const densityLevel = summary.density_level || (vehicleCount >= 6 ? 'high' : vehicleCount >= 3 ? 'medium' : vehicleCount ? 'low' : 'waiting');
  if (densityLevel === 'high') {
    level.textContent = '高密度区域';
    level.className = 'severity high';
  } else if (densityLevel === 'medium') {
    level.textContent = '中等密度';
    level.className = 'severity medium';
  } else if (densityLevel === 'low') {
    level.textContent = '低密度';
    level.className = 'severity low';
  } else {
    level.textContent = '等待车辆模型';
    level.className = 'severity medium';
  }
}

function renderRoadStats() {
  const el = $('#roadStats');
  if (!el) return;
  const points = getHeatmapPoints();
  const roadMap = state.roadMap || {};
  const summary = roadMap.summary || {};
  const heat = Array.isArray(state.roadHeat) ? state.roadHeat : [];
  const hotspots = Array.isArray(roadMap.hotspots) ? roadMap.hotspots : [];
  if (!points.length) {
    el.innerHTML = '<div class="road-empty">选择“车辆检测 hearmap.onnx”并启动检测后，这里会按车辆框底部中心点生成高级空间热力图；选择 stop.onnx/normal.onnx 时主要观察左侧区域框。</div>';
    return;
  }
  const overviewHtml = `
    <div class="road-heat-row"><span>车辆目标</span><b>${summary.vehicle_count || points.length} 车</b></div>
    <div class="road-heat-row"><span>最高密度格</span><b>${summary.max_cell_count || Math.max(1, ...heat.map(h => Number(h.count || 0)))} 车</b></div>
    <div class="road-heat-row"><span>平均置信度</span><b>${Number(summary.avg_confidence || 0).toFixed(2)}</b></div>
    <div class="road-heat-row"><span>热力算法</span><b>KDE直投</b></div>
  `;
  const hotHtml = hotspots.slice(0, 3).map((h, idx) => `
      <div class="road-car-row hotspot-row">
        <span>热点 ${idx + 1}</span>
        <em>${h.count || 0} 车 · heat ${Number(h.heat_score || 0).toFixed(2)}</em>
      </div>
    `).join('');
  const vehicleHtml = points.slice(0, 10).map(p => {
    const idText = `#${p.track_num || String(p.track_id || '').replace(/\D+/g, '') || '?'}`;
    const x = Number(p.anchor_x_norm ?? p.center_x_norm ?? 0).toFixed(2);
    const y = Number(p.anchor_y_norm ?? p.center_y_norm ?? 0).toFixed(2);
    const dens = Number(p.density_score || 0).toFixed(2);
    return `
      <div class="road-car-row">
        <span>${idText} 车辆落点</span>
        <em>x ${x} · y ${y} · dens ${dens}</em>
      </div>
    `;
  }).join('');
  el.innerHTML = `
    <div class="road-stat-title">车辆热力统计</div>
    <div class="road-stat-grid">${overviewHtml}</div>
    ${hotHtml ? `<div class="road-car-list">${hotHtml}</div>` : ''}
    <div class="road-car-list">${vehicleHtml}</div>
  `;
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
  const config = SECTION_CONFIG[section] || SECTION_CONFIG.monitor;
  state.currentSection = section;
  document.body.dataset.activeSection = section;

  $$('.nav-item').forEach(btn => btn.classList.toggle('active', btn.dataset.section === section));

  $$('.panel').forEach(panel => {
    const supported = String(panel.dataset.panel || '').split(/\s+/).filter(Boolean);
    panel.classList.toggle('hidden', !supported.includes(section));
  });

  $$('[data-section-view]').forEach(element => {
    const supported = String(element.dataset.sectionView || '').split(/\s+/).filter(Boolean);
    element.classList.toggle('section-hidden', !supported.includes(section));
  });

  const title = $('#pageTitle');
  const description = $('#pageDescription');
  const eyebrow = $('#pageEyebrow');
  if (title) title.textContent = config.title;
  if (description) description.textContent = config.description;
  if (eyebrow) eyebrow.textContent = config.eyebrow;

  applyDetectorUi(state.detectorModel);
  updateNormalRoiControls();
  renderNormalRoadRuntime();

  requestAnimationFrame(() => {
    drawHeatmap();
    if (section === 'history') drawTrend();
    drawDetectionOverlay();
  });
}

function renderAll() {
  renderKpis();
  renderPlateTable();
  renderPlateVotes();
  renderEvents();
  renderTracks();
  renderParkingMiniPanel();
  renderParkingHistoryPanel();
  renderDevices();
  renderModels();
  renderPerfMonitoring();
  updateResourceBars();
  drawHeatmap();
  drawTrend();
  drawDetectionOverlay();
  renderNormalRoadRuntime();
  updateNormalRoiControls();
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
  $('#selectNormalRoiBtn')?.addEventListener('click', openNormalRoiSelector);
  $('#normalRoiCanvas')?.addEventListener('click', onNormalRoiCanvasClick);
  $('#normalRoiUndoBtn')?.addEventListener('click', undoNormalRoiPoint);
  $('#normalRoiResetBtn')?.addEventListener('click', resetNormalRoiPoints);
  $('#normalRoiConfirmBtn')?.addEventListener('click', confirmNormalRoi);
  $('#normalRoiCancelBtn')?.addEventListener('click', closeNormalRoiModal);
  $('#normalRoiCloseBtn')?.addEventListener('click', closeNormalRoiModal);
  $('#normalRoiModal')?.addEventListener('click', (event) => {
    if (event.target === event.currentTarget) closeNormalRoiModal();
  });
  $('#normalRoiSourceSelect')?.addEventListener('change', () => {
    invalidateNormalRoi('预览来源已切换，请重新选择 ROI');
    state.normalRoi.sourceType = $('#normalRoiSourceSelect')?.value || 'file';
    updateNormalRoiControls();
  });
  $('#cameraUrlInput')?.addEventListener('input', () => {
    if (state.normalRoi.sourceType === 'camera' && state.normalRoi.confirmed) {
      invalidateNormalRoi('视频流地址已变化，请重新选择 ROI');
      updateNormalRoiControls();
    }
  });
  $('#detectorModelSelect')?.addEventListener('change', async (event) => {
    await switchDetectorModel(event.target.value);
  });
  $('#resetMockBtn')?.addEventListener('click', () => {
    state.plates = [];
    state.tracks = [];
    state.plateCount = 0;
    state.vehicleCount = 0;
    state.roadMap = { roads: [], assignments: [], heat: [] };
    state.roadAssignments = [];
    state.roadHeat = [];
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
  drawDetectionOverlay();
  if (!$('#normalRoiModal')?.classList.contains('hidden')) drawNormalRoiSelector();
});

initData();
bindEvents();
updateSection(state.currentSection);
applyDetectorUi(state.detectorModel);
renderAll();
showRuntimeInfo();
refreshAdminSummary();
refreshBackendStatus();
setInterval(() => {
  state.tick += 1;
  refreshBackendStatus();
}, 700);

// 高频轻量拉取：只更新 Canvas 框，不重绘整个页面。
setInterval(refreshRealtimeOverlay, 100);




