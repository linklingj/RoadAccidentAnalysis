import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { hasTimeline, describeScene } from './src/sceneData.js';
import { buildRoad, buildCrosswalk } from './src/roadBuilder.js';
import { placeObjects } from './src/objectPlacer.js';
import { buildTrajectories } from './src/trajectoryRenderer.js';
import { PlaybackController } from './src/playback.js';
import { ViewerUI } from './src/ui.js';

// ── Renderer / scene / camera ──────────────────────────────────────────────
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
scene.fog = new THREE.Fog(0x0e1116, 80, 400);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 2000);
camera.position.set(20, 24, 28);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495; // stay above the ground

// ── Lights / ground ────────────────────────────────────────────────────────
scene.add(new THREE.HemisphereLight(0xbcd2ff, 0x202428, 1.05));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(40, 80, 20);
scene.add(sun);

const grid = new THREE.GridHelper(400, 200, 0x2a3550, 0x1a2030);
grid.position.y = -0.01;
scene.add(grid);

// ── Content groups ─────────────────────────────────────────────────────────
const contentRoot = new THREE.Group();
scene.add(contentRoot);
let roadGroup = new THREE.Group();
let crosswalkGroup = new THREE.Group();
let staticGroup = new THREE.Group();
let trajectoryGroup = new THREE.Group();
const vehicleGroup = new THREE.Group();
contentRoot.add(roadGroup, crosswalkGroup, staticGroup, trajectoryGroup, vehicleGroup);

const playback = new PlaybackController(vehicleGroup);

let currentScene = null;
let sceneBounds = new THREE.Box3(new THREE.Vector3(-10, 0, -10), new THREE.Vector3(10, 0, 10));

// ── UI ─────────────────────────────────────────────────────────────────────
const ui = new ViewerUI({
  onFile: (file) => loadFromFile(file),
  onLoadSample: () => loadFromUrl('data/scene_data.json'),
  onMedia: (file, params) => inferMedia(file, params),
  onView: (preset) => applyView(preset),
  onTogglePlay: () => playback.toggle(),
  onSeek: (t01) => playback.seekNormalized(t01),
  onSpeedChange: (v) => { playback.playbackSpeed = v; },
});

// ── Scene application ──────────────────────────────────────────────────────
function applyScene(data) {
  currentScene = data;
  // Swap out the static geometry groups.
  contentRoot.remove(roadGroup, crosswalkGroup, staticGroup, trajectoryGroup);
  disposeGroup(roadGroup); disposeGroup(crosswalkGroup);
  disposeGroup(staticGroup); disposeGroup(trajectoryGroup);

  roadGroup = buildRoad(data.road_polygons);
  crosswalkGroup = buildCrosswalk(data.crosswalk_polygons);

  if (hasTimeline(data)) {
    staticGroup = new THREE.Group();
    trajectoryGroup = new THREE.Group();
    playback.initialize(data);
  } else {
    playback.clear();
    staticGroup = placeObjects(data.objects, data.trajectories);
    trajectoryGroup = buildTrajectories(data.trajectories);
  }
  contentRoot.add(roadGroup, crosswalkGroup, staticGroup, trajectoryGroup);

  sceneBounds = computeBounds();
  applyView('reset');
  ui.setInfo(describeScene(data));
  ui.setStatus('');
  syncSpeed();
}

function computeBounds() {
  const box = new THREE.Box3();
  box.expandByObject(roadGroup);
  box.expandByObject(crosswalkGroup);
  box.expandByObject(staticGroup);
  box.expandByObject(vehicleGroup);
  if (box.isEmpty()) box.set(new THREE.Vector3(-10, 0, -10), new THREE.Vector3(10, 0, 10));
  return box;
}

// ── View presets ───────────────────────────────────────────────────────────
function applyView(preset) {
  const center = sceneBounds.getCenter(new THREE.Vector3());
  const size = sceneBounds.getSize(new THREE.Vector3());
  const radius = Math.max(8, 0.5 * Math.hypot(size.x, size.z));

  if (preset === 'top') {
    camera.position.set(center.x, center.y + radius * 2.2, center.z + 0.001);
  } else if (preset === 'analysis' && currentScene && currentScene.camera) {
    const h = currentScene.camera.height_m || 6;
    camera.position.set(0, Math.max(h, 2), -Math.max(2, radius * 0.2));
  } else {
    // reset / 3-4 orbit view
    camera.position.set(center.x + radius * 0.9, center.y + radius * 1.1, center.z - radius * 0.9);
  }
  controls.target.copy(center);
  controls.update();
}

// ── Loaders ────────────────────────────────────────────────────────────────
async function loadFromUrl(url) {
  ui.setStatus(`Loading ${url} …`);
  try {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    applyScene(await res.json());
  } catch (err) {
    ui.setStatus(`Failed to load ${url}: ${err.message}. Serve over http:// (not file://).`, true);
  }
}

function loadFromFile(file) {
  ui.setStatus(`Loading ${file.name} …`);
  const reader = new FileReader();
  reader.onload = () => {
    try {
      applyScene(JSON.parse(reader.result));
    } catch (err) {
      ui.setStatus(`Failed to parse ${file.name}: ${err.message}`, true);
    }
  };
  reader.onerror = () => ui.setStatus(`Failed to read ${file.name}`, true);
  reader.readAsText(file);
}

// ── Server inference: upload media → poll job → visualize returned scene ────
async function inferMedia(file, params) {
  const fd = new FormData();
  fd.append('file', file);
  if (params.cameraHeight) fd.append('camera_height', params.cameraHeight);
  if (params.ppm) fd.append('ppm', params.ppm);

  ui.setStatus(`업로드 중: ${file.name} …`, false, true);
  let job;
  try {
    const res = await fetch('api/infer', { method: 'POST', body: fd });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.error || `HTTP ${res.status}`);
    }
    job = await res.json();
  } catch (err) {
    ui.setStatus(`분석 요청 실패: ${err.message} (server.py 실행 여부 확인)`, true);
    return;
  }
  pollJob(job.job_id, file.name);
}

async function pollJob(jobId, name) {
  for (;;) {
    await sleep(1000);
    let j;
    try {
      const res = await fetch(`api/jobs/${jobId}`, { cache: 'no-store' });
      j = await res.json();
    } catch (err) {
      ui.setStatus(`상태 조회 실패: ${err.message}`, true);
      return;
    }
    if (j.status === 'queued' || j.status === 'running') {
      const secs = j.elapsed_sec != null ? `${j.elapsed_sec}s` : '';
      ui.setStatus(`분석 중 (${j.status})… ${secs}`, false, true);
      continue;
    }
    if (j.status === 'done') {
      applyScene(j.scene);
      ui.setStatus(`완료: ${name}${j.elapsed_sec != null ? ` (${j.elapsed_sec}s)` : ''}`);
      return;
    }
    ui.setStatus(`분석 오류: ${j.error || '알 수 없는 오류'}`, true);
    return;
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function syncSpeed() {
  const sel = document.getElementById('speed');
  if (sel) playback.playbackSpeed = Number(sel.value) || 1;
}

// ── Render loop ────────────────────────────────────────────────────────────
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  playback.update(dt);
  controls.update();
  ui.syncPlayback({
    hasTimeline: playback.hasTimeline,
    isPlaying: playback.isPlaying,
    currentTime: playback.currentTime,
    duration: playback.duration,
    normalized: playback.normalizedTime(),
  });
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function disposeGroup(group) {
  group.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
      else o.material.dispose();
    }
  });
}

// ── Initial load: ?scene=<url> override, else the bundled sample ────────────
const params = new URLSearchParams(location.search);
loadFromUrl(params.get('scene') || 'data/scene_data.json');

// Detect the inference server to enable the "analyze media" control.
fetch('api/health', { cache: 'no-store' })
  .then((r) => (r.ok ? r.json() : Promise.reject(new Error('no server'))))
  .then(() => ui.setServerAvailable(true))
  .catch(() => ui.setServerAvailable(false));
