import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { hasTimeline, describeScene, objectWorld, computeSceneVehicleScale } from './src/sceneData.js';
import { buildRoad, buildCrosswalk } from './src/roadBuilder.js';
import { placeObjects } from './src/objectPlacer.js';
import { buildTrajectories } from './src/trajectoryRenderer.js';
import { setVehicleStyle, restyleVehicles, preloadVehicleModels, setSceneNormScale } from './src/vehicleFactory.js';
import { PlaybackController } from './src/playback.js';
import { ViewerUI } from './src/ui.js';

// ── Renderer / scene / camera ──────────────────────────────────────────────
// The viewer renders into the right half of the split stage (#app); the left
// half (#videoPane) plays the original media. All sizing is relative to #app.
const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1116);
scene.fog = new THREE.Fog(0x0e1116, 80, 400);

const DEFAULT_FOV = 55;
const camera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 0.1, 2000);
camera.position.set(20, 24, 28);

function appSize() {
  return {
    w: app.clientWidth || window.innerWidth,
    h: app.clientHeight || window.innerHeight,
  };
}
function resizeRenderer() {
  const { w, h } = appSize();
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
resizeRenderer();

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
// The pipeline exports world +X as "lateral right in the source image", but a
// three.js camera looking toward +Z sees its +X on the left, mirroring the
// reconstruction horizontally. Negating the content's X axis flips it back so
// the scene matches the original footage. three.js handles the reversed winding
// for the mirrored (negative-determinant) transform automatically.
const contentRoot = new THREE.Group();
contentRoot.scale.x = -1;
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
// Centroid of vehicle/trajectory positions — may differ from the geometric
// centre of road bounds when vehicles occupy only part of the visible road.
let contentCenter = new THREE.Vector3();

// ── Page navigation ────────────────────────────────────────────────────────
let serverAvailable = false;

function showSimulator() {
  document.body.classList.remove('on-landing');
}

function showLanding() {
  document.body.classList.add('on-landing');
}

// ── UI ─────────────────────────────────────────────────────────────────────
const ui = new ViewerUI({
  onFile: (file) => { showSimulator(); loadFromFile(file); },
  onMedia: (file, params) => { showSimulator(); inferMedia(file, params); },
  onView: (preset) => applyView(preset),
  onTogglePlay: () => playback.toggle(),
  onSeek: (t01) => playback.seekNormalized(t01),
  onSpeedChange: (v) => { playback.playbackSpeed = v; },
  onVehicleStyle: (style) => {
    setVehicleStyle(style);
    restyleVehicles(vehicleGroup, staticGroup);
  },
});

// ── Scene application ──────────────────────────────────────────────────────
function applyScene(data, media = null) {
  exitPOV();
  ui.setMedia(media);
  currentScene = data;
  // Swap out the static geometry groups.
  contentRoot.remove(roadGroup, crosswalkGroup, staticGroup, trajectoryGroup);
  disposeGroup(roadGroup); disposeGroup(crosswalkGroup);
  disposeGroup(staticGroup); disposeGroup(trajectoryGroup);

  roadGroup = buildRoad(data.road_polygons);
  crosswalkGroup = buildCrosswalk(data.crosswalk_polygons);

  // Auto-normalize vehicle scale based on detected vehicle size in this scene.
  const scaleInfo = computeSceneVehicleScale(data);
  setSceneNormScale(scaleInfo.multiplier);

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
  const summary = describeScene(data);
  summary.vehicleScale = scaleInfo;
  ui.setInfo(summary);
  ui.setStatus('');
  syncSpeed();
}

function computeBounds() {
  const box = new THREE.Box3();
  box.expandByObject(roadGroup);
  box.expandByObject(crosswalkGroup);
  box.expandByObject(staticGroup);
  box.expandByObject(vehicleGroup);

  // In video mode vehicleGroup only holds frame-0 positions (often empty).
  // Iterate every frame and every trajectory so bounds capture the full
  // spatial extent of vehicle movement throughout the clip.
  // contentRoot.scale.x = -1 mirrors scene X → world X = -scene X.
  const contentBox = new THREE.Box3();
  if (currentScene) {
    const expandContent = (xScene, zScene) => {
      const pt = new THREE.Vector3(-xScene, 0, zScene);
      box.expandByPoint(pt);
      contentBox.expandByPoint(pt);
    };
    for (const frame of (currentScene.frames || [])) {
      for (const obj of (frame.objects || [])) {
        const w = objectWorld(obj);
        expandContent(w.x, w.z);
      }
    }
    for (const traj of (currentScene.trajectories || [])) {
      for (const pt of (traj.points || [])) {
        expandContent(pt.x, pt.z);
      }
    }
    for (const obj of (currentScene.objects || [])) {
      const w = objectWorld(obj);
      expandContent(w.x, w.z);
    }
  }

  if (box.isEmpty()) box.set(new THREE.Vector3(-10, 0, -10), new THREE.Vector3(10, 0, 10));

  // contentCenter = where the action is; fall back to full-bounds centre
  // when there are no tracked objects (road-only scene).
  if (!contentBox.isEmpty()) {
    contentBox.getCenter(contentCenter).setY(0);
  } else {
    box.getCenter(contentCenter).setY(0);
  }

  return box;
}

// ── View presets ───────────────────────────────────────────────────────────
function applyView(preset) {
  exitPOV(); // any explicit view preset leaves the per-vehicle POV camera

  if (preset === 'analysis' && currentScene && currentScene.camera) {
    applyAnalysisCamera(currentScene.camera);
    return;
  }

  const size = sceneBounds.getSize(new THREE.Vector3());
  const radius = Math.max(8, 0.5 * Math.hypot(size.x, size.z));

  // Non-analysis presets use the default lens.
  if (camera.fov !== DEFAULT_FOV) {
    camera.fov = DEFAULT_FOV;
    camera.updateProjectionMatrix();
  }

  // tan(vFOV/2) — the minimum camera–target distance to keep the full scene
  // inside the vertical frustum is radius / fovHalfTan.
  const fovHalfTan = Math.tan(THREE.MathUtils.degToRad(DEFAULT_FOV / 2));

  if (preset === 'top') {
    // Height required to see every scene corner when looking down at
    // contentCenter (which may be offset from the geometric bounds centre).
    let maxCornerDist = 0;
    const cx = contentCenter.x, cz = contentCenter.z;
    for (const bx of [sceneBounds.min.x, sceneBounds.max.x]) {
      for (const bz of [sceneBounds.min.z, sceneBounds.max.z]) {
        const d = Math.hypot(bx - cx, bz - cz);
        if (d > maxCornerDist) maxCornerDist = d;
      }
    }
    const topH = Math.max(radius * 2.0, maxCornerDist / fovHalfTan) * 1.2;

    // Use a Z offset = topH * 0.01 to firmly pin the azimuth so the near end
    // of the road appears at the top of the screen, consistently regardless of
    // the previous orbit angle. Disable damping for two update() calls: the
    // first drains any accumulated spherical delta and zeroes it; the second
    // snaps the camera exactly into place with zero delta.
    const wasDamping = controls.enableDamping;
    controls.enableDamping = false;
    controls.target.copy(contentCenter);
    camera.position.set(contentCenter.x, contentCenter.y + topH, contentCenter.z + topH * 0.01);
    controls.update(); // applies + zeroes residual delta
    camera.position.set(contentCenter.x, contentCenter.y + topH, contentCenter.z + topH * 0.01);
    controls.update(); // zero delta → camera lands exactly here
    controls.enableDamping = wasDamping;

  } else {
    // reset: orbit view centred on contentCenter (where vehicles actually are),
    // at a distance that guarantees the full scene fits in the frustum.
    // Direction (~30° elevation, ~20° azimuth) — magnitude ≈ 1.
    const dist = (radius / fovHalfTan) * 1.3;
    camera.position.set(
      contentCenter.x + dist * 0.30,
      contentCenter.y + dist * 0.50,
      contentCenter.z - dist * 0.81,
    );
    controls.target.copy(contentCenter);
    controls.update();
  }
}

// Reproduce the CCTV vantage point. The pipeline projects everything relative to
// a camera at the world origin at `height_m`, looking forward (+Z) and pitched
// down by |pitch_deg|. We place the three.js camera there and aim it at the point
// where its view ray meets the ground — matching the former Unity sceneCameraAnchor
// (position = (0, height_m, 0); forward = (0, sin(pitch), cos(pitch))).
function applyAnalysisCamera(cam) {
  const h = cam.height_m > 0 ? cam.height_m : 6;
  const pitch = THREE.MathUtils.degToRad(cam.pitch_deg || 0);
  const sinP = Math.sin(pitch);
  const cosP = Math.cos(pitch);

  // Pull the viewpoint back slightly so the foreground road fits in frame.
  const pullBack = h * 1.2;
  camera.position.set(0, h, -pullBack);

  // Pivot for OrbitControls = where the view ray hits the ground (y = 0).
  // pitch < 0 ⇒ camera looks downward into +Z.
  let target;
  if (sinP < -1e-3) {
    const t = -h / sinP; // ray length from original CCTV position to the ground
    target = new THREE.Vector3(0, 0, cosP * t);
  } else {
    // (near-)level camera: just look forward a sensible distance.
    target = new THREE.Vector3(0, h, Math.max(20, h * 4));
  }

  // Match the lens to the estimated vertical FOV for faithful framing.
  const vfov = cam.vfov_deg;
  camera.fov = vfov > 10 && vfov < 120 ? vfov : DEFAULT_FOV;
  camera.updateProjectionMatrix();

  controls.target.copy(target);
  controls.update();
}

// ── Per-vehicle POV camera ──────────────────────────────────────────────────
// Click a vehicle to ride along from its viewpoint; click empty space (or pick
// any view preset) to return to free orbit. While following, OrbitControls is
// disabled and the camera is driven from the followed mesh every frame so it
// tracks the vehicle through the timeline.
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const POV_FOV = 70;
const POV_EYE_HEIGHT = 1.4;
let followTarget = null;
let pointerDown = null;

renderer.domElement.addEventListener('pointerdown', (e) => {
  pointerDown = { x: e.clientX, y: e.clientY, t: performance.now() };
});
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!pointerDown) return;
  const moved = Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y);
  const quick = performance.now() - pointerDown.t < 400;
  pointerDown = null;
  if (moved > 6 || !quick) return; // treat drags / long presses as orbit, not a pick
  pickVehicle(e);
});

function pickVehicle(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  const targets = [...vehicleGroup.children, ...staticGroup.children];
  const hits = raycaster.intersectObjects(targets, true);
  if (hits.length) enterPOV(topLevelVehicle(hits[0].object));
  else exitPOV();
}

// A pick can land on a GLB car's inner mesh; follow the top-level vehicle object
// (the direct child of vehicleGroup/staticGroup) so POV tracking and despawn
// detection work the same for procedural meshes and GLB groups.
function topLevelVehicle(obj) {
  let o = obj;
  while (o && o.parent && o.parent !== vehicleGroup && o.parent !== staticGroup) {
    o = o.parent;
  }
  return o;
}

function enterPOV(mesh) {
  followTarget = mesh;
  controls.enabled = false;
  camera.fov = POV_FOV;
  camera.updateProjectionMatrix();
  const label = mesh.name ? mesh.name.replace(/_/g, ' ') : '차량';
  ui.setStatus(`🚗 ${label} 시점 — 빈 공간 클릭 또는 시점 버튼으로 해제`);
}

function exitPOV() {
  if (!followTarget) return;
  followTarget = null;
  controls.enabled = true;
  if (camera.fov !== DEFAULT_FOV) {
    camera.fov = DEFAULT_FOV;
    camera.updateProjectionMatrix();
  }
  controls.target.copy(sceneBounds.getCenter(new THREE.Vector3()));
  controls.update();
  ui.setStatus('');
}

const _povPos = new THREE.Vector3();
const _povDir = new THREE.Vector3();
function updatePOVCamera() {
  if (!followTarget.parent) { exitPOV(); return; } // vehicle despawned
  if (followTarget.visible === false) return;       // inactive frame: hold pose

  followTarget.getWorldPosition(_povPos);
  followTarget.getWorldDirection(_povDir); // mesh +Z forward, in world space
  _povDir.y = 0;
  if (_povDir.lengthSq() < 1e-6) _povDir.set(0, 0, 1);
  _povDir.normalize();

  camera.position.set(_povPos.x, _povPos.y + POV_EYE_HEIGHT, _povPos.z);
  camera.lookAt(
    _povPos.x + _povDir.x * 12,
    POV_EYE_HEIGHT * 0.6,
    _povPos.z + _povDir.z * 12,
  );
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

  // Keep the uploaded file locally (object URL) to play as the original media in
  // the left pane — no need to round-trip it back from the server.
  const isVideo =
    (file.type && file.type.startsWith('video/')) ||
    /\.(mp4|avi|mov|mkv|webm|m4v)$/i.test(file.name);
  const media = { url: URL.createObjectURL(file), type: isVideo ? 'video' : 'image' };

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
    URL.revokeObjectURL(media.url);
    ui.setStatus(`분석 요청 실패: ${err.message} (server.py 실행 여부 확인)`, true);
    return;
  }
  pollJob(job.job_id, file.name, media);
}

async function pollJob(jobId, name, media = null) {
  for (;;) {
    await sleep(1000);
    let j;
    try {
      const res = await fetch(`api/jobs/${jobId}`, { cache: 'no-store' });
      j = await res.json();
    } catch (err) {
      if (media) URL.revokeObjectURL(media.url);
      ui.setStatus(`상태 조회 실패: ${err.message}`, true);
      return;
    }
    if (j.status === 'queued' || j.status === 'running') {
      const secs = j.elapsed_sec != null ? `${j.elapsed_sec}s` : '';
      ui.setStatus(`분석 중 (${j.status})… ${secs}`, false, true);
      continue;
    }
    if (j.status === 'done') {
      applyScene(j.scene, media);
      ui.setStatus(`완료: ${name}${j.elapsed_sec != null ? ` (${j.elapsed_sec}s)` : ''}`);
      return;
    }
    if (media) URL.revokeObjectURL(media.url);
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
  if (followTarget) updatePOVCamera(); // ride along the picked vehicle
  else controls.update();
  ui.syncPlayback({
    hasTimeline: playback.hasTimeline,
    isPlaying: playback.isPlaying,
    currentTime: playback.currentTime,
    duration: playback.duration,
    normalized: playback.normalizedTime(),
    speed: playback.playbackSpeed,
  });
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', resizeRenderer);
if (window.ResizeObserver) new ResizeObserver(resizeRenderer).observe(app);

function disposeGroup(group) {
  group.traverse((o) => {
    // GLB vehicles share geometry/materials with the cached source model; never
    // dispose those or later clones would reference freed GPU resources.
    if (isUnderGLB(o)) return;
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      if (Array.isArray(o.material)) o.material.forEach((m) => m.dispose());
      else o.material.dispose();
    }
  });
}

function isUnderGLB(o) {
  for (let p = o; p; p = p.parent) {
    if (p.userData && p.userData.isGLB) return true;
  }
  return false;
}

// ── Landing page wiring ─────────────────────────────────────────────────────
document.getElementById('homeBtn').addEventListener('click', showLanding);

// Sample cards
document.querySelectorAll('.sample-card').forEach((card) => {
  card.addEventListener('click', async () => {
    const sampleId = card.dataset.sample;
    showSimulator();
    await loadSample(sampleId);
  });
});

// Landing upload button
const landingMediaInput = document.getElementById('landingMediaInput');
const landingAnalyzeBtn = document.getElementById('landingAnalyzeBtn');
landingMediaInput.addEventListener('change', (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  landingMediaInput.value = '';
  if (!serverAvailable) {
    ui.setStatus('서버 모드 전용입니다 — `python server.py` 실행 후 사용하세요.', true);
    showSimulator();
    return;
  }
  showSimulator();
  inferMedia(file, {
    cameraHeight: Number(document.getElementById('landingCamH').value) || undefined,
    ppm: Number(document.getElementById('landingPpm').value) || undefined,
  });
});

async function loadSample(sampleId) {
  ui.setStatus(`샘플 로드 중…`, false, true);
  try {
    let sceneData = null;

    // Try server API first (fastest); fall back to static file.
    if (serverAvailable) {
      const res = await fetch(`api/samples/${sampleId}`, { cache: 'no-store' });
      if (res.ok) sceneData = await res.json();
    }
    if (!sceneData) {
      const res = await fetch('data/scene_data.json', { cache: 'no-cache' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      sceneData = await res.json();
    }

    // Probe for a matching video file (data/<sampleId>.mp4 / .webm).
    // Using a HEAD request avoids downloading the video just to check existence.
    let media = null;
    for (const ext of ['mp4', 'webm']) {
      const videoUrl = `data/${sampleId}.${ext}`;
      try {
        const vRes = await fetch(videoUrl, { method: 'HEAD', cache: 'no-store' });
        if (vRes.ok) { media = { url: videoUrl, type: 'video' }; break; }
      } catch (_) { /* not available */ }
    }

    applyScene(sceneData, media);
  } catch (err) {
    ui.setStatus(`샘플 로드 실패: ${err.message}`, true);
  }
}

// ── Initial load ────────────────────────────────────────────────────────────
// Preload GLB car model in the background; ?scene= query override skips landing.
const urlParams = new URLSearchParams(location.search);
const sceneOverride = urlParams.get('scene');
preloadVehicleModels().finally(() => {
  if (sceneOverride) {
    showSimulator();
    loadFromUrl(sceneOverride);
  }
  // else: stay on landing page, three.js renders the empty grid behind it
});

// Detect inference server: enable analyze button and sample API.
const serverStatusEl = document.getElementById('serverStatus');
fetch('api/health', { cache: 'no-store' })
  .then((r) => (r.ok ? r.json() : Promise.reject(new Error('no server'))))
  .then(() => {
    serverAvailable = true;
    ui.setServerAvailable(true);
    landingAnalyzeBtn.classList.remove('disabled');
    landingMediaInput.disabled = false;
    if (serverStatusEl) {
      serverStatusEl.textContent = '● 서버 연결됨';
      serverStatusEl.className = 'server-status ok';
    }
  })
  .catch(() => {
    serverAvailable = false;
    ui.setServerAvailable(false);
    landingAnalyzeBtn.classList.add('disabled');
    if (serverStatusEl) {
      serverStatusEl.textContent = '● 서버 없음 — 샘플만 이용 가능';
      serverStatusEl.className = 'server-status err';
    }
  });
