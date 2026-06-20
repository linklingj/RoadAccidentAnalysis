// Scene JSON helpers.
//
// The JSON contract is produced by infer.py (export_scene_json) in real-world
// metre coordinates: X = lateral (right), Z = forward depth, ground plane Y = 0.
// We map world (x_m, z_m) -> three.js (x, 0, z), Y-up, so the upstream BEV
// geometry math carries over unchanged.

/**
 * True when the scene carries a per-frame timeline (video mode).
 */
export function hasTimeline(data) {
  return !!data && Array.isArray(data.frames) && data.frames.length >= 2 && (data.fps || 0) > 0;
}

/**
 * Smoothed world position with raw fallback: legacy JSONs omit the smoothed
 * fields (parse to undefined/0), in which case we fall back to raw (x_m, z_m).
 */
export function objectWorld(obj, y = 0) {
  const sx = obj.x_m_smoothed || 0;
  const sz = obj.z_m_smoothed || 0;
  const rx = obj.x_m || 0;
  const rz = obj.z_m || 0;
  if (sx === 0 && sz === 0 && (rx !== 0 || rz !== 0)) {
    return { x: rx, y, z: rz };
  }
  return { x: sx, y, z: sz };
}

/** Raw (unsmoothed) world position. */
export function objectWorldRaw(obj, y = 0) {
  return { x: obj.x_m || 0, y, z: obj.z_m || 0 };
}

/** Shoelace signed area of a polygon given as [{x, z}, ...] (m^2). */
export function signedArea(points) {
  let a = 0;
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    a += points[i].x * points[j].z - points[j].x * points[i].z;
  }
  return a * 0.5;
}

/**
 * Estimate a normalization scale for vehicle models based on the scene's
 * detected vehicle dimensions (length_m when present) or position spread.
 * Returns { step (1-5), label, multiplier } where multiplier scales the
 * three.js vehicle model so its apparent size fits the scene geometry.
 *
 * Steps: 매우 작음 / 작음 / 보통 / 큼 / 매우 큼
 */
export function computeSceneVehicleScale(data) {
  // Primary: use length_m from car objects / frame objects.
  const lengths = [];
  for (const obj of (data.objects || [])) {
    if (obj.length_m != null && (obj.class_name || '').toLowerCase() === 'car') {
      lengths.push(obj.length_m);
    }
  }
  const frames = data.frames || [];
  const stride = Math.max(1, Math.floor(frames.length / 60));
  for (let i = 0; i < frames.length; i += stride) {
    for (const obj of (frames[i].objects || [])) {
      if (obj.length_m != null && (obj.class_name || '').toLowerCase() === 'car') {
        lengths.push(obj.length_m);
      }
    }
  }
  if (lengths.length >= 3) {
    lengths.sort((a, b) => a - b);
    const median = lengths[Math.floor(lengths.length / 2)];
    return _stepFromLength(median);
  }

  // Fallback: estimate from spread of all object positions.
  const pts = [];
  for (const obj of (data.objects || [])) {
    if (obj.x_m != null) pts.push([obj.x_m, obj.z_m]);
  }
  for (const traj of (data.trajectories || [])) {
    for (const p of (traj.points || [])) pts.push([p.x, p.z]);
  }
  for (let i = 0; i < frames.length; i += stride) {
    for (const obj of (frames[i].objects || [])) {
      if (obj.x_m != null) pts.push([obj.x_m, obj.z_m]);
    }
  }
  if (pts.length === 0) return { step: 3, label: '보통', multiplier: 0.65 };

  let cx = 0, cz = 0;
  for (const [x, z] of pts) { cx += x; cz += z; }
  cx /= pts.length; cz /= pts.length;
  const dists = pts
    .map(([x, z]) => Math.sqrt((x - cx) ** 2 + (z - cz) ** 2))
    .sort((a, b) => a - b);
  const mad = dists[Math.floor(dists.length / 2)];
  return _stepFromSpread(mad);
}

// MODEL_TARGET_LENGTH_M = 4.4 m (must match vehicleFactory.js constant).
const _MODEL_LEN = 4.4;

function _stepFromLength(medLen) {
  // multiplier = target render length / model length so the car fits its footprint.
  if (medLen < 1.5) return { step: 1, label: '매우 작음', multiplier: _snap(medLen / _MODEL_LEN, 0.30) };
  if (medLen < 2.5) return { step: 2, label: '작음',     multiplier: _snap(medLen / _MODEL_LEN, 0.50) };
  if (medLen < 3.5) return { step: 3, label: '보통',     multiplier: _snap(medLen / _MODEL_LEN, 0.65) };
  if (medLen < 4.5) return { step: 4, label: '큼',       multiplier: _snap(medLen / _MODEL_LEN, 0.85) };
  return               { step: 5, label: '매우 큼',     multiplier: _snap(medLen / _MODEL_LEN, 1.10) };
}

function _stepFromSpread(mad) {
  if (mad < 1.5) return { step: 1, label: '매우 작음', multiplier: 0.30 };
  if (mad < 4.0) return { step: 2, label: '작음',     multiplier: 0.50 };
  if (mad < 10)  return { step: 3, label: '보통',     multiplier: 0.65 };
  if (mad < 25)  return { step: 4, label: '큼',       multiplier: 0.85 };
  return           { step: 5, label: '매우 큼',     multiplier: 1.10 };
}

// Round to 2 decimal places and clamp to [0.20, 1.50].
function _snap(v, fallback) {
  if (!Number.isFinite(v) || v <= 0) return fallback;
  return Math.round(Math.min(1.50, Math.max(0.20, v)) * 100) / 100;
}

/** Quick scene summary for the info panel. */
export function describeScene(data) {
  return {
    mode: hasTimeline(data) ? 'video' : 'image',
    roads: (data.road_polygons || []).length,
    crosswalks: (data.crosswalk_polygons || []).length,
    objects: (data.objects || []).length,
    trajectories: (data.trajectories || []).length,
    tracks: (data.tracks || []).length,
    frames: (data.frames || []).length,
    fps: data.fps || 0,
    camera: data.camera || null,
  };
}
