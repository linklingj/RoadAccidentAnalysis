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
