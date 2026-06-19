import * as THREE from 'three';
import { createVehicle } from './vehicleFactory.js';
import { objectWorldRaw } from './sceneData.js';

// Static-mode placement (single image scenes): drop one vehicle per detected
// object, oriented along its trajectory when one is available.

const FORWARD = new THREE.Vector3(0, 0, 1);

export function placeObjects(objects, trajectories, yOffset = 0) {
  const group = new THREE.Group();
  group.name = 'ObjectGroup';
  if (!Array.isArray(objects)) return group;

  const trajLookup = new Map();
  if (Array.isArray(trajectories)) {
    for (const t of trajectories) {
      if (t) trajLookup.set(t.track_id, t);
    }
  }

  for (const obj of objects) {
    if (!obj) continue;
    const mesh = createVehicle(obj.class_name, obj.track_id);

    const w = objectWorldRaw(obj, yOffset);
    mesh.position.set(w.x, w.y, w.z);

    const traj = trajLookup.get(obj.track_id);
    if (traj && traj.points && traj.points.length >= 2) {
      const dir = computeForward(traj.points);
      if (dir.lengthSq() > 1e-4) {
        mesh.quaternion.setFromUnitVectors(FORWARD, dir.normalize());
      }
    }
    const tag = obj.track_id >= 0 ? 'T' + obj.track_id : 'det';
    mesh.name = `${obj.class_name}_${tag}`;
    group.add(mesh);
  }
  return group;
}

// Direction from the last trajectory segment.
function computeForward(pts) {
  const last = pts.length - 1;
  const prev = Math.max(0, last - 3);
  const a = pts[prev];
  const b = pts[last];
  return new THREE.Vector3(b.x - a.x, 0, b.z - a.z);
}
