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
    const mesh = createVehicle(obj.class_name, obj.track_id, obj.color);
    mesh.userData.trackId = obj.track_id >= 0 ? obj.track_id : null;
    mesh.userData.rawObj = obj;

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

// Robust direction from trajectory using circular-median outlier filtering.
function computeForward(pts) {
  if (pts.length < 2) return new THREE.Vector3();

  const stepAngles = [];
  const stepMags = [];
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i].x - pts[i - 1].x;
    const dz = pts[i].z - pts[i - 1].z;
    const magSq = dx * dx + dz * dz;
    if (magSq < 1e-8) continue;
    stepAngles.push(Math.atan2(dx, dz));
    stepMags.push(Math.sqrt(magSq));
  }
  if (stepAngles.length === 0) return new THREE.Vector3();

  // Circular median.
  const ref = stepAngles[0];
  const wrapped = stepAngles.map(a => {
    let d = a - ref;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    return ref + d;
  });
  const sortedW = [...wrapped].sort((a, b) => a - b);
  const medAngle = sortedW[Math.floor(sortedW.length / 2)];

  // Reject steps deviating more than 45° from the median.
  const THRESH = Math.PI / 4;
  let sumSin = 0, sumCos = 0;
  for (let i = 0; i < stepAngles.length; i++) {
    if (Math.abs(wrapped[i] - medAngle) > THRESH) continue;
    sumSin += Math.sin(stepAngles[i]) * stepMags[i];
    sumCos += Math.cos(stepAngles[i]) * stepMags[i];
  }
  if (sumSin === 0 && sumCos === 0) {
    sumSin = Math.sin(medAngle);
    sumCos = Math.cos(medAngle);
  }

  const finalAngle = Math.atan2(sumSin, sumCos);
  return new THREE.Vector3(Math.sin(finalAngle), 0, Math.cos(finalAngle));
}
