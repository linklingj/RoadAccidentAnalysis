import * as THREE from 'three';
import { trackColor } from './vehicleFactory.js';

// Static-mode trajectory polylines: one coloured line per track, rendered
// slightly above the road plane.

const Y_OFFSET = 0.05;

export function buildTrajectories(trajectories) {
  const group = new THREE.Group();
  group.name = 'TrajectoryGroup';
  if (!Array.isArray(trajectories)) return group;

  for (const traj of trajectories) {
    const pts = traj && traj.points;
    if (!pts || pts.length < 2) continue;

    const positions = new Float32Array(pts.length * 3);
    for (let i = 0; i < pts.length; i++) {
      positions[i * 3] = pts[i].x;
      positions[i * 3 + 1] = Y_OFFSET;
      positions[i * 3 + 2] = pts[i].z;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({ color: trackColor(traj.track_id) });
    const line = new THREE.Line(geometry, material);
    line.name = 'Trajectory_T' + traj.track_id;
    group.add(line);
  }
  return group;
}
