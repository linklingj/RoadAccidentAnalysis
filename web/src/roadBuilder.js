import * as THREE from 'three';
import { signedArea } from './sceneData.js';

// Builds flat ground-plane meshes from world-space polygons. We triangulate with
// THREE.ShapeUtils.triangulateShape (earcut, handles concave polygons) and emit
// vertices directly at (x, yOffset, z) so the result stays consistent with
// object/trajectory placement.

const ROAD_OPTS = { color: 0x33373d, yOffset: 0.0, minArea: 0.5, roughness: 0.95 };
const CROSSWALK_OPTS = { color: 0xd5d8dd, yOffset: 0.02, minArea: 0.2, roughness: 0.7 };

export function buildRoad(polygons) {
  return buildPolygons(polygons, ROAD_OPTS, 'Road');
}

export function buildCrosswalk(polygons) {
  return buildPolygons(polygons, CROSSWALK_OPTS, 'Crosswalk');
}

function buildPolygons(polygons, opts, namePrefix) {
  const group = new THREE.Group();
  group.name = namePrefix + 'Group';
  if (!Array.isArray(polygons)) return group;

  const material = new THREE.MeshStandardMaterial({
    color: opts.color,
    roughness: opts.roughness,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });

  polygons.forEach((poly, i) => {
    const pts = poly && poly.points;
    if (!pts || pts.length < 3) return;
    if (Math.abs(signedArea(pts)) < opts.minArea) return;

    const geometry = triangulate(pts, opts.yOffset);
    if (!geometry) return;
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `${namePrefix}_${i}`;
    mesh.receiveShadow = true;
    group.add(mesh);
  });
  return group;
}

function triangulate(pts, yOffset) {
  const contour = pts.map((p) => new THREE.Vector2(p.x, p.z));
  const faces = THREE.ShapeUtils.triangulateShape(contour, []);
  if (!faces.length) return null;

  const positions = new Float32Array(pts.length * 3);
  for (let i = 0; i < pts.length; i++) {
    positions[i * 3] = pts[i].x;
    positions[i * 3 + 1] = yOffset;
    positions[i * 3 + 2] = pts[i].z;
  }
  const index = [];
  for (const f of faces) index.push(f[0], f[1], f[2]);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(index);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  return geometry;
}
