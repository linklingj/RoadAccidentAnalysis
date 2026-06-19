import * as THREE from 'three';

// Procedural stand-ins for per-class vehicles/pedestrians. Each mesh's forward
// axis is +Z and its base sits on Y = 0, so placing it at a ground point
// (y = yOffset) and yawing +Z toward the motion vector orients it correctly.

// Dimensions are (width = lateral X, height = Y, length = forward Z) in metres.
// Lengths are intentionally compact (~half real-world) to read better at scene scale.
const CLASS_SHAPES = {
  car: { kind: 'box', size: [1.8, 1.6, 2.25] },
  truck: { kind: 'box', size: [2.5, 3.0, 4.0] },
  bus: { kind: 'box', size: [2.6, 3.2, 5.5] },
  riders: { kind: 'box', size: [0.8, 1.7, 1.1] },
  person: { kind: 'capsule', radius: 0.3, height: 1.7 },
};
const FALLBACK_SHAPE = { kind: 'box', size: [1.8, 1.6, 2.25] };

// All objects share one colour for a clean, uniform look.
const OBJECT_COLOR = 0xffffff;

function shapeFor(className) {
  if (!className) return FALLBACK_SHAPE;
  return CLASS_SHAPES[className.toLowerCase()] || FALLBACK_SHAPE;
}

/**
 * Deterministic per-track colour (HSV hue stepped by 47°, S = 0.7, V = 1) so
 * vehicles and trajectory lines share the same palette.
 */
export function trackColor(trackId) {
  const hue = (((trackId * 47) % 360) + 360) % 360;
  return new THREE.Color().setRGB(...hsvToRgb(hue / 360, 0.7, 1.0));
}

/** Unified object colour (white). */
export function objectColor() {
  return new THREE.Color(OBJECT_COLOR);
}

/**
 * Build a mesh for the given class. Geometry is translated up by half its height
 * so the mesh origin is the ground-contact point.
 */
export function createVehicle(className, trackId) {
  const shape = shapeFor(className);
  let geometry;
  let height;
  if (shape.kind === 'capsule') {
    height = shape.height;
    const cyl = Math.max(0.01, shape.height - 2 * shape.radius);
    geometry = new THREE.CapsuleGeometry(shape.radius, cyl, 6, 12);
  } else {
    const [w, h, l] = shape.size;
    height = h;
    geometry = new THREE.BoxGeometry(w, h, l);
  }
  geometry.translate(0, height / 2, 0);

  const material = new THREE.MeshStandardMaterial({
    color: objectColor(),
    metalness: 0.1,
    roughness: 0.65,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  return mesh;
}

// HSV -> RGB (h, s, v in [0,1]) returning [r, g, b] in [0,1], matching the BEV
// renderer / trajectory palette.
function hsvToRgb(h, s, v) {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}
