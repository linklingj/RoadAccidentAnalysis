import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Vehicles are rendered from a low-poly GLTF car model (assets/coupe.glb); other
// classes (pedestrians, riders, trucks, buses) keep procedural stand-ins. Each
// object's forward axis is +Z and its base sits on Y = 0, so placing it at a
// ground point and yawing +Z toward the motion vector orients it correctly.

// Classes drawn with the GLB car model. Unknown classes fall back to the car
// too (they are most often generic vehicles); pedestrians/riders/trucks/buses
// keep their distinct procedural shapes below.
const GLB_CLASSES = new Set(['car']);
const GLB_MODEL_URL = 'assets/coupe.glb';
// Real-world car length the model is normalised to (its longest horizontal
// axis maps to this many metres) before per-instance styling.
const MODEL_TARGET_LENGTH_M = 4.4;
// Flip to -1 if the model ends up facing away from its direction of travel.
const MODEL_FORWARD_SIGN = 1;

// Procedural fallbacks. Dimensions are (width = lateral X, height = Y,
// length = forward Z) in metres; lengths are compact to read at scene scale.
const CLASS_SHAPES = {
  truck: { kind: 'box', size: [2.5, 3.0, 4.0] },
  bus: { kind: 'box', size: [2.6, 3.2, 5.5] },
  riders: { kind: 'box', size: [0.8, 1.7, 1.1] },
  person: { kind: 'capsule', radius: 0.3, height: 1.7 },
};
const FALLBACK_BOX = { kind: 'box', size: [1.8, 1.6, 2.25] };

// Mutable vehicle style, tweakable from the dev panel (F1). `color`/`lengthScale`
// apply to procedural shapes only; `scale` scales every object. DEFAULT_SCALE is
// the slider's resting value, used as the reference at which a GLB car renders at
// its real-world MODEL_TARGET_LENGTH_M.
const DEFAULT_SCALE = 0.6;
const STYLE = { color: '#ffffff', scale: DEFAULT_SCALE, lengthScale: 1.4 };

// Cached source model + a load promise so vehicles can be created synchronously
// once preloaded. Null until loaded (or if loading failed → procedural fallback).
let glbSource = null;
let glbPromise = null;

/**
 * Kick off (once) the GLB car model load and cache the result. Resolves whether
 * or not the load succeeds; on failure GLB classes fall back to a box.
 */
export function preloadVehicleModels() {
  if (glbPromise) return glbPromise;
  const loader = new GLTFLoader();
  glbPromise = loader
    .loadAsync(GLB_MODEL_URL)
    .then((gltf) => {
      glbSource = gltf.scene;
      return glbSource;
    })
    .catch((err) => {
      console.warn(`[vehicleFactory] failed to load ${GLB_MODEL_URL}:`, err);
      glbSource = null;
      return null;
    });
  return glbPromise;
}

export function setVehicleStyle(partial) {
  if (partial.color != null) STYLE.color = partial.color;
  if (partial.scale != null) STYLE.scale = partial.scale;
  if (partial.lengthScale != null) STYLE.lengthScale = partial.lengthScale;
}

// Apply the current style to one top-level vehicle object (a procedural Mesh or
// a GLB Group). Procedural meshes take the shared colour and a length-stretched
// scale; GLB cars keep their own materials and scale uniformly relative to the
// slider's resting value so the default renders at real-world size.
export function styleVehicle(obj) {
  if (!obj) return;
  if (obj.userData && obj.userData.isGLB) {
    const factor = DEFAULT_SCALE > 0 ? STYLE.scale / DEFAULT_SCALE : STYLE.scale;
    obj.scale.setScalar(factor);
    return;
  }
  if (obj.isMesh) {
    if (obj.material && obj.material.color) obj.material.color.set(STYLE.color);
    obj.scale.set(STYLE.scale, STYLE.scale, STYLE.scale * STYLE.lengthScale);
  }
}

// Re-apply the style to every top-level vehicle under the given groups (live
// update). Vehicles are direct children of these groups, so we don't recurse.
export function restyleVehicles(...groups) {
  for (const g of groups) {
    if (g) for (const child of g.children) styleVehicle(child);
  }
}

function shapeFor(className) {
  if (!className) return FALLBACK_BOX;
  return CLASS_SHAPES[className.toLowerCase()] || FALLBACK_BOX;
}

/**
 * Deterministic per-track colour (HSV hue stepped by 47°, S = 0.7, V = 1) so
 * vehicles and trajectory lines share the same palette.
 */
export function trackColor(trackId) {
  const hue = (((trackId * 47) % 360) + 360) % 360;
  return new THREE.Color().setRGB(...hsvToRgb(hue / 360, 0.7, 1.0));
}

/**
 * Build a vehicle for the given class. GLB classes clone the cached car model
 * (normalised to real-world size, centred, sitting on the ground, facing +Z);
 * other classes get a procedural mesh. The current dev style is applied.
 */
export function createVehicle(className, trackId) {
  const name = (className || '').toLowerCase();
  if (glbSource && (GLB_CLASSES.has(name) || !CLASS_SHAPES[name])) {
    const group = new THREE.Group();
    group.userData.isGLB = true;
    group.add(normalizeModel(glbSource.clone(true), MODEL_TARGET_LENGTH_M, MODEL_FORWARD_SIGN));
    group.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    styleVehicle(group);
    return group;
  }
  return createProcedural(shapeFor(className));
}

// Wrap a cloned model so its longest horizontal axis spans `targetLen` metres,
// its footprint is centred on the origin, its base rests on Y = 0, and its
// forward axis is +Z. Returns the wrapper group (normalisation scale lives here;
// per-instance style scale is applied to the enclosing group separately).
function normalizeModel(model, targetLen, forwardSign) {
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const lengthAxis = Math.max(size.x, size.z) || 1;
  const s = targetLen / lengthAxis;

  // Centre on X/Z and drop the base to Y = 0 (in the model's own units).
  model.position.set(-center.x, -box.min.y, -center.z);

  const wrap = new THREE.Group();
  wrap.add(model);
  wrap.scale.setScalar(s);
  // If the model's length runs along X, yaw it so it points down +Z.
  if (size.x > size.z) wrap.rotation.y = Math.PI / 2;
  if (forwardSign < 0) wrap.rotation.y += Math.PI;
  return wrap;
}

function createProcedural(shape) {
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
  geometry.translate(0, height / 2, 0); // origin at ground contact

  const material = new THREE.MeshStandardMaterial({
    color: STYLE.color,
    metalness: 0.1,
    roughness: 0.65,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  styleVehicle(mesh);
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
