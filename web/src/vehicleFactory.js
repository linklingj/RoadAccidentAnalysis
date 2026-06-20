import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Vehicles are rendered from GLTF models; other classes (pedestrians, riders)
// keep procedural stand-ins. Each object's forward axis is +Z and its base
// sits on Y = 0, so placing it at a ground point and yawing +Z toward the
// motion vector orients it correctly.

// Per-class GLB configs. `targetLen` is the real-world length (metres) the
// model is normalised to. `forwardSign` flips orientation if needed (-1).
// Unknown vehicle classes fall back to the car config.
const GLB_CONFIGS = {
  car:    { url: 'assets/coupe.glb', targetLen: 4.4, forwardSign: 1 },
  bus:    { url: 'assets/coupe.glb', targetLen: 5.5, forwardSign: 1 },
  truck:  { url: 'assets/armor.glb', targetLen: 4.0, forwardSign: 1 },
  person: { url: 'assets/human.glb', targetLen: 0.5, forwardSign: 1 },
  riders: { url: 'assets/bike.glb',  targetLen: 2.0, forwardSign: 1 },
};
const GLB_FALLBACK_CONFIG = GLB_CONFIGS.car;

// Cars are detected as black/white upstream (scene JSON `color`). The coupe GLB
// ships a single dark, palette-textured paint, so we can't just tint the
// material (black × any colour stays dark). Instead we rebuild the paint as a
// solid colour and drop the baked map, keeping wheels/glass distinct. One
// variant model is built per colour and reused; unknown colours fall back to the
// untouched GLB so legacy scenes look exactly as before.
const CAR_URL = GLB_CONFIGS.car.url;
const CAR_PAINT = {
  white: { color: 0xe9e9ec, metalness: 0.15, roughness: 0.5 },
  black: { color: 0x1b1b1f, metalness: 0.35, roughness: 0.45 },
};
const CAR_WHEEL = { color: 0x202022, metalness: 0.5, roughness: 0.55 };
const carVariants = {}; // colorKey ('white'|'black') → THREE.Group (recoloured source)

// Procedural fallbacks. Dimensions are (width = lateral X, height = Y,
// length = forward Z) in metres; lengths are compact to read at scene scale.
const CLASS_SHAPES = {
  riders: { kind: 'box', size: [0.8, 1.7, 1.1] },
  person: { kind: 'capsule', radius: 0.3, height: 1.7 },
};
const FALLBACK_BOX = { kind: 'box', size: [1.8, 1.6, 2.25] };

// Mutable vehicle style, tweakable from the dev panel (F1). `color`/`lengthScale`
// apply to procedural shapes only; `scale` scales every object. DEFAULT_SCALE is
// the slider's resting value, used as the reference at which a GLB car renders at
// its real-world MODEL_TARGET_LENGTH_M.
const DEFAULT_SCALE = 0.35;
const STYLE = { color: '#ffffff', scale: DEFAULT_SCALE, lengthScale: 1.4 };

// Per-scene normalization factor set by computeSceneVehicleScale() on each load.
// Multiplied into the rendered scale so vehicles appear consistently sized
// relative to the road regardless of PPM calibration differences.
let sceneNormScale = 1.0;

export function setSceneNormScale(factor) {
  sceneNormScale = (Number.isFinite(factor) && factor > 0) ? factor : 1.0;
}

// Cached source models keyed by URL. Null value = load failed.
const glbSources = {};   // url → THREE.Group | null
const glbLoadMap = {};   // url → Promise

/**
 * Kick off (once per URL) all GLB model loads and cache results. Resolves when
 * all loads finish; failed URLs leave their entry as null (procedural fallback).
 */
export function preloadVehicleModels() {
  const loader = new GLTFLoader();
  const uniqueUrls = [...new Set(Object.values(GLB_CONFIGS).map((c) => c.url))];
  const promises = uniqueUrls.map((url) => {
    if (url in glbLoadMap) return glbLoadMap[url];
    glbLoadMap[url] = loader
      .loadAsync(url)
      .then((gltf) => { glbSources[url] = gltf.scene; })
      .catch((err) => {
        console.warn(`[vehicleFactory] failed to load ${url}:`, err);
        glbSources[url] = null;
      });
    return glbLoadMap[url];
  });
  return Promise.all(promises);
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
// sceneNormScale is multiplied in so every scene auto-fits vehicle size to its
// spatial extent without touching the user's manual scale slider.
export function styleVehicle(obj) {
  if (!obj) return;
  if (obj.userData && obj.userData.isGLB) {
    const factor = DEFAULT_SCALE > 0 ? STYLE.scale / DEFAULT_SCALE : STYLE.scale;
    obj.scale.setScalar(factor * sceneNormScale);
    return;
  }
  if (obj.isMesh) {
    if (obj.material && obj.material.color) obj.material.color.set(STYLE.color);
    const s = STYLE.scale * sceneNormScale;
    obj.scale.set(s, s, s * STYLE.lengthScale);
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
 * Build a vehicle for the given class. Classes with a GLB_CONFIGS entry clone
 * their cached model (normalised to real-world size, centred, sitting on the
 * ground, facing +Z); unknown vehicle classes fall back to the car GLB; classes
 * with a procedural shape (person, riders) always get a procedural mesh.
 */
export function createVehicle(className, trackId, color) {
  const name = (className || '').toLowerCase();
  // person and riders always use procedural shapes
  if (!CLASS_SHAPES[name] || GLB_CONFIGS[name]) {
    const cfg = GLB_CONFIGS[name] || GLB_FALLBACK_CONFIG;
    // Cars get a black/white repaint; every other class uses the GLB as-is.
    const src = (name === 'car') ? carSourceFor(color) : glbSources[cfg.url];
    if (src) {
      const group = new THREE.Group();
      group.userData.isGLB = true;
      group.add(normalizeModel(src.clone(true), cfg.targetLen, cfg.forwardSign));
      group.traverse((o) => { if (o.isMesh) o.castShadow = true; });
      styleVehicle(group);
      return group;
    }
  }
  return createProcedural(shapeFor(className));
}

// Resolve the car source model for a detected colour. 'white'/'black' return a
// lazily-built, cached recoloured clone; anything else (missing colour, legacy
// scenes) returns the untouched GLB.
function carSourceFor(color) {
  const base = glbSources[CAR_URL];
  if (!base) return null;
  const key = (color === 'white' || color === 'black') ? color : null;
  if (!key) return base;
  if (!carVariants[key]) carVariants[key] = buildCarVariant(base, key);
  return carVariants[key];
}

// Deep-clone the car GLB and replace its paint/wheel materials with solid colours
// so the variant reads clearly as white or black. Glass is left intact. Cloned
// materials are owned by this variant, so per-colour instances never bleed into
// each other.
function buildCarVariant(base, key) {
  const root = base.clone(true);
  const paint = CAR_PAINT[key] || CAR_PAINT.black;
  root.traverse((o) => {
    if (!o.isMesh || !o.material) return;
    const matName = (o.material.name || '').toLowerCase();
    if (matName.includes('glass')) {
      o.material = o.material.clone(); // isolate from other variants/instances
      return;
    }
    // Wheel meshes are the cylinders; keep them dark in both variants.
    const isWheel = (o.name || '').toLowerCase().startsWith('cylinder');
    const spec = isWheel ? CAR_WHEEL : paint;
    o.material = new THREE.MeshStandardMaterial({
      color: spec.color,
      metalness: spec.metalness,
      roughness: spec.roughness,
    });
  });
  return root;
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
