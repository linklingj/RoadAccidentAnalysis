import * as THREE from 'three';
import { createVehicle } from './vehicleFactory.js';
import { objectWorld, hasTimeline } from './sceneData.js';

// Video-timeline playback: builds a per-track timeline of smoothed world
// positions, spawns one vehicle per track, and on every frame interpolates
// position between the two bracketing keyframes while yawing the mesh toward a
// least-squares motion trend.

const FORWARD = new THREE.Vector3(0, 0, 1);

export class PlaybackController {
  constructor(parent) {
    this.parent = parent; // THREE.Group vehicles are added to

    // Tunables (playback + heading smoothing).
    this.autoPlay = true;
    this.playbackSpeed = 1;
    this.loop = true;
    this.yOffset = 0;
    this.headingWindowFrames = 15;
    this.headingTrendSamples = 10;
    this.headingSlewDegPerSec = 240;
    this.headingMinDisplacement = 0.05;

    this.hasTimeline = false;
    this.isPlaying = false;
    this.currentTime = 0;
    this.duration = 0;

    this._fps = 30;
    this._frameCount = 0;
    this._tracks = [];
  }

  initialize(data) {
    this.clear();
    if (!hasTimeline(data)) {
      this.hasTimeline = false;
      this.isPlaying = false;
      this.duration = 0;
      this.currentTime = 0;
      return;
    }

    this.hasTimeline = true;
    this._fps = data.fps > 0 ? data.fps : 30;
    this._frameCount = data.frames.length;
    this.duration = Math.max(0, (this._frameCount - 1) / this._fps);

    this._buildTrackTimelines(data);
    this._spawnVehicles();

    this.currentTime = 0;
    this._applyTime(0, true, 1 / this._fps);
    this.isPlaying = this.autoPlay;
  }

  _buildTrackTimelines(data) {
    const byId = new Map();

    // Seed from the registry so class_name is reliable even when a track is
    // missing from the first frame it appears in.
    if (Array.isArray(data.tracks)) {
      for (const t of data.tracks) {
        if (!t || byId.has(t.track_id)) continue;
        byId.set(t.track_id, newTimeline(t.track_id, t.class_name));
      }
    }

    for (const frame of data.frames) {
      if (!frame || !Array.isArray(frame.objects)) continue;
      const fi = frame.frame_index;
      for (const obj of frame.objects) {
        if (!obj || obj.track_id < 0) continue;
        let tl = byId.get(obj.track_id);
        if (!tl) {
          tl = newTimeline(obj.track_id, obj.class_name);
          byId.set(obj.track_id, tl);
        }
        if (!tl.className && obj.class_name) tl.className = obj.class_name;
        const w = objectWorld(obj, this.yOffset);
        tl.frames.push(fi);
        tl.positions.push(new THREE.Vector3(w.x, w.y, w.z));
        if (fi < tl.startFrame) tl.startFrame = fi;
        if (fi > tl.endFrame) tl.endFrame = fi;
      }
    }

    this._tracks = [];
    for (const tl of byId.values()) {
      if (tl.frames.length === 0) continue;
      this._tracks.push(tl);
    }
  }

  _spawnVehicles() {
    for (const track of this._tracks) {
      const mesh = createVehicle(track.className, track.trackId);
      mesh.name = `Vehicle_T${track.trackId}_${track.className}`;
      mesh.visible = false;
      track.instance = mesh;
      this.parent.add(mesh);
    }
  }

  update(dt) {
    if (!this.hasTimeline || !this.isPlaying) return;
    this.currentTime += dt * this.playbackSpeed;
    if (this.currentTime >= this.duration) {
      if (this.loop) {
        this.currentTime = this.duration > 0 ? this.currentTime % this.duration : 0;
      } else {
        this.currentTime = this.duration;
        this.isPlaying = false;
      }
    }
    this._applyTime(this.currentTime, false, dt);
  }

  play() {
    if (!this.hasTimeline) return;
    if (this.currentTime >= this.duration && !this.loop) this.currentTime = 0;
    this.isPlaying = true;
  }

  pause() {
    this.isPlaying = false;
  }

  toggle() {
    if (!this.hasTimeline) return;
    this.isPlaying ? this.pause() : this.play();
  }

  seekNormalized(t01) {
    if (!this.hasTimeline) return;
    t01 = Math.min(1, Math.max(0, t01));
    this.currentTime = this.duration * t01;
    this._applyTime(this.currentTime, true, 1 / this._fps);
  }

  normalizedTime() {
    return this.duration > 0 ? Math.min(1, this.currentTime / this.duration) : 0;
  }

  _applyTime(t, force, dt) {
    const frameFloat = t * this._fps;
    const frameLow = Math.floor(frameFloat);
    const frameHigh = Math.min(this._frameCount - 1, frameLow + 1);
    const alpha = Math.min(1, Math.max(0, frameFloat - frameLow));
    const slewStep = THREE.MathUtils.degToRad(this.headingSlewDegPerSec) * dt;

    for (const track of this._tracks) {
      const mesh = track.instance;
      if (!mesh) continue;

      const active = frameLow >= track.startFrame && frameLow <= track.endFrame + 1;
      if (!active) {
        mesh.visible = false;
        track.hasHeading = false;
        continue;
      }

      const posLow = this._sampleFrame(track, frameLow);
      if (!posLow) {
        mesh.visible = false;
        continue;
      }
      const posHigh = this._sampleFrame(track, frameHigh) || posLow;

      const worldPos = posLow.clone().lerp(posHigh, alpha);
      mesh.visible = true;

      const targetRot = this._computeHeading(track, frameFloat, worldPos);
      mesh.position.copy(worldPos);
      if (force) {
        mesh.quaternion.copy(targetRot);
      } else {
        mesh.quaternion.rotateTowards(targetRot, slewStep);
      }
    }
  }

  // Least-squares trend fit over the trailing window.
  _computeHeading(track, frameFloat, currentPos) {
    const window = Math.max(1, this.headingWindowFrames);
    const startFrame = frameFloat - window;
    const samples = Math.max(2, this.headingTrendSamples);

    const lo = Math.max(startFrame, track.startFrame);
    const hi = Math.min(frameFloat, track.endFrame);
    if (hi - lo < 1e-3) {
      return track.hasHeading ? track.lastHeading : IDENTITY.clone();
    }

    let sumF = 0, sumX = 0, sumZ = 0, sumFF = 0, sumFX = 0, sumFZ = 0, count = 0;
    let firstPos = currentPos, latestPos = currentPos;
    const tmp = new THREE.Vector3();
    for (let s = 0; s < samples; s++) {
      const tt = s / (samples - 1);
      const f = lo + (hi - lo) * tt;
      const p = this._sampleFrameFloat(track, f, tmp);
      if (!p) continue;
      if (count === 0) firstPos = p.clone();
      latestPos = p.clone();
      sumF += f; sumX += p.x; sumZ += p.z;
      sumFF += f * f; sumFX += f * p.x; sumFZ += f * p.z;
      count++;
    }
    if (count < 2) {
      return track.hasHeading ? track.lastHeading : IDENTITY.clone();
    }

    const n = count;
    const denom = n * sumFF - sumF * sumF;
    let dir;
    if (Math.abs(denom) < 1e-9) {
      dir = new THREE.Vector3(latestPos.x - firstPos.x, 0, latestPos.z - firstPos.z);
    } else {
      const slopeX = (n * sumFX - sumF * sumX) / denom;
      const slopeZ = (n * sumFZ - sumF * sumZ) / denom;
      dir = new THREE.Vector3(slopeX, 0, slopeZ);
    }

    const effectiveWindow = hi - lo;
    const displacementSq = dir.lengthSq() * effectiveWindow * effectiveWindow;
    if (displacementSq < this.headingMinDisplacement * this.headingMinDisplacement) {
      return track.hasHeading ? track.lastHeading : IDENTITY.clone();
    }

    const q = new THREE.Quaternion().setFromUnitVectors(FORWARD, dir.normalize());
    track.lastHeading = q;
    track.hasHeading = true;
    return q;
  }

  // Position at a fractional frame.
  _sampleFrameFloat(track, frame, out) {
    const low = Math.floor(frame);
    const a = this._sampleFrame(track, low);
    if (!a) return null;
    const b = this._sampleFrame(track, low + 1) || a;
    const alpha = Math.min(1, Math.max(0, frame - low));
    return (out || new THREE.Vector3()).copy(a).lerp(b, alpha);
  }

  // Position at an integer frame via binary search + linear interpolation.
  // Returns a Vector3 or null.
  _sampleFrame(track, frameIndex) {
    const fr = track.frames;
    const pos = track.positions;
    if (fr.length === 0) return null;
    if (frameIndex <= fr[0]) return pos[0];
    if (frameIndex >= fr[fr.length - 1]) return pos[fr.length - 1];

    let lo = 0, hi = fr.length - 1;
    while (lo + 1 < hi) {
      const mid = (lo + hi) >> 1;
      if (fr[mid] <= frameIndex) lo = mid; else hi = mid;
    }
    if (fr[lo] === frameIndex) return pos[lo];
    if (fr[hi] === frameIndex) return pos[hi];
    const a = (frameIndex - fr[lo]) / (fr[hi] - fr[lo]);
    return pos[lo].clone().lerp(pos[hi], a);
  }

  clear() {
    for (const track of this._tracks) {
      if (track.instance) {
        this.parent.remove(track.instance);
        disposeMesh(track.instance);
      }
    }
    this._tracks = [];
    this.hasTimeline = false;
    this.isPlaying = false;
  }
}

const IDENTITY = new THREE.Quaternion();

function newTimeline(trackId, className) {
  return {
    trackId,
    className: className || '',
    frames: [],
    positions: [],
    startFrame: Number.POSITIVE_INFINITY,
    endFrame: Number.NEGATIVE_INFINITY,
    instance: null,
    lastHeading: new THREE.Quaternion(),
    hasHeading: false,
  };
}

function disposeMesh(mesh) {
  if (mesh.geometry) mesh.geometry.dispose();
  if (mesh.material) mesh.material.dispose();
}
