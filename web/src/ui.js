// DOM glue: scene loading controls, the playback bar, the info panel and view
// presets. main.js owns the three.js side and passes in handlers; this module
// only touches the DOM.

export class ViewerUI {
  constructor(handlers) {
    this.h = handlers; // { onFile, onMedia, onView, onTogglePlay, onSeek, onSpeedChange, onVehicleStyle }
    this._draggingSlider = false;
    this.serverAvailable = false;

    this.$ = (id) => document.getElementById(id);
    this.statusEl = this.$('status');
    this.infoEl = this.$('info');
    this.playBtn = this.$('playBtn');
    this.slider = this.$('seek');
    this.timeLabel = this.$('timeLabel');
    this.speedSel = this.$('speed');
    this.mediaInput = this.$('mediaInput');
    this.analyzeBtn = this.$('analyzeBtn');
    this.camH = this.$('camH');
    this.ppm = this.$('ppm');
    this.playbackBar = this.$('playbackBar');
    this.devPanel = this.$('devPanel');
    this.devColor = this.$('devColor');
    this.vehScale = this.$('vehScale');
    this.devLength = this.$('devLength');
    this.vehScaleVal = this.$('vehScaleVal');
    this.devLengthVal = this.$('devLengthVal');
    this.devMode = this.$('devMode');
    this.devFrame = this.$('devFrame');

    this.vehicleInfo = this.$('vehicleInfo');
    this.viTrackId = this.$('viTrackId');
    this.viX = this.$('viX');
    this.viZ = this.$('viZ');
    this.viVx = this.$('viVx');
    this.viVz = this.$('viVz');

    // Left-pane original media (synced to the 3D timeline).
    this.mediaVideo = this.$('mediaVideo');
    this.mediaImage = this.$('mediaImage');
    this.mediaEmpty = this.$('mediaEmpty');
    this._mediaURL = null;   // active object URL (revoked when replaced)
    this._mediaType = null;  // 'video' | 'image' | null

    this._wire();
  }

  _wire() {
    this.mediaInput.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) {
        if (!this.serverAvailable) {
          this.setStatus('서버 모드 전용입니다 — `python server.py` 실행 후 사용하세요.', true);
        } else {
          this.h.onMedia(file, {
            cameraHeight: Number(this.camH.value) || undefined,
            ppm: Number(this.ppm.value) || undefined,
          });
        }
      }
      this.mediaInput.value = '';
    });

    this.playBtn.addEventListener('click', () => this.h.onTogglePlay());
    this.slider.addEventListener('pointerdown', () => { this._draggingSlider = true; });
    window.addEventListener('pointerup', () => { this._draggingSlider = false; });
    this.slider.addEventListener('input', () => {
      this.h.onSeek(Number(this.slider.value) / 1000);
    });
    this.speedSel.addEventListener('change', () => {
      this.h.onSpeedChange(Number(this.speedSel.value));
    });

    for (const btn of document.querySelectorAll('[data-view]')) {
      btn.addEventListener('click', () => this.h.onView(btn.dataset.view));
    }

    // F1 toggles the developer panel (live vehicle size / colour tweaks).
    window.addEventListener('keydown', (e) => {
      if (e.key === 'F1') { e.preventDefault(); this.toggleDev(); }
    });
    const emitStyle = () => this._emitVehicleStyle();
    this.devColor.addEventListener('input', emitStyle);
    this.vehScale.addEventListener('input', emitStyle);
    this.devLength.addEventListener('input', emitStyle);
    this.$('devReset').addEventListener('click', () => {
      this.devColor.value = '#ffffff';
      this.vehScale.value = '0.15';
      this.devLength.value = '1';
      this._emitVehicleStyle();
    });

    // Drag & drop a *_scene.json anywhere onto the page.
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    ['dragenter', 'dragover'].forEach((ev) => window.addEventListener(ev, (e) => {
      stop(e); document.body.classList.add('dragging');
    }));
    ['dragleave', 'drop'].forEach((ev) => window.addEventListener(ev, (e) => {
      stop(e);
      if (ev === 'drop') {
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) this._handleDroppedFile(file);
      }
      if (ev === 'dragleave' && e.relatedTarget) return;
      document.body.classList.remove('dragging');
    }));
  }

  // Route a dropped file by type: JSON → load scene; video/image → server inference.
  _handleDroppedFile(file) {
    const name = (file.name || '').toLowerCase();
    const isJson = file.type === 'application/json' || name.endsWith('.json');
    const isMedia =
      (file.type && (file.type.startsWith('video/') || file.type.startsWith('image/'))) ||
      /\.(mp4|avi|mov|mkv|webm|m4v|png|jpe?g|bmp|webp)$/.test(name);
    if (isJson) {
      this.h.onFile(file);
    } else if (isMedia) {
      if (!this.serverAvailable) {
        this.setStatus('영상/이미지 분석은 서버 모드 전용입니다 — `python server.py` 실행 후 사용하세요.', true);
      } else {
        this.h.onMedia(file, {
          cameraHeight: Number(this.camH.value) || undefined,
          ppm: Number(this.ppm.value) || undefined,
        });
      }
    } else {
      this.setStatus(`지원하지 않는 파일 형식입니다: ${file.name}`, true);
    }
  }

  toggleDev() {
    this.devPanel.classList.toggle('hidden');
  }

  _emitVehicleStyle() {
    const scale = Number(this.vehScale.value) || 1;
    const lengthScale = Number(this.devLength.value) || 1;
    this.vehScaleVal.textContent = `${scale.toFixed(2)}×`;
    this.devLengthVal.textContent = `${lengthScale.toFixed(2)}×`;
    this.h.onVehicleStyle({ color: this.devColor.value, scale, lengthScale });
  }

  setStatus(msg, isError = false, busy = false) {
    this.statusEl.innerHTML = msg
      ? (busy ? `<span class="spinner"></span> ${escapeHtml(msg)}` : escapeHtml(msg))
      : '';
    this.statusEl.classList.toggle('error', !!isError);
    this.statusEl.style.display = msg ? 'block' : 'none';
  }

  // Enable/disable the server-only "analyze media" control based on a health check.
  setServerAvailable(ok) {
    this.serverAvailable = ok;
    if (this.analyzeBtn) {
      this.analyzeBtn.classList.toggle('disabled', !ok);
      this.analyzeBtn.title = ok
        ? '영상/이미지를 서버에 업로드하여 추론'
        : '서버 모드 전용 — `python server.py` 실행 필요';
    }
    if (this.mediaInput) this.mediaInput.disabled = !ok;
  }

  // Show the original media in the left pane. `source` = { url, type } or null.
  // Takes ownership of object URLs, revoking the previous one on replace.
  setMedia(source) {
    if (this._mediaURL && this._mediaURL.startsWith('blob:') && this._mediaURL !== (source && source.url)) {
      URL.revokeObjectURL(this._mediaURL);
    }
    this._mediaURL = source ? source.url : null;
    this._mediaType = source ? source.type : null;

    // Stop/clear the video either way so a previous clip never keeps playing.
    this.mediaVideo.pause();
    this.mediaVideo.removeAttribute('src');
    this.mediaVideo.load();
    this.mediaVideo.style.display = 'none';
    this.mediaImage.removeAttribute('src');
    this.mediaImage.style.display = 'none';
    this.mediaEmpty.style.display = 'none';

    if (!source) {
      this.mediaEmpty.style.display = 'block';
    } else if (source.type === 'video') {
      this.mediaVideo.src = source.url;
      this.mediaVideo.muted = true;
      this.mediaVideo.style.display = 'block';
      this.mediaVideo.load();
    } else {
      this.mediaImage.src = source.url;
      this.mediaImage.style.display = 'block';
    }
  }

  // Keep the left-pane video aligned with the 3D timeline: match play state and
  // speed, and nudge currentTime only when it drifts (avoids constant seeking).
  _syncVideo(p) {
    const v = this.mediaVideo;
    if (this._mediaType !== 'video' || !v.src || !p.hasTimeline) return;

    if (Number.isFinite(p.speed) && p.speed > 0 && v.playbackRate !== p.speed) {
      try { v.playbackRate = p.speed; } catch (_) { /* unsupported rate */ }
    }
    if (p.isPlaying && v.paused) v.play().catch(() => {});
    if (!p.isPlaying && !v.paused) v.pause();

    if (v.readyState >= 1 && Number.isFinite(v.duration) && v.duration > 0) {
      const target = Math.max(0, Math.min(p.currentTime, v.duration - 0.05));
      if (Math.abs(v.currentTime - target) > 0.2) v.currentTime = target;
    }
  }

  setInfo(summary) {
    const cam = summary.camera || {};
    const vs = summary.vehicleScale;
    const vsLabel = vs
      ? `${vs.label} (×${vs.multiplier.toFixed(2)}, 단계 ${vs.step}/5)`
      : '—';
    const rows = [
      ['Roads', summary.roads],
      ['Crosswalks', summary.crosswalks],
      ['Objects', summary.objects],
      ['Tracks', summary.mode === 'video' ? summary.tracks : summary.trajectories],
      ['차량 크기', vsLabel],
      ['Camera height', cam.height_m != null ? `${cam.height_m} m` : '—'],
      ['Pitch / Roll', cam.pitch_deg != null ? `${cam.pitch_deg.toFixed(1)}° / ${cam.roll_deg.toFixed(1)}°` : '—'],
      ['vFOV', cam.vfov_deg != null ? `${cam.vfov_deg.toFixed(1)}°` : '—'],
    ];
    this.infoEl.innerHTML = rows
      .map(([k, v]) => `<div class="row"><span>${k}</span><b>${v}</b></div>`)
      .join('');
  }

  setDevMode(mode, frames, fps) {
    if (!this.devMode) return;
    this.devMode.textContent = mode === 'video'
      ? `video (${frames} fr @ ${fps.toFixed(1)} fps)`
      : 'image';
  }

  setDevFrame(frameIdx, totalFrames) {
    if (!this.devFrame) return;
    this.devFrame.textContent = totalFrames > 0
      ? `${frameIdx} / ${totalFrames - 1}`
      : '—';
  }

  showVehicleInfo(data) {
    if (!this.vehicleInfo) return;
    this.vehicleInfo.classList.remove('hidden');
    this._applyVehicleInfo(data);
  }

  updateVehicleInfo(data) {
    if (!this.vehicleInfo || this.vehicleInfo.classList.contains('hidden')) return;
    this._applyVehicleInfo(data);
  }

  hideVehicleInfo() {
    if (this.vehicleInfo) this.vehicleInfo.classList.add('hidden');
  }

  _applyVehicleInfo(data) {
    const fmt = (v) => (v != null ? Number(v).toFixed(3) : '—');
    this.viTrackId.textContent = data && data.track_id != null ? data.track_id : '—';
    this.viX.textContent = fmt(data && data.x_m_smoothed);
    this.viZ.textContent = fmt(data && data.z_m_smoothed);
    this.viVx.textContent = fmt(data && data.vx_m);
    this.viVz.textContent = fmt(data && data.vz_m);
  }

  syncPlayback(p) {
    this.playbackBar.classList.toggle('disabled', !p.hasTimeline);
    this.playBtn.textContent = p.isPlaying ? '⏸ Pause' : '▶ Play';
    this.playBtn.disabled = !p.hasTimeline;
    this.slider.disabled = !p.hasTimeline;
    if (!this._draggingSlider) {
      this.slider.value = String(Math.round(p.normalized * 1000));
    }
    this.timeLabel.textContent = p.hasTimeline
      ? `${fmt(p.currentTime)} / ${fmt(p.duration)}`
      : '--:-- / --:--';
    this._syncVideo(p);
  }
}

function fmt(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const cs = Math.floor((sec - Math.floor(sec)) * 100);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}.${String(cs).padStart(2, '0')}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}
