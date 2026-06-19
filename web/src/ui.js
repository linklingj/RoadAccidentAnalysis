// DOM glue: scene loading controls, the playback bar, the info panel and view
// presets. main.js owns the three.js side and passes in handlers; this module
// only touches the DOM.

export class ViewerUI {
  constructor(handlers) {
    this.h = handlers; // { onFile, onLoadSample, onMedia, onView, onTogglePlay, onSeek, onSpeedChange }
    this._draggingSlider = false;
    this.serverAvailable = false;

    this.$ = (id) => document.getElementById(id);
    this.statusEl = this.$('status');
    this.infoEl = this.$('info');
    this.playBtn = this.$('playBtn');
    this.slider = this.$('seek');
    this.timeLabel = this.$('timeLabel');
    this.speedSel = this.$('speed');
    this.fileInput = this.$('fileInput');
    this.mediaInput = this.$('mediaInput');
    this.analyzeBtn = this.$('analyzeBtn');
    this.camH = this.$('camH');
    this.ppm = this.$('ppm');
    this.playbackBar = this.$('playbackBar');

    this._wire();
  }

  _wire() {
    this.fileInput.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) this.h.onFile(file);
      this.fileInput.value = '';
    });
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
    this.$('sampleBtn').addEventListener('click', () => this.h.onLoadSample());

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

  setInfo(summary) {
    const cam = summary.camera || {};
    const rows = [
      ['Mode', summary.mode === 'video' ? `video (${summary.frames} frames @ ${summary.fps.toFixed(2)} fps)` : 'image'],
      ['Roads', summary.roads],
      ['Crosswalks', summary.crosswalks],
      ['Objects', summary.objects],
      ['Tracks', summary.mode === 'video' ? summary.tracks : summary.trajectories],
      ['Camera height', cam.height_m != null ? `${cam.height_m} m` : '—'],
      ['Pitch / Roll', cam.pitch_deg != null ? `${cam.pitch_deg.toFixed(1)}° / ${cam.roll_deg.toFixed(1)}°` : '—'],
      ['vFOV', cam.vfov_deg != null ? `${cam.vfov_deg.toFixed(1)}°` : '—'],
    ];
    this.infoEl.innerHTML = rows
      .map(([k, v]) => `<div class="row"><span>${k}</span><b>${v}</b></div>`)
      .join('');
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
