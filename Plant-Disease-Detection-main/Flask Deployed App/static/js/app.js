(function () {
  'use strict';

  const qs = (selector, root) => (root || document).querySelector(selector);
  const qsa = (selector, root) => Array.from((root || document).querySelectorAll(selector));

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatProbability(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
  }

  function formatDate(value) {
    try {
      return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value));
    } catch (_error) {
      return 'Recently';
    }
  }

  function showToast(message, isError) {
    const existing = qs('.toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = `toast${isError ? ' is-error' : ''}`;
    toast.setAttribute('role', isError ? 'alert' : 'status');
    toast.textContent = message;
    document.body.appendChild(toast);
    window.setTimeout(() => toast.remove(), 4200);
  }

  function setupSidebar() {
    const sidebar = qs('[data-sidebar]');
    const scrim = qs('.sidebar-scrim');
    if (!sidebar) return;
    const open = () => { sidebar.classList.add('is-open'); if (scrim) scrim.classList.add('is-visible'); };
    const close = () => { sidebar.classList.remove('is-open'); if (scrim) scrim.classList.remove('is-visible'); };
    qsa('[data-sidebar-open]').forEach((button) => button.addEventListener('click', open));
    qsa('[data-sidebar-close]').forEach((button) => button.addEventListener('click', close));
    qsa('.side-nav a').forEach((link) => link.addEventListener('click', close));
  }

  function saveHistory(result) {
    try {
      const stored = JSON.parse(localStorage.getItem('plantcare_prediction_history') || '[]');
      const item = { ...result, created_at: new Date().toISOString() };
      stored.unshift(item);
      localStorage.setItem('plantcare_prediction_history', JSON.stringify(stored.slice(0, 20)));
    } catch (_error) {
      // Local history is optional development storage; a prediction must still render.
    }
  }

  function loadHistory() {
    try {
      const value = JSON.parse(localStorage.getItem('plantcare_prediction_history') || '[]');
      return Array.isArray(value) ? value : [];
    } catch (_error) {
      return [];
    }
  }

  function renderTopPredictions(predictions) {
    return (predictions || []).map((item, index) => `
      <div class="top-prediction">
        <div><strong>${index + 1}. ${escapeHtml(item.display_name)}</strong><span>${escapeHtml(item.class_key)}</span></div>
        <b>${formatProbability(item.probability)}</b>
      </div>`).join('');
  }

  function renderPrediction(result, target) {
    const prediction = result.prediction || {};
    const knowledge = result.knowledge || {};
    const uncertain = Boolean(result.is_uncertain);
    const healthy = Boolean(knowledge.is_healthy);
    target.innerHTML = `
      <section class="result-shell" aria-live="polite">
        <div class="card result-image-card">
          <img src="${escapeHtml(result.image_url || '')}" alt="Analyzed plant image">
        </div>
        <div class="card result-info">
          <span class="result-state${uncertain ? ' is-uncertain' : ''}">${uncertain ? 'Review image' : healthy ? 'Healthy condition' : 'Disease screening result'}</span>
          <h2>${uncertain ? 'Prediction uncertain' : escapeHtml(prediction.plant || 'Plant')}</h2>
          <p class="result-condition">${uncertain ? 'Capture a clearer image to improve the screening result' : escapeHtml(prediction.condition || '')}</p>
          <div class="result-probability">
            <div class="probability-header"><span>Model probability</span><strong>${formatProbability(result.probability)}</strong></div>
            <div class="probability-track" aria-label="Model probability"><div class="probability-fill" style="width:${Math.max(0, Math.min(100, Number(result.probability || 0) * 100))}%"></div></div>
            <p class="result-note">Softmax score, not a calibrated guarantee. Screening threshold: ${formatProbability(result.uncertainty_threshold)}.</p>
          </div>
          ${uncertain ? '<div class="alert" role="status"><svg class="icon"><use href="#icon-alert"></use></svg><span>Unable to confidently identify the condition from this image. Keep the affected leaf centered, improve the light, and try again.</span></div>' : ''}
          <h3>Top predictions</h3>
          <div class="top-predictions">${renderTopPredictions(result.top_predictions)}</div>
          <h3>${healthy ? 'Verified source information' : 'Condition information'}</h3>
          <p class="result-copy">${escapeHtml(knowledge.description || 'No description is available in the verified project source.')}</p>
          <h3>${healthy ? 'Care information' : 'Management information'}</h3>
          <p class="result-copy">${escapeHtml(knowledge.management || 'No management information is available in the verified project source.')}</p>
          ${knowledge.supplement_name ? `<h3>Supplementary information</h3><p class="result-copy">${escapeHtml(knowledge.supplement_name)}</p>${knowledge.supplement_buy_url ? `<p><a class="text-link" href="${escapeHtml(knowledge.supplement_buy_url)}" target="_blank" rel="noopener noreferrer">Open verified source link <svg class="icon"><use href="#icon-external"></use></svg></a></p>` : ''}` : ''}
          <p class="result-note">Model: ${escapeHtml(result.model_version)} · Input: ${escapeHtml(result.input_size)}</p>
          <div class="result-actions"><button class="button button-primary" type="button" data-analyze-another>Analyze another image</button><a class="button button-secondary" href="/history">View history</a></div>
        </div>
      </section>
      <div class="knowledge-grid">
        <div class="card knowledge-card"><h3>What to do next</h3><p>Use this screening as an early signal. Confirm important treatment decisions with a local agronomist or plant pathologist.</p></div>
        <div class="card knowledge-card"><h3>Source note</h3><p>Disease information is displayed from the project's verified CSV sources. No additional treatment claims were added by the application.</p></div>
      </div>`;
    target.hidden = false;
    qsa('[data-analyze-another]', target).forEach((button) => button.addEventListener('click', () => {
      target.hidden = true;
      const form = qs('[data-plant-doctor]');
      if (form) form.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }));
  }

  function setupPlantDoctor() {
    const root = qs('[data-plant-doctor]');
    if (!root) return;
    const fileInput = qs('[data-image-input]', root);
    const uploadZone = qs('[data-upload-zone]', root);
    const uploadPanel = qs('[data-upload-panel]', root);
    const cameraPanel = qs('[data-camera-panel]', root);
    const cameraVideo = qs('[data-camera-video]', root);
    const cameraStatus = qs('[data-camera-status]', root);
    const capturePreview = qs('[data-capture-preview]', root);
    const previewImage = qs('[data-preview-image]', root);
    const analyzeButton = qs('[data-analyze]', root);
    const loading = qs('[data-analysis-loading]', root);
    const errorTarget = qs('[data-analysis-error]', root);
    const resultTarget = qs('[data-analysis-result]', root);
    let selectedFile = null;
    let stream = null;
    let currentMode = 'upload';

    const setError = (message) => {
      if (!errorTarget) return;
      errorTarget.hidden = !message;
      errorTarget.querySelector('[data-error-text]').textContent = message || '';
    };
    const setFile = (file) => {
      if (!file) return;
      if (!file.type || !file.type.startsWith('image/')) {
        setError('Please choose a JPEG, PNG, or WebP image.');
        return;
      }
      selectedFile = file;
      setError('');
      const reader = new FileReader();
      reader.onload = () => {
        previewImage.src = reader.result;
        capturePreview.hidden = false;
        uploadZone.hidden = true;
        analyzeButton.disabled = false;
      };
      reader.readAsDataURL(file);
    };
    const stopCamera = () => {
      if (stream) stream.getTracks().forEach((track) => track.stop());
      stream = null;
      if (cameraVideo) cameraVideo.srcObject = null;
    };
    const showUploadMode = () => {
      currentMode = 'upload';
      stopCamera();
      uploadPanel.hidden = false;
      cameraPanel.hidden = true;
      if (!selectedFile) capturePreview.hidden = true;
    };
    const showCameraMode = async () => {
      currentMode = 'camera';
      uploadPanel.hidden = true;
      cameraPanel.hidden = false;
      capturePreview.hidden = true;
      selectedFile = null;
      analyzeButton.disabled = true;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        cameraStatus.textContent = 'Camera access is not available in this browser. You can upload an image instead.';
        cameraPanel.dataset.cameraState = 'unavailable';
        return;
      }
      cameraStatus.textContent = 'Requesting camera permission…';
      cameraPanel.dataset.cameraState = 'loading';
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
        cameraVideo.srcObject = stream;
        await cameraVideo.play();
        cameraStatus.textContent = 'Live preview ready. Center one affected leaf before capturing.';
        cameraPanel.dataset.cameraState = 'live';
      } catch (error) {
        const denied = error && (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError');
        cameraStatus.textContent = denied ? 'Camera access is required to capture a plant image. Allow access or upload an image instead.' : 'No usable camera was found. Upload an image instead.';
        cameraPanel.dataset.cameraState = denied ? 'permission-denied' : 'error';
        stopCamera();
      }
    };

    qsa('[data-mode-tab]', root).forEach((tab) => tab.addEventListener('click', () => {
      qsa('[data-mode-tab]', root).forEach((item) => item.classList.toggle('is-active', item === tab));
      if (tab.dataset.mode === 'camera') showCameraMode(); else showUploadMode();
    }));
    qs('[data-browse]', root).addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', () => setFile(fileInput.files[0]));
    ['dragenter', 'dragover'].forEach((eventName) => uploadZone.addEventListener(eventName, (event) => { event.preventDefault(); uploadZone.classList.add('is-dragging'); }));
    ['dragleave', 'drop'].forEach((eventName) => uploadZone.addEventListener(eventName, (event) => { event.preventDefault(); uploadZone.classList.remove('is-dragging'); }));
    uploadZone.addEventListener('drop', (event) => setFile(event.dataTransfer.files[0]));
    qs('[data-capture]', root).addEventListener('click', () => {
      if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
        cameraStatus.textContent = 'The camera is still starting. Try again in a moment.';
        return;
      }
      const canvas = document.createElement('canvas');
      canvas.width = cameraVideo.videoWidth;
      canvas.height = cameraVideo.videoHeight;
      canvas.getContext('2d').drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (!blob) { setError('The camera frame could not be captured. Please try again.'); return; }
        setFile(new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' }));
        stopCamera();
        cameraPanel.hidden = true;
        cameraStatus.textContent = 'Capture ready for review.';
      }, 'image/jpeg', .92);
    });
    qsa('[data-retake]', root).forEach((button) => button.addEventListener('click', () => { selectedFile = null; capturePreview.hidden = true; if (currentMode === 'camera') showCameraMode(); else { uploadZone.hidden = false; analyzeButton.disabled = true; } }));
    analyzeButton.addEventListener('click', async () => {
      if (!selectedFile) { setError('Choose or capture an image first.'); return; }
      setError('');
      loading.classList.add('is-visible');
      analyzeButton.disabled = true;
      const formData = new FormData();
      formData.append('image', selectedFile, selectedFile.name || 'plant-image.jpg');
      const context = qs('[data-prediction-context]');
      if (context) {
        const field = qs('[data-field-select]', context);
        const crop = qs('[data-crop-select]', context);
        if (field && field.value) formData.append('field_id', field.value);
        if (crop && crop.value) formData.append('crop_cycle_id', crop.value);
      }
      try {
        const response = await fetch('/api/predict', { method: 'POST', body: formData, headers: { Accept: 'application/json' } });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'The image could not be analyzed.');
        saveHistory(payload);
        renderPrediction(payload, resultTarget);
        resultTarget.scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (error) {
        setError(error.message || 'The image could not be analyzed. Please try again.');
      } finally {
        loading.classList.remove('is-visible');
        analyzeButton.disabled = !selectedFile;
      }
    });
    window.addEventListener('beforeunload', stopCamera);
    const initialError = root.dataset.initialError;
    if (initialError) setError(initialError);
    const selected = sessionStorage.getItem('plantcare_selected_prediction');
    if (selected) {
      try { renderPrediction(JSON.parse(selected), resultTarget); sessionStorage.removeItem('plantcare_selected_prediction'); } catch (_error) { sessionStorage.removeItem('plantcare_selected_prediction'); }
    }
  }

  function renderMongoHistory(items, root) {
    const list = qs('[data-history-list]', root);
    const empty = qs('[data-history-empty]', root);
    if (!items.length) { list.innerHTML = ''; empty.hidden = false; return; }
    empty.hidden = true;
    list.innerHTML = items.map((item) => `
      <article class="history-item">
        <img class="history-thumb" src="${escapeHtml(item.image_reference || '')}" alt="">
        <span class="history-copy"><strong>${escapeHtml([item.predicted_plant, item.predicted_condition].filter(Boolean).join(' · ') || item.predicted_class || 'Prediction')}</strong><span>${escapeHtml(formatDate(item.created_at))} · ${escapeHtml(item.model_version || '')}</span></span>
        <b class="history-score">${formatProbability(item.model_probability)}</b>
      </article>`).join('');
  }

  async function setupHistory() {
    const root = qs('[data-history]');
    if (!root) return;
    const list = qs('[data-history-list]', root);
    const empty = qs('[data-history-empty]', root);
    try {
      const response = await fetch('/api/predictions', { headers: { Accept: 'application/json' } });
      if (response.ok) {
        const payload = await response.json();
        if (payload.status === 'ok' && Array.isArray(payload.data)) {
          renderMongoHistory(payload.data, root);
          return;
        }
      }
    } catch (_error) {
      // Fall back to the existing browser history when MongoDB is unavailable.
    }
    const history = loadHistory();
    if (!history.length) { empty.hidden = false; return; }
    empty.hidden = true;
    list.innerHTML = history.map((item, index) => `
      <button class="history-item" type="button" data-history-index="${index}">
        <img class="history-thumb" src="${escapeHtml(item.image_url || '')}" alt="">
        <span class="history-copy"><strong>${escapeHtml(item.prediction && item.prediction.display_name)}</strong><span>${escapeHtml(formatDate(item.created_at))} · ${escapeHtml(item.model_version)}</span></span>
        <b class="history-score">${formatProbability(item.probability)}</b>
        <svg class="icon"><use href="#icon-arrow-right"></use></svg>
      </button>`).join('');
    qsa('[data-history-index]', root).forEach((button) => button.addEventListener('click', () => {
      sessionStorage.setItem('plantcare_selected_prediction', JSON.stringify(history[Number(button.dataset.historyIndex)]));
      window.location.href = '/plant-doctor';
    }));
  }

  function setupDashboardHistory() {
    const root = qs('[data-dashboard-history]');
    if (!root) return;
    const list = qs('[data-dashboard-history-list]', root);
    const empty = qs('[data-dashboard-history-empty]', root);
    const history = loadHistory().slice(0, 3);
    if (!history.length) return;
    empty.hidden = true;
    list.innerHTML = history.map((item) => `
      <a class="history-item" href="/history">
        <img class="history-thumb" src="${escapeHtml(item.image_url || '')}" alt="">
        <span class="history-copy"><strong>${escapeHtml(item.prediction && item.prediction.display_name)}</strong><span>${escapeHtml(formatDate(item.created_at))}</span></span>
        <b class="history-score">${formatProbability(item.probability)}</b>
      </a>`).join('');
  }

  function setupLibrary() {
    const root = qs('[data-library]');
    if (!root) return;
    const search = qs('[data-library-search]', root);
    const filter = qs('[data-library-filter]', root);
    const items = qsa('[data-library-item]', root);
    const empty = qs('[data-library-empty]', root);
    const conditions = Array.from(new Set(items.map((item) => item.dataset.condition).filter(Boolean))).sort();
    conditions.forEach((condition) => {
      if (condition === 'healthy') return;
      const option = document.createElement('option');
      option.value = condition;
      option.textContent = condition.replace(/\b\w/g, (letter) => letter.toUpperCase());
      filter.appendChild(option);
    });
    const update = () => {
      const query = (search.value || '').toLowerCase().trim();
      const type = filter.value;
      let visible = 0;
      items.forEach((item) => {
        const matchesQuery = !query || item.textContent.toLowerCase().includes(query);
        const matchesType = (type === 'all' || type === 'healthy') ? (type === 'all' || item.dataset.type === type) : item.dataset.condition === type;
        item.hidden = !(matchesQuery && matchesType);
        if (!item.hidden) visible += 1;
      });
      if (empty) empty.hidden = visible > 0;
    };
    search.addEventListener('input', update);
    filter.addEventListener('change', update);
  }

  function setupUiOnlyForms() {
    qsa('[data-ui-only-form]').forEach((form) => form.addEventListener('submit', (event) => {
      event.preventDefault();
      showToast('This form is ready for the next data and database phase. Nothing was saved.', false);
    }));
  }

  setupSidebar();
  setupPlantDoctor();
  setupHistory();
  setupDashboardHistory();
  setupLibrary();
  setupUiOnlyForms();
}());
