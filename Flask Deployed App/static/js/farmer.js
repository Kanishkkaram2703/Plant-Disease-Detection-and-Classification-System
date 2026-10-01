(function () {
  'use strict';

  const qs = (selector, root) => (root || document).querySelector(selector);
  const qsa = (selector, root) => Array.from((root || document).querySelectorAll(selector));
  const esc = (value) => String(value == null ? '' : value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  const dateText = (value) => {
    if (!value) return 'Not set';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? 'Not set' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium' }).format(parsed);
  };
  const dateTimeText = (value) => {
    if (!value) return 'Not set';
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? 'Not set' : new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(parsed);
  };

  function notice(message, error) {
    const old = qs('[data-farmer-toast]');
    if (old) old.remove();
    const node = document.createElement('div');
    node.className = `toast${error ? ' is-error' : ''}`;
    node.dataset.farmerToast = 'true';
    node.setAttribute('role', error ? 'alert' : 'status');
    node.textContent = message;
    document.body.appendChild(node);
    window.setTimeout(() => node.remove(), 4200);
  }

  async function api(path, options) {
    const request = options || {};
    const response = await fetch(path, {
      ...request,
      headers: { Accept: 'application/json', 'Content-Type': 'application/json', ...(request.headers || {}) }
    });
    const payload = await response.json().catch(() => ({ status: 'error', error: 'The server returned an unreadable response.' }));
    if (!response.ok || payload.status === 'error') {
      const error = new Error(payload.error || 'The request could not be completed.');
      error.code = payload.code;
      error.status = response.status;
      throw error;
    }
    return payload.data;
  }

  function formObject(form) {
    const data = {};
    new FormData(form).forEach((value, key) => {
      if (value === '') return;
      if (key.startsWith('location_')) {
        data.location = data.location || {};
        data.location[key.replace('location_', '')] = value;
      } else {
        data[key] = value;
      }
    });
    return data;
  }

  function setSubmitting(form, submitting) {
    form.dataset.submitting = submitting ? 'true' : 'false';
    qsa('button[type="submit"]', form).forEach((button) => {
      button.disabled = submitting;
      if (submitting) button.dataset.originalText = button.textContent;
      button.textContent = submitting ? 'Saving…' : (button.dataset.originalText || button.textContent);
    });
  }

  function showProfileRequired(target) {
    if (!target) return;
    target.hidden = false;
    target.innerHTML = '<div class="empty-state"><div class="empty-icon"><svg class="icon"><use href="#icon-settings"></use></svg></div><h3>Create your farmer profile first</h3><p>Fields and farm records are scoped to a signed-in profile.</p><a class="button button-primary" href="/profile">Create profile</a></div>';
  }

  function initials(name) {
    return String(name || 'Farmer').split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0].toUpperCase()).join('') || 'F';
  }

  function setupAccountMenu() {
    const menu = qs('[data-account-menu]');
    if (!menu) return;
    const toggle = qs('[data-account-toggle]', menu);
    const popover = qs('[data-account-popover]', menu);
    if (toggle && popover) {
      toggle.addEventListener('click', (event) => {
        event.stopPropagation();
        const open = popover.hidden;
        popover.hidden = !open;
        toggle.setAttribute('aria-expanded', String(open));
      });
      document.addEventListener('click', () => {
        popover.hidden = true;
        toggle.setAttribute('aria-expanded', 'false');
      });
    }
    qsa('[data-logout]').forEach((button) => button.addEventListener('click', async () => {
      try { await api('/logout', { method: 'POST' }); window.location.href = '/'; }
      catch (error) { notice(error.message, true); }
    }));
  }

  function fillForm(form, profile) {
    ['name', 'email', 'phone', 'farm_name', 'location'].forEach((key) => {
      const input = qs(`[name="${key}"]`, form);
      if (input) input.value = profile && profile[key] ? profile[key] : '';
    });
  }

  function renderProfile(profile, summary, form, loggedOut) {
    if (!profile) {
      summary.hidden = true;
      form.hidden = true;
      loggedOut.hidden = false;
      return;
    }
    loggedOut.hidden = true;
    summary.hidden = false;
    form.hidden = true;
    qs('[data-profile-initials]', summary).textContent = initials(profile.name);
    qs('[data-profile-name]', summary).textContent = profile.name || 'Farmer';
    qs('[data-profile-farm]', summary).textContent = profile.farm_name || 'Farm name not set';
    qs('[data-profile-email]', summary).textContent = profile.email || 'Not set';
    qs('[data-profile-phone]', summary).textContent = profile.phone || 'Not set';
    qs('[data-profile-location]', summary).textContent = profile.location || 'Not set';
  }

  async function setupProfile() {
    const form = qs('[data-profile-form]');
    const summary = qs('[data-profile-summary]');
    const loggedOut = qs('[data-profile-logged-out]');
    if (!form || !summary || !loggedOut) return;
    const banner = qs('[data-profile-message]');
    const message = qs('[data-profile-message-text]');
    let existing = null;
    const showMessage = (text, error) => { message.textContent = text; banner.hidden = !text; banner.classList.toggle('is-error', Boolean(error)); };
    try {
      existing = await api('/api/profile');
      if (existing) fillForm(form, existing);
      renderProfile(existing, summary, form, loggedOut);
    } catch (error) {
      showMessage(error.message, true);
      renderProfile(null, summary, form, loggedOut);
    }
    const edit = qs('[data-edit-profile]', summary);
    if (edit) edit.addEventListener('click', () => { fillForm(form, existing); summary.hidden = true; form.hidden = false; });
    const create = qs('[data-create-profile]', loggedOut);
    if (create) create.addEventListener('click', () => { loggedOut.hidden = true; form.hidden = false; });
    const cancel = qs('[data-cancel-profile]', form);
    if (cancel) cancel.addEventListener('click', () => renderProfile(existing, summary, form, loggedOut));
    if (form.dataset.bound) return;
    form.dataset.bound = 'true';
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!form.reportValidity() || form.dataset.submitting === 'true') return;
      setSubmitting(form, true);
      try {
        existing = await api('/api/profile', { method: existing ? 'PATCH' : 'POST', body: JSON.stringify(formObject(form)) });
        showMessage('Profile saved.', false);
        renderProfile(existing, summary, form, loggedOut);
        notice('Profile saved.');
      } catch (error) { showMessage(error.message, true); }
      finally { setSubmitting(form, false); }
    });
  }

  function fieldCard(field) {
    const location = field.location || {};
    const locationText = [location.village, location.district, location.state].filter(Boolean).join(', ') || 'Location not set';
    return `<article class="card catalog-card"><span class="catalog-tag${field.status === 'archived' ? ' is-disease' : ''}">${esc(field.status || 'active')}</span><h3>${esc(field.field_name)}</h3><div class="catalog-plant">${esc(field.area)} ${esc(field.area_unit)}</div><p>${esc(locationText)}<br>${esc(field.soil_type || 'Soil not set')} · ${esc(field.irrigation_method || 'Irrigation not set')}</p><div class="card-footer"><a class="text-link" href="/fields/${esc(field._id)}">Open field</a><a class="button button-secondary" href="/fields/${esc(field._id)}/edit">Edit</a><button class="button button-danger" type="button" data-delete-field="${esc(field._id)}" data-field-name="${esc(field.field_name)}">Delete</button></div></article>`;
  }

  async function setupFields() {
    const root = qs('[data-fields-page]');
    if (!root) return;
    const grid = qs('[data-fields-list]', root);
    const render = async () => {
      try {
        const fields = await api('/api/fields');
        if (!fields.length) { grid.innerHTML = '<div class="empty-state"><div class="empty-icon"><svg class="icon"><use href="#icon-map"></use></svg></div><h3>No fields added yet.</h3><p>Add your first field to start managing crops and field activity.</p><a class="button button-primary" href="/fields/add">Add field</a></div>'; return; }
        grid.innerHTML = fields.map(fieldCard).join('');
        qsa('[data-delete-field]', grid).forEach((button) => button.addEventListener('click', async () => {
          const fieldName = button.dataset.fieldName || 'this field';
          if (!window.confirm(`Delete ${fieldName}? Related records are preserved by archiving the field when dependencies exist.`)) return;
          button.disabled = true;
          try {
            const result = await api(`/api/fields/${button.dataset.deleteField}`, { method: 'DELETE' });
            notice(result.archived ? 'Field archived so related records remain safe.' : 'Field deleted.');
            await render();
          } catch (error) { button.disabled = false; notice(error.message, true); }
        }));
      } catch (error) { if (error.code === 'profile_required') showProfileRequired(grid); else grid.innerHTML = `<div class="alert"><svg class="icon"><use href="#icon-alert"></use></svg><span>${esc(error.message)}</span></div>`; }
    };
    await render();
  }

  function setupFieldControls(form) {
    const soil = qs('[name="soil_type"]', form);
    const soilHelp = qs('[data-soil-help]', form);
    if (soil && soilHelp && !soil.dataset.bound) {
      const updateSoil = () => { const selected = soil.options[soil.selectedIndex]; soilHelp.textContent = selected && selected.dataset.description ? selected.dataset.description : 'Select the closest match. You can choose Other if unsure.'; };
      soil.addEventListener('change', updateSoil);
      soil.dataset.bound = 'true';
      updateSoil();
    }
    const method = qs('[name="irrigation_method"]', form);
    const other = qs('[data-other-irrigation]', form);
    if (method && other && !method.dataset.bound) {
      const updateOther = () => { other.hidden = method.value !== 'Other'; };
      method.addEventListener('change', updateOther);
      method.dataset.bound = 'true';
      updateOther();
    }
  }

  async function setupFieldForm() {
    const form = qs('[data-field-form]');
    if (!form) return;
    const fieldId = form.dataset.fieldId;
    setupFieldControls(form);
    if (fieldId) {
      try {
        const field = await api(`/api/fields/${fieldId}`);
        Object.entries(field).forEach(([key, value]) => { const input = qs(`[name="${key}"]`, form); if (input && value != null && typeof value !== 'object') input.value = value; });
        Object.entries(field.location || {}).forEach(([key, value]) => { const input = qs(`[name="location_${key}"]`, form); if (input) input.value = value; });
        setupFieldControls(form);
      } catch (error) { notice(error.message, true); }
    }
    if (form.dataset.bound) return;
    form.dataset.bound = 'true';
    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      if (!form.reportValidity() || form.dataset.submitting === 'true') return;
      setSubmitting(form, true);
      try { await api(fieldId ? `/api/fields/${fieldId}` : '/api/fields', { method: fieldId ? 'PATCH' : 'POST', body: JSON.stringify(formObject(form)) }); window.location.href = '/fields'; }
      catch (error) { notice(error.message, true); setSubmitting(form, false); }
    });
  }

  function optionMarkup(items, label) {
    return items.map((item) => `<option value="${esc(item._id)}">${esc(label(item))}</option>`).join('');
  }

  function populateCropSelect(select, crops) {
    const form = select.closest('form');
    const fieldSelect = form ? qs('[data-field-select]', form) : null;
    const hiddenField = form ? qs('input[name="field_id"]', form) : null;
    const fieldId = fieldSelect && fieldSelect.value ? fieldSelect.value : hiddenField && hiddenField.value ? hiddenField.value : '';
    const selected = select.value;
    const filtered = fieldId ? crops.filter((crop) => crop.field_id === fieldId) : crops;
    select.innerHTML = '<option value="">Optional crop cycle</option>' + optionMarkup(filtered, (item) => `${item.crop_name}${item.variety ? ` · ${item.variety}` : ''}`);
    if (filtered.some((item) => item._id === selected)) select.value = selected;
  }

  async function loadOptions(root) {
    try {
      const options = await api('/api/field-options');
      qsa('[data-field-select]', root || document).forEach((select) => {
        const selected = select.value;
        select.innerHTML = '<option value="">Select field</option>' + optionMarkup(options.fields, (item) => item.field_name);
        if (selected) select.value = selected;
        if (!select.dataset.bound) {
          select.dataset.bound = 'true';
          select.addEventListener('change', () => qsa('[data-crop-select]', select.closest('form') || root || document).forEach((cropSelect) => populateCropSelect(cropSelect, options.crops)));
        }
      });
      qsa('[data-crop-select]', root || document).forEach((select) => {
        populateCropSelect(select, options.crops);
        if (!select.dataset.bound) select.dataset.bound = 'true';
      });
      return options;
    } catch (error) { if (error.code !== 'profile_required') notice(error.message, true); return null; }
  }

  function simpleList(items, emptyText, renderer) {
    return items.length ? items.map(renderer).join('') : `<div class="empty-state"><div class="empty-icon"><svg class="icon"><use href="#icon-info"></use></svg></div><h3>${esc(emptyText)}</h3><p>No records are available yet.</p></div>`;
  }

  async function setupFieldDetail() {
    const root = qs('[data-field-detail]');
    if (!root) return;
    const fieldId = root.dataset.fieldId;
    try {
      const data = await api(`/api/fields/${fieldId}/overview`);
      const field = data.field;
      const location = field.location || {};
      qs('[data-field-name]', root).textContent = field.field_name;
      qs('[data-field-meta]', root).textContent = `${field.area} ${field.area_unit} · ${[location.village, location.district, location.state].filter(Boolean).join(', ') || 'Location not set'}`;
      qsa('[data-field-value]', root).forEach((node) => { const value = field[node.dataset.fieldValue]; node.textContent = value == null || value === '' ? 'Not set' : value; });
      qs('[data-field-location]', root).textContent = [location.village, location.district, location.state, location.country].filter(Boolean).join(', ') || 'Not set';
      qs('[data-crop-list]', root).innerHTML = simpleList(data.crops, 'No crop cycles for this field', (crop) => `<article class="card detail-card"><h3>${esc(crop.crop_name)}</h3><p>${esc(crop.variety || 'Variety not set')} · ${esc(crop.current_growth_stage || 'Growth stage not set')}<br>Planted ${esc(dateText(crop.planting_date))} · ${esc(crop.status)}</p></article>`);
      qs('[data-task-list]', root).innerHTML = simpleList(data.tasks.filter((task) => task.field_id === fieldId), 'No tasks for this field', (task) => `<article class="card detail-card"><h3>${esc(task.title)}</h3><p>${task.generated ? 'Recommended' : 'My task'} · ${esc(task.status)} · ${esc(task.priority)}<br>Due ${esc(dateText(task.due_date))}</p></article>`);
      qs('[data-irrigation-list]', root).innerHTML = simpleList(data.irrigation, 'No irrigation records', (item) => `<article class="card detail-card"><h3>${esc(item.irrigation_method || 'Irrigation record')}</h3><p>${esc(dateTimeText(item.recorded_at))}${item.duration_minutes ? ` · ${esc(item.duration_minutes)} minutes` : ''}</p></article>`);
      qs('[data-health-list]', root).innerHTML = simpleList(data.health, 'No health observations', (item) => `<article class="card detail-card"><h3>${esc(item.health_status)}</h3><p>${esc(dateText(item.observation_date))}<br>${esc(item.symptoms_notes || 'No notes')}</p></article>`);
      qs('[data-prediction-list]', root).innerHTML = simpleList(data.predictions, 'No linked predictions', (item) => `<article class="card detail-card"><h3>${esc(item.predicted_condition || item.predicted_class)}</h3><p>${esc(item.predicted_plant || '')} · ${(Number(item.model_probability || 0) * 100).toFixed(1)}%<br>${esc(dateText(item.created_at))}</p></article>`);
      qsa('[data-crop-select]', root).forEach((select) => { select.innerHTML = '<option value="">Optional crop cycle</option>' + optionMarkup(data.crops, (item) => `${item.crop_name}${item.variety ? ` · ${item.variety}` : ''}`); });
      qsa('[data-detail-form]', root).forEach((form) => {
        if (form.dataset.bound) return;
        form.dataset.bound = 'true';
        form.addEventListener('submit', async (event) => {
          event.preventDefault();
          if (!form.reportValidity() || form.dataset.submitting === 'true') return;
          setSubmitting(form, true);
          try { await api(form.dataset.endpoint, { method: 'POST', body: JSON.stringify(formObject(form)) }); notice('Record saved.'); form.reset(); await setupFieldDetail(); }
          catch (error) { notice(error.message, true); }
          finally { setSubmitting(form, false); }
        });
      });
    } catch (error) { if (error.code === 'profile_required') showProfileRequired(root); else notice(error.message, true); }
  }

  async function setupCropPage() {
    const root = qs('[data-crops-page]');
    if (!root) return;
    const list = qs('[data-crop-list]', root);
    const options = await loadOptions(root);
    const render = async () => {
      try {
        const crops = await api('/api/crops');
        const fields = (options && options.fields) || [];
        const fieldNames = Object.fromEntries(fields.map((field) => [field._id, field.field_name]));
        list.innerHTML = simpleList(crops, 'No crop cycles yet', (crop) => `<article class="card detail-card"><h3>${esc(crop.crop_name)}</h3><p>${esc(fieldNames[crop.field_id] || 'Field')} · ${esc(crop.variety || 'Variety not set')} · ${esc(crop.current_growth_stage || 'Growth stage not set')}<br>Planted ${esc(dateText(crop.planting_date))} · ${esc(crop.status)}</p></article>`);
      } catch (error) { if (error.code === 'profile_required') showProfileRequired(list); else notice(error.message, true); }
    };
    await render();
    const form = qs('[data-crop-form]', root);
    if (form && !form.dataset.bound) {
      form.dataset.bound = 'true';
      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!form.reportValidity() || form.dataset.submitting === 'true') return;
        setSubmitting(form, true);
        try { await api('/api/crops', { method: 'POST', body: JSON.stringify(formObject(form)) }); notice('Crop cycle saved.'); form.reset(); await loadOptions(root); await render(); }
        catch (error) { notice(error.message, true); }
        finally { setSubmitting(form, false); }
      });
    }
  }

  function taskCard(task, fieldNames, cropNames) {
    const link = task.field_id ? ` · ${esc(fieldNames[task.field_id] || 'Field')}` : '';
    const crop = task.crop_cycle_id ? ` · ${esc(cropNames[task.crop_cycle_id] || 'Crop')}` : '';
    const reason = task.generated && task.generation_reason ? `<br><span class="form-help">Reason: ${esc(task.generation_reason)}</span>` : '';
    return `<article class="card detail-card"><span class="catalog-tag${task.generated ? '' : ' is-disease'}">${task.generated ? 'Recommended' : 'My task'}</span><h3>${esc(task.title)}</h3><p>${esc(task.status)} · ${esc(task.priority)}${link}${crop}<br>Due ${esc(dateText(task.due_date))}${reason}</p>${task.status !== 'completed' && task.status !== 'cancelled' ? `<button class="button button-secondary" type="button" data-complete-task="${esc(task._id)}">Mark complete</button>` : ''}</article>`;
  }

  async function setupTasks() {
    const root = qs('[data-tasks-page]');
    if (!root) return;
    const list = qs('[data-task-list]', root);
    const generatedList = qs('[data-generated-task-list]', root);
    const options = await loadOptions(root);
    const render = async () => {
      try {
        const tasks = await api('/api/tasks');
        const fields = (options && options.fields) || [];
        const crops = (options && options.crops) || [];
        const fieldNames = Object.fromEntries(fields.map((field) => [field._id, field.field_name]));
        const cropNames = Object.fromEntries(crops.map((crop) => [crop._id, crop.crop_name]));
        const generated = tasks.filter((task) => task.generated);
        const manual = tasks.filter((task) => !task.generated);
        generatedList.innerHTML = simpleList(generated, 'No recommended tasks yet', (task) => taskCard(task, fieldNames, cropNames));
        list.innerHTML = simpleList(manual, 'No tasks yet', (task) => taskCard(task, fieldNames, cropNames));
        qsa('[data-complete-task]', root).forEach((button) => button.addEventListener('click', async () => { button.disabled = true; try { await api(`/api/tasks/${button.dataset.completeTask}/complete`, { method: 'POST' }); await render(); } catch (error) { button.disabled = false; notice(error.message, true); } }));
      } catch (error) { if (error.code === 'profile_required') showProfileRequired(list); else notice(error.message, true); }
    };
    await render();
    const form = qs('[data-task-form]', root);
    if (form && !form.dataset.bound) {
      form.dataset.bound = 'true';
      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!form.reportValidity() || form.dataset.submitting === 'true') return;
        setSubmitting(form, true);
        try { await api('/api/tasks', { method: 'POST', body: JSON.stringify(formObject(form)) }); notice('Task saved.'); form.reset(); await render(); }
        catch (error) { notice(error.message, true); }
        finally { setSubmitting(form, false); }
      });
    }
  }

  async function setupIrrigation() {
    const root = qs('[data-irrigation-page]');
    if (!root) return;
    const list = qs('[data-irrigation-list]', root);
    await loadOptions(root);
    const render = async () => {
      try {
        const rows = await api('/api/irrigation');
        list.innerHTML = simpleList(rows, 'No irrigation records yet', (item) => `<article class="card detail-card"><h3>${esc(item.irrigation_method || 'Irrigation record')}</h3><p>${esc(dateTimeText(item.recorded_at))}${item.duration_minutes ? ` · ${esc(item.duration_minutes)} minutes` : ''}<br>${esc(item.notes || 'No notes')}</p></article>`);
      } catch (error) { if (error.code === 'profile_required') showProfileRequired(list); else notice(error.message, true); }
    };
    await render();
    const form = qs('[data-irrigation-form]', root);
    if (form && !form.dataset.bound) {
      form.dataset.bound = 'true';
      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!form.reportValidity() || form.dataset.submitting === 'true') return;
        setSubmitting(form, true);
        try { await api('/api/irrigation', { method: 'POST', body: JSON.stringify(formObject(form)) }); notice('Irrigation record saved.'); form.reset(); await loadOptions(root); await render(); }
        catch (error) { notice(error.message, true); }
        finally { setSubmitting(form, false); }
      });
    }
  }

  async function setupNotifications() {
    const root = qs('[data-notifications-page]');
    if (!root) return;
    const list = qs('[data-notification-list]', root);
    const render = async () => {
      try {
        const rows = await api('/api/notifications');
        list.innerHTML = simpleList(rows, 'You are all caught up', (item) => `<article class="card detail-card"><span class="catalog-tag${item.read ? '' : ' is-disease'}">${item.read ? 'Read' : 'Unread'}</span><h3>${esc(item.title)}</h3><p>${esc(item.message)}<br>${esc(dateTimeText(item.created_at))}</p>${item.read ? '' : `<button class="button button-secondary" type="button" data-read-notification="${esc(item._id)}">Mark read</button>`}</article>`);
        qsa('[data-read-notification]', root).forEach((button) => button.addEventListener('click', async () => { button.disabled = true; try { await api(`/api/notifications/${button.dataset.readNotification}/read`, { method: 'POST' }); await render(); } catch (error) { button.disabled = false; notice(error.message, true); } }));
      } catch (error) { if (error.code === 'profile_required') showProfileRequired(list); else notice(error.message, true); }
    };
    const markAll = qs('[data-mark-all-read]', root);
    if (markAll && !markAll.dataset.bound) { markAll.dataset.bound = 'true'; markAll.addEventListener('click', async () => { markAll.disabled = true; try { await api('/api/notifications/read-all', { method: 'POST' }); await render(); } catch (error) { notice(error.message, true); } finally { markAll.disabled = false; } }); }
    await render();
  }

  async function setupDashboard() {
    const root = qs('[data-dashboard-page]');
    if (!root) return;
    try {
      const dashboard = await api('/api/dashboard');
      const stats = dashboard.stats || {};
      qsa('[data-stat]', root).forEach((node) => { node.textContent = stats[node.dataset.stat] == null ? '0' : stats[node.dataset.stat]; });
      const fields = qs('[data-dashboard-fields]', root);
      if (fields) fields.innerHTML = simpleList(dashboard.fields || [], 'No fields added yet.', fieldCard);
      const tasks = qs('[data-dashboard-tasks]', root);
      if (tasks) tasks.innerHTML = simpleList(dashboard.upcoming_tasks || [], 'No pending tasks.', (task) => taskCard(task, {}, {}));
    } catch (error) { if (error.code === 'profile_required') showProfileRequired(qs('[data-dashboard-database-state]', root)); else notice(error.message, true); }
  }

  async function setupSettings() {
    const root = qs('[data-settings-page]');
    if (!root) return;
    const form = qs('[data-settings-form]', root);
    const checkbox = qs('[data-notifications-enabled]', root);
    const message = qs('[data-settings-message-text]', root);
    const banner = qs('[data-settings-message]', root);
    const applyTheme = (theme) => {
      const actual = theme === 'system' ? (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light') : theme;
      document.documentElement.dataset.theme = actual;
    };
    try {
      const settings = await api('/api/settings');
      qs('[name="theme"]', form).value = settings.theme;
      qs('[name="unit_preference"]', form).value = settings.unit_preference;
      checkbox.checked = settings.notifications_enabled;
      applyTheme(settings.theme);
    } catch (error) { showProfileRequired(root); return; }
    if (!form.dataset.bound) {
      form.dataset.bound = 'true';
      form.addEventListener('change', (event) => { if (event.target.name === 'theme') applyTheme(event.target.value); });
      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (form.dataset.submitting === 'true') return;
        setSubmitting(form, true);
        try { const result = await api('/api/settings', { method: 'PATCH', body: JSON.stringify({ theme: qs('[name="theme"]', form).value, unit_preference: qs('[name="unit_preference"]', form).value, notifications_enabled: checkbox.checked }) }); applyTheme(result.theme); message.textContent = 'Preferences saved.'; banner.hidden = false; banner.classList.remove('is-error'); notice('Preferences saved.'); }
        catch (error) { message.textContent = error.message; banner.hidden = false; banner.classList.add('is-error'); }
        finally { setSubmitting(form, false); }
      });
    }
  }

  async function setupPredictionContext() {
    const root = qs('[data-prediction-context]');
    if (root) await loadOptions(root);
  }

  setupAccountMenu();
  setupProfile();
  setupFields();
  setupFieldForm();
  setupCropPage();
  setupFieldDetail();
  setupTasks();
  setupIrrigation();
  setupNotifications();
  setupDashboard();
  setupSettings();
  setupPredictionContext();
}());
