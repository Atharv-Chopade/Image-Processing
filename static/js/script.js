/* PixelLab – front-end logic.
   The operation list and every parameter control are generated from GET /api/operations,
   so the browser and the Flask back end always agree on names, ranges and defaults. */
(() => {
  "use strict";

  /* ------------------------------------------------------------------ helpers */
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  const state = {
    categories: [],
    ops: [],
    opMap: {},
    limits: {},
    currentOp: null,
    controls: {},          // parameter name -> { spec, wrap, get, set }
    info: null,            // last state payload from the server (null = no image)
    originalKey: null,     // which uploaded image the left preview currently shows
    busy: false,
    activeTab: "history",
    dirty: { histogram: true, channels: true },
  };

  /* -------------------------------------------------------------------- theme */
  function storeGet(key) { try { return localStorage.getItem(key); } catch (_) { return null; } }
  function storeSet(key, value) { try { localStorage.setItem(key, value); } catch (_) { /* ignore */ } }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-bs-theme", theme);
    storeSet("pixellab-theme", theme);
    const icon = $("#themeToggle i");
    icon.className = theme === "dark" ? "bi bi-sun-fill" : "bi bi-moon-stars-fill";
  }
  setTheme(storeGet("pixellab-theme") ||
           (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("#themeToggle").addEventListener("click", () => {
    setTheme(document.documentElement.getAttribute("data-bs-theme") === "dark" ? "light" : "dark");
  });

  /* ------------------------------------------------------------ network + UI */
  function toast(message, type = "danger") {
    const node = el("div", `toast align-items-center text-bg-${type} border-0`);
    node.setAttribute("role", "alert");
    node.setAttribute("aria-live", "assertive");
    const row = el("div", "d-flex");
    row.append(el("div", "toast-body", message));
    const close = el("button", `btn-close ${type === "warning" ? "" : "btn-close-white"} me-2 m-auto`);
    close.type = "button";
    close.setAttribute("data-bs-dismiss", "toast");
    close.setAttribute("aria-label", "Close");
    row.append(close);
    node.append(row);
    $("#toastContainer").append(node);
    node.addEventListener("hidden.bs.toast", () => node.remove());
    new bootstrap.Toast(node, { delay: type === "danger" ? 7000 : 3500 }).show();
  }

  async function api(url, options = {}) {
    const response = await fetch(url, { credentials: "same-origin", ...options });
    let data = null;
    try { data = await response.json(); } catch (_) { /* the body was not JSON */ }
    if (!response.ok || (data && data.ok === false)) {
      throw new Error((data && data.error) || `The request failed (HTTP ${response.status}).`);
    }
    return data;
  }

  const postJson = (url, body = {}) => api(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  function setImage(img, url) {
    return new Promise((resolve) => {
      img.onload = () => resolve(true);
      img.onerror = () => resolve(false);
      img.src = url;
    });
  }

  function setBusyUI(busy) {
    $("#processedBusy").classList.toggle("d-none", !busy);
    $("#originalBusy").classList.toggle("d-none", !busy);
    refreshButtons();
  }

  async function runBusy(task) {
    if (state.busy) return;
    state.busy = true;
    setBusyUI(true);
    try {
      await task();
    } catch (error) {
      toast(error.message || "Something went wrong.");
    } finally {
      state.busy = false;
      setBusyUI(false);
    }
  }

  function refreshButtons() {
    const loaded = !!state.info;
    const hasHistory = loaded && state.info.history.length > 0;
    $("#applyBtn").disabled = state.busy || !loaded || !state.currentOp;
    $("#undoBtn").disabled = state.busy || !hasHistory;
    $("#resetBtn").disabled = state.busy || !hasHistory;
    $("#downloadBtn").disabled = state.busy || !loaded;
    $("#resetParamsBtn").disabled = !state.currentOp;
  }

  /* ------------------------------------------------------------ operation list */
  function renderOperationList(filter = "") {
    const host = $("#operationList");
    host.replaceChildren();
    const query = filter.trim().toLowerCase();
    let shown = 0;

    state.categories.forEach((category) => {
      const ops = state.ops.filter((op) => op.category === category &&
        (!query || op.label.toLowerCase().includes(query) || op.key.includes(query)));
      if (!ops.length) return;

      const group = el("div", "op-group");
      group.append(el("div", "op-group-title", category));
      const row = el("div", "op-buttons");
      ops.forEach((op) => {
        const button = el("button", "op-btn", op.label);
        button.type = "button";
        button.dataset.op = op.key;
        if (state.currentOp === op.key) button.classList.add("active");
        button.addEventListener("click", () => selectOperation(op.key));
        row.append(button);
        shown += 1;
      });
      group.append(row);
      host.append(group);
    });

    if (!shown) host.append(el("div", "text-body-secondary small py-2", "No operations match your search."));
  }

  function selectOperation(key) {
    const op = state.opMap[key];
    if (!op) return;
    state.currentOp = key;
    $$(".op-btn").forEach((button) => button.classList.toggle("active", button.dataset.op === key));
    $("#paramTitle").textContent = op.label;
    $("#paramDescription").textContent = op.description;
    renderParams(op);
    refreshButtons();
  }

  /* ------------------------------------------------------------ parameter controls */
  function appendHelp(wrap, spec) {
    if (spec.help) wrap.append(el("div", "form-text", spec.help));
  }

  function buildControl(spec, wrap) {
    const id = `param-${spec.name}`;

    if (spec.type === "bool") {
      const box = el("div", "form-check form-switch");
      const input = el("input", "form-check-input");
      input.type = "checkbox";
      input.id = id;
      input.setAttribute("role", "switch");
      input.checked = !!spec.default;
      const label = el("label", "form-check-label", spec.label);
      label.htmlFor = id;
      box.append(input, label);
      wrap.append(box);
      appendHelp(wrap, spec);
      input.addEventListener("change", updateVisibility);
      return { get: () => input.checked, set: (value) => { input.checked = !!value; } };
    }

    if (spec.type === "choice") {
      const label = el("label", "form-label small fw-semibold mb-1", spec.label);
      label.htmlFor = id;
      const select = el("select", "form-select form-select-sm");
      select.id = id;
      spec.choices.forEach((choice) => {
        const option = el("option", "", choice.label);
        option.value = String(choice.value);
        select.append(option);
      });
      select.value = String(spec.default);
      select.addEventListener("change", updateVisibility);
      wrap.append(label, select);
      appendHelp(wrap, spec);
      return { get: () => select.value, set: (value) => { select.value = String(value); } };
    }

    // int / float: slider + number box kept in sync
    const isInt = spec.type === "int";
    const header = el("div", "d-flex justify-content-between align-items-center mb-1 gap-2");
    const label = el("label", "form-label small fw-semibold mb-0", spec.label);
    label.htmlFor = `${id}-number`;
    const number = el("input", "form-control form-control-sm param-number");
    number.type = "number";
    number.id = `${id}-number`;
    number.min = spec.min;
    number.max = spec.max;
    number.step = spec.step;
    header.append(label, number);

    const range = el("input", "form-range");
    range.type = "range";
    range.min = spec.min;
    range.max = spec.max;
    range.step = spec.step;
    range.setAttribute("aria-label", spec.label);

    const normalize = (raw) => {
      let value = parseFloat(raw);
      if (!Number.isFinite(value)) value = spec.default;
      value = Math.min(spec.max, Math.max(spec.min, value));
      if (isInt) {
        value = Math.round(value);
        if (spec.odd && value % 2 === 0) value = value + 1 <= spec.max ? value + 1 : value - 1;
      } else {
        value = Math.round(value * 1000) / 1000;
      }
      return value;
    };
    const sync = (value) => { number.value = value; range.value = value; };

    range.addEventListener("input", () => sync(normalize(range.value)));
    number.addEventListener("change", () => sync(normalize(number.value)));
    sync(spec.default);

    wrap.append(header, range);
    appendHelp(wrap, spec);
    return {
      get: () => (isInt ? parseInt(number.value, 10) : parseFloat(number.value)),
      set: (value) => sync(normalize(value)),
    };
  }

  function renderParams(op) {
    const host = $("#paramForm");
    host.replaceChildren();
    state.controls = {};
    if (!op.params.length) {
      host.append(el("div", "text-body-secondary small", "This operation has no parameters."));
      return;
    }
    op.params.forEach((spec) => {
      const wrap = el("div", "param-item mb-3");
      wrap.dataset.param = spec.name;
      const control = buildControl(spec, wrap);
      state.controls[spec.name] = { spec, wrap, ...control };
      host.append(wrap);
    });
    applyDynamicDefaults();
    updateVisibility();
  }

  function currentValues() {
    const values = {};
    Object.entries(state.controls).forEach(([name, control]) => { values[name] = control.get(); });
    return values;
  }

  function updateVisibility() {
    const values = currentValues();
    Object.values(state.controls).forEach((control) => {
      const condition = control.spec.show_if;
      let visible = true;
      if (condition) {
        visible = Object.entries(condition).every(([name, allowed]) =>
          allowed.map(String).includes(String(values[name])));
      }
      control.wrap.classList.toggle("d-none", !visible);
    });
  }

  function applyDynamicDefaults() {
    if (!state.info) return;
    const sizes = {
      image_width: state.info.current_size.width,
      image_height: state.info.current_size.height,
    };
    Object.values(state.controls).forEach((control) => {
      const source = control.spec.default_from;
      if (source && sizes[source]) control.set(sizes[source]);
    });
  }

  function restoreDefaults() {
    Object.values(state.controls).forEach((control) => control.set(control.spec.default));
    applyDynamicDefaults();
    updateVisibility();
  }

  /* ------------------------------------------------------------ rendering state */
  function showEmptyState() {
    state.info = null;
    state.originalKey = null;
    ["original", "processed"].forEach((name) => {
      $(`#${name}Img`).classList.add("d-none");
      $(`#${name}Empty`).classList.remove("d-none");
      $(`#${name}Badge`).textContent = "";
    });
    $("#fileName").textContent = "";
    renderHistory();
    renderInfo();
    refreshButtons();
  }

  async function updateFromState(info) {
    state.info = info;
    const version = encodeURIComponent(info.token);
    const originalKey = info.token.split("-")[0];

    const loads = [setImage($("#processedImg"), `/api/image/processed?v=${version}`)];
    if (state.originalKey !== originalKey) {
      loads.push(setImage($("#originalImg"), `/api/image/original?v=${version}`));
      state.originalKey = originalKey;
    }
    await Promise.all(loads);

    ["original", "processed"].forEach((name) => {
      $(`#${name}Img`).classList.remove("d-none");
      $(`#${name}Empty`).classList.add("d-none");
    });
    $("#originalBadge").textContent = `${info.original_size.width} × ${info.original_size.height} px`;
    $("#processedBadge").textContent = `${info.current_size.width} × ${info.current_size.height} px`;
    $("#fileName").textContent = info.filename;

    renderHistory();
    renderInfo();
    markAnalysisDirty();
    refreshButtons();
  }

  function formatParams(params) {
    const entries = Object.entries(params || {});
    if (!entries.length) return "no parameters";
    return entries.map(([key, value]) => `${key.replace(/_/g, " ")}: ${value}`).join(" · ");
  }

  function renderHistory() {
    const list = $("#historyList");
    list.replaceChildren();
    const steps = state.info ? state.info.history : [];
    if (!steps.length) {
      const item = el("li", "text-body-secondary",
        state.info
          ? "No operations applied yet. Pick an operation, adjust its parameters and press Apply."
          : "Upload an image to start processing.");
      list.append(item);
      return;
    }
    steps.forEach((step) => {
      const item = el("li", "history-item");
      item.append(el("span", "history-index", String(step.index)));
      const body = el("div");
      body.append(el("div", "history-label", step.label), el("div", "history-params", formatParams(step.params)));
      item.append(body);
      list.append(item);
    });
  }

  function renderInfo() {
    const host = $("#infoContent");
    host.replaceChildren();
    if (!state.info) {
      host.append(el("p", "text-body-secondary mb-0", "No image loaded."));
      return;
    }
    const info = state.info;
    const rows = [
      ["File name", info.filename],
      ["Original size", `${info.original_size.width} × ${info.original_size.height} px`],
      ["Current size", `${info.current_size.width} × ${info.current_size.height} px`],
      ["Operations applied", `${info.history.length} of ${info.max_steps} allowed`],
      ["Pixel format", "8-bit, 3 channels (BGR in OpenCV)"],
    ];
    const table = el("table", "table table-sm info-table mb-0");
    const body = el("tbody");
    rows.forEach(([name, value]) => {
      const row = el("tr");
      row.append(el("th", "", name), el("td", "", value));
      body.append(row);
    });
    table.append(body);
    host.append(table);
  }

  /* ------------------------------------------------------------ analysis tabs */
  function markAnalysisDirty() {
    state.dirty.histogram = true;
    state.dirty.channels = true;
    refreshActiveAnalysis();
  }

  function refreshActiveAnalysis() {
    if (!state.info) return;
    if (state.activeTab === "histogram" && state.dirty.histogram) loadHistogram();
    if (state.activeTab === "channels" && state.dirty.channels) loadChannels();
  }

  function statsTable(stats) {
    const table = el("table", "table table-sm stats-table mb-0");
    const head = el("thead");
    const headRow = el("tr");
    ["Channel", "Mean", "Std", "Min", "Max", "Median"].forEach((name) => headRow.append(el("th", "", name)));
    head.append(headRow);
    const body = el("tbody");
    [["Red", "red"], ["Green", "green"], ["Blue", "blue"], ["Gray", "gray"]].forEach(([name, key]) => {
      const row = el("tr");
      const s = stats[key];
      [name, s.mean.toFixed(2), s.std.toFixed(2), s.min, s.max, s.median].forEach((value) =>
        row.append(el("td", "", String(value))));
      body.append(row);
    });
    table.append(head, body);
    return table;
  }

  function histogramColumn(title, report) {
    const col = el("div", "col-lg-6");
    col.append(el("h3", "fs-6 fw-semibold mb-2", title));
    [["RGB channels", report.rgb], ["Grayscale (luminance)", report.gray]].forEach(([caption, source]) => {
      const figure = el("figure", "hist-figure");
      const img = el("img");
      img.src = source;
      img.alt = `${title}: ${caption} histogram`;
      figure.append(img, el("figcaption", "small text-body-secondary mt-1", caption));
      col.append(figure);
    });
    col.append(statsTable(report.stats));
    return col;
  }

  async function loadHistogram() {
    const host = $("#histogramContent");
    state.dirty.histogram = false;
    host.classList.add("opacity-50");
    try {
      const data = await postJson("/api/histogram");
      const row = el("div", "row g-4");
      row.append(histogramColumn("Original", data.original), histogramColumn("Processed", data.processed));
      host.replaceChildren(row);
    } catch (error) {
      state.dirty.histogram = true;
      toast(error.message);
    } finally {
      host.classList.remove("opacity-50");
    }
  }

  async function loadChannels() {
    const host = $("#channelsContent");
    state.dirty.channels = false;
    host.classList.add("opacity-50");
    try {
      const data = await postJson("/api/channels", {
        source: $("#channelSource").value,
        mode: $("#channelMode").value,
      });
      const row = el("div", "row g-3");
      data.channels.forEach((channel) => {
        const col = el("div", "col-md-4");
        const figure = el("figure", "channel-card");
        const img = el("img");
        img.src = channel.image;
        img.alt = `${channel.name} channel`;
        figure.append(img, el("figcaption", "", channel.name));
        col.append(figure);
        row.append(col);
      });
      host.replaceChildren(row);
    } catch (error) {
      state.dirty.channels = true;
      toast(error.message);
    } finally {
      host.classList.remove("opacity-50");
    }
  }

  $$('#analysisTabs button[data-bs-toggle="tab"]').forEach((button) => {
    button.addEventListener("shown.bs.tab", (event) => {
      state.activeTab = event.target.dataset.tab;
      refreshActiveAnalysis();
    });
  });
  $("#channelSource").addEventListener("change", () => { state.dirty.channels = true; refreshActiveAnalysis(); });
  $("#channelMode").addEventListener("change", () => { state.dirty.channels = true; refreshActiveAnalysis(); });

  /* ------------------------------------------------------------ actions */
  async function applyOperation() {
    if (!state.info) { toast("Upload an image first.", "warning"); return; }
    if (!state.currentOp) { toast("Select an operation first.", "warning"); return; }
    const op = state.opMap[state.currentOp];
    await runBusy(async () => {
      const payload = await postJson("/api/process", { operation: op.key, params: currentValues() });
      await updateFromState(payload);
      applyDynamicDefaults();
      toast(`${op.label} applied.`, "success");
    });
  }

  async function undo() {
    await runBusy(async () => {
      await updateFromState(await postJson("/api/undo"));
      applyDynamicDefaults();
    });
  }

  async function resetImage() {
    await runBusy(async () => {
      await updateFromState(await postJson("/api/reset"));
      applyDynamicDefaults();
      toast("Restored the original image.", "success");
    });
  }

  async function handleFile(file) {
    if (!file) return;
    const maxMb = state.limits.max_upload_mb || 10;
    const allowed = state.limits.allowed_extensions || [];
    const extension = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if (allowed.length && !allowed.includes(extension)) {
      toast(`Unsupported file type${extension ? ` ".${extension}"` : ""}. Allowed: ${allowed.join(", ")}.`, "warning");
      return;
    }
    if (file.size === 0) { toast("That file is empty.", "warning"); return; }
    if (file.size > maxMb * 1024 * 1024) {
      toast(`The file is ${(file.size / 1048576).toFixed(1)} MB. The limit is ${maxMb} MB.`, "warning");
      return;
    }
    await runBusy(async () => {
      const form = new FormData();
      form.append("image", file);
      const payload = await api("/api/upload", { method: "POST", body: form });
      await updateFromState(payload);
      applyDynamicDefaults();
      toast(payload.notice || `Loaded ${payload.filename}.`, payload.notice ? "info" : "success");
    });
  }

  /* ------------------------------------------------------------ events */
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", () => {
    handleFile(fileInput.files[0]);
    fileInput.value = "";          // allow choosing the same file again
  });
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  }));
  dropzone.addEventListener("drop", (event) => {
    const files = event.dataTransfer && event.dataTransfer.files;
    if (files && files.length) handleFile(files[0]);
  });
  // A file dropped outside the drop zone must not make the browser navigate away.
  ["dragover", "drop"].forEach((name) => window.addEventListener(name, (event) => event.preventDefault()));

  $("#opSearch").addEventListener("input", (event) => renderOperationList(event.target.value));
  $("#applyBtn").addEventListener("click", applyOperation);
  $("#resetParamsBtn").addEventListener("click", restoreDefaults);
  $("#undoBtn").addEventListener("click", undo);
  $("#resetBtn").addEventListener("click", resetImage);

  $$("[data-download]").forEach((link) => link.addEventListener("click", (event) => {
    if (!state.info || state.busy) {
      event.preventDefault();
      toast("Upload an image first.", "warning");
    }
  }));

  const viewerModal = new bootstrap.Modal($("#viewerModal"));
  [$("#originalImg"), $("#processedImg")].forEach((img) => img.addEventListener("click", () => {
    $("#viewerTitle").textContent = img.dataset.title;
    $("#viewerImg").src = img.src;
    viewerModal.show();
  }));

  /* ------------------------------------------------------------ start-up */
  async function init() {
    try {
      const catalogue = await api("/api/operations");
      state.categories = catalogue.categories;
      state.ops = catalogue.operations;
      state.limits = catalogue.limits;
      state.opMap = Object.fromEntries(catalogue.operations.map((op) => [op.key, op]));
      $("#limitText").textContent =
        `${state.limits.allowed_extensions.join(", ").toUpperCase()} · up to ${state.limits.max_upload_mb} MB.`;
      renderOperationList();
      if (state.ops.length) selectOperation(state.ops[0].key);
    } catch (error) {
      toast(`Could not load the operations: ${error.message}`);
    }

    try {
      const current = await api("/api/state");
      if (current.loaded) {
        await updateFromState(current);
        applyDynamicDefaults();
      } else {
        showEmptyState();
      }
    } catch (error) {
      showEmptyState();
      toast(error.message);
    }
    refreshButtons();
  }

  init();
})();
