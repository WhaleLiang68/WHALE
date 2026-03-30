const state = {
  files: [],
  selectedCsv: "",
  rows: [],
  filteredRows: [],
  selectedRunIndex: null,
  payload: null,
  archiveCache: {},
  instances: [],
  generatorLayout: null,
  activeDetailTab: "overview",
  tableSort: { key: "runIndex", direction: "desc" },
};

const EMPTY_REMARK_TOKEN = "__EMPTY__";
const SVG_NS = "http://www.w3.org/2000/svg";

const SINGLE_TABLE_SORT_COLUMNS = [
  { key: "runIndex", label: "\u8fd0\u884c", type: "number" },
  { key: "date", label: "\u65e5\u671f", type: "date" },
  { key: "fitness", label: "\u9002\u5e94\u5ea6", type: "number" },
  { key: "runtimeSeconds", label: "\u8fd0\u884c\u65f6\u95f4", type: "number" },
  { key: "bestResultSeconds", label: "\u6700\u597d\u89e3\u8017\u65f6", type: "number" },
  { key: "gbestUpdates", label: "gbest", type: "number" },
  { key: "aspectRatioValid", label: "\u5bbd\u9ad8\u6bd4", type: "validity" },
];

const MULTI_TABLE_SORT_COLUMNS = [
  { key: "runIndex", label: "\u8fd0\u884c", type: "number" },
  { key: "date", label: "\u65e5\u671f", type: "date" },
  { key: "decisionScore", label: "\u51b3\u7b56\u5206\u6570", type: "number" },
  { key: "paretoSize", label: "Pareto \u6863\u6848", type: "number" },
  { key: "repMhc", label: "MHC", type: "number" },
  { key: "repCr", label: "CR", type: "number" },
  { key: "repDr", label: "DR", type: "number" },
  { key: "repAr", label: "AR", type: "number" },
  { key: "aspectRatioValid", label: "\u5bbd\u9ad8\u6bd4", type: "validity" },
];

const MIXED_TABLE_SORT_COLUMNS = [
  { key: "runIndex", label: "\u8fd0\u884c", type: "number" },
  { key: "viewMode", label: "\u89c6\u56fe", type: "mode" },
  { key: "date", label: "\u65e5\u671f", type: "date" },
  { key: "primaryMetric", label: "\u6838\u5fc3\u6307\u6807", type: "number" },
  { key: "runtimeSeconds", label: "\u8fd0\u884c\u65f6\u95f4", type: "number" },
  { key: "secondaryMetric", label: "\u5173\u952e\u8865\u5145\u6307\u6807", type: "number" },
  { key: "aspectRatioValid", label: "\u5bbd\u9ad8\u6bd4", type: "validity" },
];

const elements = {
  fileSelect: document.getElementById("fileSelect"),
  searchInput: document.getElementById("searchInput"),
  validityFilter: document.getElementById("validityFilter"),
  dateFilter: document.getElementById("dateFilter"),
  remarkFilter: document.getElementById("remarkFilter"),
  runRange: document.getElementById("runRange"),
  resetFilters: document.getElementById("resetFilters"),
  refreshButton: document.getElementById("refreshButton"),
  summaryCards: document.getElementById("summaryCards"),
  currentFile: document.getElementById("currentFile"),
  currentInstance: document.getElementById("currentInstance"),
  currentAlgorithm: document.getElementById("currentAlgorithm"),
  currentViewMode: document.getElementById("currentViewMode"),
  statusBanner: document.getElementById("statusBanner"),
  tablePanelTitle: document.getElementById("tablePanelTitle"),
  chartPanel1Title: document.getElementById("chartPanel1Title"),
  chartPanel1Subtitle: document.getElementById("chartPanel1Subtitle"),
  chartPanel2Title: document.getElementById("chartPanel2Title"),
  chartPanel2Subtitle: document.getElementById("chartPanel2Subtitle"),
  chartPanel3Title: document.getElementById("chartPanel3Title"),
  chartPanel3Subtitle: document.getElementById("chartPanel3Subtitle"),
  chartPanel4Title: document.getElementById("chartPanel4Title"),
  chartPanel4Subtitle: document.getElementById("chartPanel4Subtitle"),
  fitnessChart: document.getElementById("fitnessChart"),
  timingChart: document.getElementById("timingChart"),
  histogramChart: document.getElementById("histogramChart"),
  scatterChart: document.getElementById("scatterChart"),
  resultsTable: document.getElementById("resultsTable"),
  tableMeta: document.getElementById("tableMeta"),
  focusPanelTitle: document.getElementById("focusPanelTitle"),
  focusPanelSubtitle: document.getElementById("focusPanelSubtitle"),
  detailPanelTitle: document.getElementById("detailPanelTitle"),
  detailPanelSubtitle: document.getElementById("detailPanelSubtitle"),
  detailContent: document.getElementById("detailContent"),
  instanceSelect: document.getElementById("instanceSelect"),
  docPathInput: document.getElementById("docPathInput"),
  extractIndexInput: document.getElementById("extractIndexInput"),
  manualSolutionInput: document.getElementById("manualSolutionInput"),
  generateFromRunButton: document.getElementById("generateFromRunButton"),
  generateFromDocButton: document.getElementById("generateFromDocButton"),
  generateManualButton: document.getElementById("generateManualButton"),
  clearManualButton: document.getElementById("clearManualButton"),
  generatorMeta: document.getElementById("generatorMeta"),
  generatorLayoutHost: document.getElementById("generatorLayoutHost"),
  downloadSvgButton: document.getElementById("downloadSvgButton"),
  downloadPngButton: document.getElementById("downloadPngButton"),
  tooltip: document.getElementById("chartTooltip"),
};

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  initialize();
});

function bindEvents() {
  elements.fileSelect.addEventListener("change", () => {
    const csvPath = elements.fileSelect.value;
    if (csvPath) {
      loadResults(csvPath);
    }
  });

  elements.searchInput.addEventListener("input", render);
  elements.validityFilter.addEventListener("change", render);
  elements.dateFilter?.addEventListener("change", render);
  elements.remarkFilter?.addEventListener("change", render);
  elements.runRange.addEventListener("change", render);

  elements.resetFilters.addEventListener("click", () => {
    elements.searchInput.value = "";
    elements.validityFilter.value = "all";
    if (elements.dateFilter) elements.dateFilter.value = "all";
    if (elements.remarkFilter) elements.remarkFilter.value = "all";
    elements.runRange.value = "all";
    render();
  });

  elements.refreshButton.addEventListener("click", () => {
    if (state.selectedCsv) {
      loadResults(state.selectedCsv, true);
    }
  });

  elements.generateFromRunButton?.addEventListener("click", generateLayoutFromRun);
  elements.generateFromDocButton?.addEventListener("click", generateLayoutFromDocument);
  elements.generateManualButton?.addEventListener("click", generateLayoutFromManual);
  elements.clearManualButton?.addEventListener("click", () => {
    if (elements.manualSolutionInput) {
      elements.manualSolutionInput.value = "";
      elements.manualSolutionInput.focus();
    }
  });
  elements.downloadSvgButton?.addEventListener("click", downloadCurrentSvg);
  elements.downloadPngButton?.addEventListener("click", downloadCurrentPng);

  window.addEventListener("resize", debounce(renderChartsOnly, 120));
}

async function initialize() {
  try {
    showStatus("正在加载结果文件列表...", false);
    const filePayload = await requestJson("/api/files");
    state.files = filePayload.files || [];
    state.selectedCsv = filePayload.defaultCsv || state.files[0] || "";
    renderFileOptions();
    await loadInstances();

    if (!state.selectedCsv) {
      throw new Error("没有找到可用的结果文件。");
    }

    await loadResults(state.selectedCsv);
  } catch (error) {
    showStatus(error.message || "初始化失败。", true);
  }
}

async function loadResults(csvPath, silent = false) {
  try {
    if (!silent) {
      showStatus("正在读取结果数据...", false);
    }

    const payload = await requestJson(`/api/results?csv=${encodeURIComponent(csvPath)}`);
    state.payload = payload;
    state.rows = payload.rows || [];
    state.selectedCsv = payload.csvPath;
    state.archiveCache = {};

    syncGeneratorDefaultInstance();
    renderFileOptions();
    renderDateRemarkOptions();
    render();
    hideStatus();
  } catch (error) {
    showStatus(error.message || "读取结果数据失败。", true);
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    ...options,
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(payload?.error || `Request failed (${response.status}).`);
  }

  return payload || {};
}

function render() {
  state.filteredRows = applyFilters(state.rows);
  ensureSelectionWithinRows();
  ensureActiveDetailTab(getSelectedRow());
  renderHeader();
  renderPanelHeaders();
  renderSummary();
  renderTable();
  renderDetail();
  syncDynamicDetailElements();
  renderChartsOnly();
}

function renderHeader() {
  const currentRow = getSelectedRow() || state.filteredRows[0] || state.rows[0] || null;
  elements.currentFile.textContent = state.selectedCsv || "-";
  elements.currentInstance.textContent = currentRow?.instance || "-";
  elements.currentAlgorithm.textContent = currentRow?.algorithm || "-";
  if (elements.currentViewMode) {
    elements.currentViewMode.innerHTML = currentRow ? createViewModeBadgeText(getRowViewMode(currentRow)) : "-";
  }
  syncGeneratorDefaultInstance();
}

function isMultiObjectiveRow(row) {
  return Boolean(row?.paretoArchivePath || row?.moRunId) || isNumber(row?.repMhc) || isNumber(row?.paretoSize);
}

function getRowViewMode(row) {
  return isMultiObjectiveRow(row) ? "mo" : "so";
}

function getViewModeLabel(mode) {
  return mode === "mo" ? "\u591a\u76ee\u6807\u89c6\u56fe" : "\u5355\u76ee\u6807\u89c6\u56fe";
}

function getViewModeShortLabel(mode) {
  return mode === "mo" ? "MO \u591a\u76ee\u6807" : "SO \u5355\u76ee\u6807";
}

function getRowsViewMode(rows = state.filteredRows) {
  if (!rows.length) {
    return "so";
  }

  const modes = new Set(rows.map((row) => getRowViewMode(row)));
  if (modes.size === 1) {
    return modes.has("mo") ? "mo" : "so";
  }

  return "mixed";
}

function getRowsForViewMode(mode, rows = state.filteredRows) {
  return rows.filter((row) => getRowViewMode(row) === mode);
}

function getCurrentViewMode() {
  const selectedRow = getSelectedRow();
  if (selectedRow) {
    return getRowViewMode(selectedRow);
  }

  const rows = state.filteredRows.length ? state.filteredRows : state.rows;
  const rowsMode = getRowsViewMode(rows);
  if (rowsMode !== "mixed") {
    return rowsMode;
  }

  const fallbackRow = rows[rows.length - 1] || null;
  return fallbackRow ? getRowViewMode(fallbackRow) : "so";
}

function getCurrentViewRows() {
  return getRowsForViewMode(getCurrentViewMode(), state.filteredRows);
}

function isCurrentViewMultiObjective() {
  return getCurrentViewMode() === "mo";
}

function currentFitnessLabel(row = getSelectedRow()) {
  return getRowViewMode(row) === "mo" ? "\u51b3\u7b56\u5206\u6570" : "\u9002\u5e94\u5ea6";
}

function getTableColumns() {
  const rows = state.filteredRows.length ? state.filteredRows : state.rows;
  const mode = getRowsViewMode(rows);
  if (mode === "mixed") {
    return MIXED_TABLE_SORT_COLUMNS;
  }
  return mode === "mo" ? MULTI_TABLE_SORT_COLUMNS : SINGLE_TABLE_SORT_COLUMNS;
}

function currentPrimaryMetricValue(row) {
  if (getRowViewMode(row) === "mo") {
    return isNumber(row.decisionScore) ? row.decisionScore : row.fitness;
  }
  return row?.fitness;
}

function currentSecondaryMetricValue(row) {
  return getRowViewMode(row) === "mo" ? row?.paretoSize : row?.gbestUpdates;
}

function createViewModeBadgeText(mode) {
  return `<span class="badge mode-badge ${mode === "mo" ? "mode-mo" : "mode-so"}">${getViewModeShortLabel(mode)}</span>`;
}

function syncDynamicDetailElements() {
  elements.chartPanel3Title = document.getElementById("chartPanel3Title");
  elements.chartPanel3Subtitle = document.getElementById("chartPanel3Subtitle");
  elements.chartPanel4Title = document.getElementById("chartPanel4Title");
  elements.chartPanel4Subtitle = document.getElementById("chartPanel4Subtitle");
  elements.histogramChart = document.getElementById("histogramChart");
  elements.scatterChart = document.getElementById("scatterChart");
}

function renderPanelHeaders() {
  const currentRow = getSelectedRow();
  const currentMode = getCurrentViewMode();
  const tableMode = getRowsViewMode(state.filteredRows);
  const tabs = getDetailTabsForRow(currentRow).map((tab) => tab.label).join(" / ");

  if (elements.tablePanelTitle) {
    elements.tablePanelTitle.textContent =
      tableMode === "mixed"
        ? "\u8fd0\u884c\u660e\u7ec6\u5de5\u4f5c\u53f0"
        : currentMode === "mo"
        ? "\u591a\u76ee\u6807\u8fd0\u884c\u660e\u7ec6"
        : "\u5355\u76ee\u6807\u8fd0\u884c\u660e\u7ec6";
  }

  if (elements.focusPanelTitle) {
    elements.focusPanelTitle.textContent = currentMode === "mo"
      ? "\u591a\u76ee\u6807\u8fd0\u884c\u6458\u8981"
      : "\u5355\u76ee\u6807\u8fd0\u884c\u6458\u8981";
  }

  if (elements.focusPanelSubtitle) {
    elements.focusPanelSubtitle.textContent = currentRow
      ? `\u5df2\u9009\u8fd0\u884c #${currentRow.runIndex} \u00b7 ${currentRow.instance || "-"} / ${currentRow.algorithm || "-"}`
      : "\u5f53\u524d\u9762\u677f\u53ea\u5c55\u793a\u9009\u4e2d\u8fd0\u884c\u7684\u6838\u5fc3\u6458\u8981\u4e0e\u9996\u9875\u6982\u89c8\u56fe\u3002";
  }

  if (elements.detailPanelTitle) {
    elements.detailPanelTitle.textContent = currentMode === "mo"
      ? "\u591a\u76ee\u6807\u8fd0\u884c\u8be6\u60c5"
      : "\u5355\u76ee\u6807\u8fd0\u884c\u8be6\u60c5";
  }

  if (elements.detailPanelSubtitle) {
    elements.detailPanelSubtitle.textContent = currentRow
      ? `\u5f53\u524d\u67e5\u770b #${currentRow.runIndex} \u00b7 \u53ef\u5207\u6362\u6807\u7b7e\uff1a${tabs}`
      : "\u5728\u4e0b\u65b9\u6807\u7b7e\u9875\u4e2d\u67e5\u770b\u6982\u89c8\u3001\u5e03\u5c40\u3001\u5206\u6790\u4e0e\u5907\u6ce8\u3002";
  }
}

function defaultTableSortDirection(key) {
  if (["runIndex", "date", "paretoSize", "repCr", "repDr", "repAr", "secondaryMetric"].includes(key)) {
    return "desc";
  }
  return "asc";
}

function ensureTableSortKey() {
  const columns = getTableColumns();
  if (!columns.some((column) => column.key === state.tableSort.key)) {
    state.tableSort = { key: "runIndex", direction: "desc" };
  }
}

function setChartPanelMetadata(panels) {
  const targets = [
    [elements.chartPanel1Title, elements.chartPanel1Subtitle],
    [elements.chartPanel2Title, elements.chartPanel2Subtitle],
    [elements.chartPanel3Title, elements.chartPanel3Subtitle],
    [elements.chartPanel4Title, elements.chartPanel4Subtitle],
  ];

  targets.forEach(([titleNode, subtitleNode], index) => {
    const panel = panels[index] || {};
    if (titleNode) titleNode.textContent = panel.title || "-";
    if (subtitleNode) subtitleNode.textContent = panel.subtitle || "";
  });
}

function renderFileOptions() {
  const files = state.files.length ? state.files : [state.selectedCsv].filter(Boolean);
  elements.fileSelect.innerHTML = "";

  files.forEach((filePath) => {
    const option = document.createElement("option");
    option.value = filePath;
    option.textContent = filePath;
    option.selected = filePath === state.selectedCsv;
    elements.fileSelect.appendChild(option);
  });
}

function normalizeFilterText(value) {
  return String(value ?? "").trim();
}

function renderDateRemarkOptions() {
  if (!elements.dateFilter || !elements.remarkFilter) {
    return;
  }

  const previousDate = elements.dateFilter.value || "all";
  const previousRemark = elements.remarkFilter.value || "all";

  const uniqueDates = [...new Set(state.rows.map((row) => normalizeFilterText(row.date)).filter(Boolean))]
    .sort((left, right) => right.localeCompare(left));

  const uniqueRemarks = [...new Set(state.rows.map((row) => normalizeFilterText(row.remark)))].sort(
    (left, right) => left.localeCompare(right),
  );

  elements.dateFilter.innerHTML = "";
  const allDateOption = document.createElement("option");
  allDateOption.value = "all";
  allDateOption.textContent = "全部日期";
  elements.dateFilter.appendChild(allDateOption);

  uniqueDates.forEach((dateValue) => {
    const option = document.createElement("option");
    option.value = dateValue;
    option.textContent = dateValue;
    elements.dateFilter.appendChild(option);
  });

  elements.remarkFilter.innerHTML = "";
  const allRemarkOption = document.createElement("option");
  allRemarkOption.value = "all";
  allRemarkOption.textContent = "全部备注";
  elements.remarkFilter.appendChild(allRemarkOption);

  uniqueRemarks.forEach((remarkValue) => {
    const option = document.createElement("option");
    const isEmpty = remarkValue === "";
    option.value = isEmpty ? EMPTY_REMARK_TOKEN : remarkValue;
    option.textContent = isEmpty
      ? "(空备注)"
      : remarkValue.length > 24
      ? `${remarkValue.slice(0, 24)}...`
      : remarkValue;
    option.title = remarkValue;
    elements.remarkFilter.appendChild(option);
  });

  elements.dateFilter.value =
    previousDate !== "all" && uniqueDates.includes(previousDate) ? previousDate : "all";

  const hasPreviousRemark =
    previousRemark === EMPTY_REMARK_TOKEN
      ? uniqueRemarks.includes("")
      : previousRemark !== "all" && uniqueRemarks.includes(previousRemark);
  elements.remarkFilter.value = hasPreviousRemark ? previousRemark : "all";
}

function applyFilters(rows) {
  const keyword = elements.searchInput.value.trim().toLowerCase();
  const validity = elements.validityFilter.value;
  const dateValue = elements.dateFilter?.value || "all";
  const remarkValue = elements.remarkFilter?.value || "all";
  const runRange = elements.runRange.value;

  let filtered = [...rows];

  if (keyword) {
    filtered = filtered.filter((row) => {
      const haystack = [
        row.instance,
        row.algorithm,
        row.date,
        row.solution,
        row.remark,
        row.startTime,
        row.endTime,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return haystack.includes(keyword);
    });
  }

  if (validity !== "all") {
    filtered = filtered.filter((row) => {
      if (validity === "true") return row.aspectRatioValid === true;
      if (validity === "false") return row.aspectRatioValid === false;
      return row.aspectRatioValid === null;
    });
  }

  if (dateValue !== "all") {
    filtered = filtered.filter((row) => normalizeFilterText(row.date) === dateValue);
  }

  if (remarkValue !== "all") {
    filtered = filtered.filter((row) => {
      const remarkText = normalizeFilterText(row.remark);
      if (remarkValue === EMPTY_REMARK_TOKEN) {
        return remarkText === "";
      }
      return remarkText === remarkValue;
    });
  }

  filtered.sort((left, right) => left.runIndex - right.runIndex);

  if (runRange !== "all") {
    filtered = filtered.slice(-Number(runRange));
  }

  return filtered;
}


function ensureSelectionWithinRows() {
  if (!state.filteredRows.length) {
    state.selectedRunIndex = null;
    return;
  }

  const selectedExists = state.filteredRows.some(
    (row) => row.runIndex === state.selectedRunIndex,
  );

  if (selectedExists) {
    return;
  }

  const rowsMode = getRowsViewMode(state.filteredRows);
  if (rowsMode === "mixed") {
    state.selectedRunIndex = state.filteredRows[state.filteredRows.length - 1].runIndex;
    return;
  }

  const bestRow = state.filteredRows.reduce((best, row) => {
    const currentValue = currentPrimaryMetricValue(row);
    if (!isNumber(currentValue)) {
      return best;
    }

    if (!best) {
      return row;
    }

    const bestValue = currentPrimaryMetricValue(best);
    if (!isNumber(bestValue) || currentValue < bestValue) {
      return row;
    }

    if (
      rowsMode === "mo" &&
      currentValue === bestValue &&
      (row.paretoSize || 0) > (best.paretoSize || 0)
    ) {
      return row;
    }

    return best;
  }, null);

  state.selectedRunIndex = (bestRow || state.filteredRows[0]).runIndex;
}


function createSummaryCard(card) {
  const valueHtml = card.valueHtml ?? escapeHtml(String(card.value ?? "-"));
  const metaHtml = card.meta ? `<div class="summary-meta">${card.meta}</div>` : `<div class="summary-meta">&nbsp;</div>`;
  return `
    <article class="summary-card summary-card-compact ${card.tone || "ink"}">
      <span class="summary-label">${card.label}</span>
      <strong class="summary-value ${card.valueClass || ""}">${valueHtml}</strong>
      ${metaHtml}
    </article>
  `;
}

function renderSummary() {
  const row = getSelectedRow();
  elements.summaryCards.innerHTML = "";

  if (!row) {
    elements.summaryCards.innerHTML = `<div class="empty-state">\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u8fd0\u884c\u6458\u8981\u3002</div>`;
    return;
  }

  const cards = getRowViewMode(row) === "mo"
    ? [
        { label: "\u51b3\u7b56\u5206\u6570", value: formatNumber(row.decisionScore ?? row.fitness, 4), meta: `\u8fd0\u884c #${row.runIndex}`, tone: "green" },
        { label: "Pareto \u6863\u6848\u5927\u5c0f", value: formatInteger(row.paretoSize), meta: `\u8fed\u4ee3 ${formatInteger(row.iterations)}`, tone: "gold" },
        { label: "\u4ee3\u8868\u89e3 MHC", value: formatNumber(row.repMhc, 3), meta: "\u5f53\u524d\u4ee3\u8868\u89e3", tone: "blue" },
        { label: "\u4ee3\u8868\u89e3 CR", value: formatNumber(row.repCr, 3), meta: "\u5f53\u524d\u4ee3\u8868\u89e3", tone: "orange" },
        { label: "\u4ee3\u8868\u89e3 DR", value: formatNumber(row.repDr, 3), meta: "\u5f53\u524d\u4ee3\u8868\u89e3", tone: "blue" },
        { label: "\u4ee3\u8868\u89e3 AR", value: formatNumber(row.repAr, 3), meta: "\u5f53\u524d\u4ee3\u8868\u89e3", tone: "ink" },
        { label: "\u4ee3\u8868\u89e3\u66f4\u65b0\u6b21\u6570", value: formatInteger(row.gbestUpdates), meta: `\u603b\u8017\u65f6 ${formatSeconds(row.runtimeSeconds, 1)}`, tone: "green" },
        { label: "\u5bbd\u9ad8\u6bd4", valueHtml: createBadge(row.aspectRatioValid), meta: formatDateCell(row.date), valueClass: "summary-value-badge", tone: "ink" },
      ]
    : [
        { label: "\u9002\u5e94\u5ea6", value: formatNumber(row.fitness, 3), meta: `\u8fd0\u884c #${row.runIndex}`, tone: "green" },
        { label: "\u603b\u8fd0\u884c\u65f6\u95f4", value: formatSeconds(row.runtimeSeconds, 1), meta: formatDateCell(row.date), tone: "orange" },
        { label: "\u8fbe\u5230\u6700\u597d\u89e3\u65f6\u95f4", value: formatSeconds(row.bestResultSeconds, 1), meta: `\u8fed\u4ee3 ${formatInteger(row.iterations)}`, tone: "blue" },
        { label: "gbest \u66f4\u65b0\u6b21\u6570", value: formatInteger(row.gbestUpdates), meta: "\u5168\u5c40\u6700\u597d\u89e3\u5237\u65b0\u6b21\u6570", tone: "gold" },
        { label: "\u5bbd\u9ad8\u6bd4", valueHtml: createBadge(row.aspectRatioValid), meta: formatDateTime(row.endTime), valueClass: "summary-value-badge", tone: "ink" },
      ];

  elements.summaryCards.innerHTML = cards.map(createSummaryCard).join("");
}

function renderChartsOnly() {
  if (isCurrentViewMultiObjective()) {
    setChartPanelMetadata([
      {
        title: "\u8fd0\u884c\u7ea7\u6982\u89c8\u56fe",
        subtitle: "\u6309\u8fd0\u884c\u5e8f\u53f7\u67e5\u770b decision score \u4e0e Pareto \u6863\u6848\u89c4\u6a21\u53d8\u5316\u3002",
      },
      {
        title: "Pareto \u524d\u6cbf\u6563\u70b9\u56fe",
        subtitle: "\u5f53\u524d\u8fd0\u884c\u7684 Pareto \u524d\u6cbf\uff0c\u6a2a\u8f74 MHC\uff0c\u7eb5\u8f74 CR\uff0c\u989c\u8272\u8868\u793a DR\u3002",
      },
      {
        title: "\u4ee3\u8868\u89e3\u5bf9\u6bd4\u5361",
        subtitle: "\u4ece\u5f53\u524d\u8fd0\u884c\u7684 Pareto \u6863\u6848\u4e2d\u6bd4\u8f83\u4ee3\u8868\u89e3\u4e0e\u524d\u6cbf\u7edf\u8ba1\u3002",
      },
      {
        title: "\u5e73\u884c\u5750\u6807\u56fe",
        subtitle: "\u5728\u56db\u76ee\u6807\u7a7a\u95f4\u67e5\u770b\u5f53\u524d Pareto \u6863\u6848\u7684\u5206\u5e03\u3002",
      },
    ]);
    renderMoRunOverviewChart(elements.fitnessChart);
    renderMoParetoScatterChart(elements.timingChart);
    renderMoRepresentativeComparisonChart(elements.histogramChart);
    renderMoParallelCoordinatesChart(elements.scatterChart);
    return;
  }

  setChartPanelMetadata([
    { title: "\u9002\u5e94\u5ea6\u8d8b\u52bf", subtitle: "\u6309\u8fd0\u884c\u5e8f\u53f7\u67e5\u770b\u5355\u6b21\u9002\u5e94\u5ea6\u4e0e\u7d2f\u8ba1\u6700\u597d\u503c\u3002" },
    { title: "\u8017\u65f6\u5bf9\u6bd4", subtitle: "\u6bd4\u8f83\u603b\u8fd0\u884c\u65f6\u95f4\u4e0e\u8fbe\u5230\u6700\u597d\u89e3\u65f6\u95f4\u3002" },
    { title: "\u9002\u5e94\u5ea6\u5206\u5e03", subtitle: "\u67e5\u770b\u5f53\u524d\u7b5b\u9009\u7ed3\u679c\u4e2d\u7684\u9002\u5e94\u5ea6\u5206\u5e03\u3002" },
    { title: "\u9002\u5e94\u5ea6-\u8017\u65f6\u6563\u70b9", subtitle: "\u6bd4\u8f83\u8fd0\u884c\u65f6\u95f4\u3001\u9002\u5e94\u5ea6\u4e0e\u5bbd\u9ad8\u6bd4\u72b6\u6001\u3002" },
  ]);
  renderFitnessChart();
  renderTimingChart();
  renderHistogramChart();
  renderScatterChart();
}

function appendChartNote(container, text) {
  const note = document.createElement("div");
  note.className = "chart-note";
  note.textContent = text;
  container.appendChild(note);
}

function toArchiveItems(payload) {
  const items = Array.isArray(payload?.items) ? payload.items : [];
  return items
    .map((item) => ({
      index: Number(item.index),
      decisionScore: Number(item.decisionScore),
      mhc: Number(item.mhc),
      cr: Number(item.cr),
      dr: Number(item.dr),
      ar: Number(item.ar),
      dInf: Number(item.dInf),
    }))
    .filter((item) => Number.isFinite(item.index));
}

function nearlyEqual(left, right, epsilon = 1e-6) {
  if (!isNumber(left) || !isNumber(right)) {
    return false;
  }
  return Math.abs(left - right) <= epsilon * Math.max(1, Math.abs(left), Math.abs(right));
}

function findRepresentativeArchiveItem(payload, row) {
  const items = toArchiveItems(payload);
  const representativeIndex = Number(payload?.representativeArchiveIndex);
  const byIndex = items.find((item) => item.index === representativeIndex);
  if (byIndex) {
    return byIndex;
  }

  return (
    items.find(
      (item) =>
        nearlyEqual(item.mhc, row.repMhc) &&
        nearlyEqual(item.cr, row.repCr) &&
        nearlyEqual(item.dr, row.repDr) &&
        nearlyEqual(item.ar, row.repAr),
    ) || items[0] || null
  );
}

async function getArchivePayload(row) {
  const archivePath = String(row?.paretoArchivePath || "").trim();
  if (!archivePath) {
    return null;
  }

  let payload = state.archiveCache[archivePath];
  if (!payload) {
    payload = await requestJson(`/api/archive?path=${encodeURIComponent(archivePath)}`);
    state.archiveCache[archivePath] = payload;
  }
  return payload;
}

function interpolateSeriesColor(value, minValue, maxValue) {
  if (!isNumber(value) || !isNumber(minValue) || !isNumber(maxValue) || minValue === maxValue) {
    return "#3f6ecf";
  }

  const ratio = clamp((value - minValue) / (maxValue - minValue), 0, 1);
  const start = { r: 63, g: 110, b: 207 };
  const end = { r: 14, g: 133, b: 120 };
  const r = Math.round(start.r + (end.r - start.r) * ratio);
  const g = Math.round(start.g + (end.g - start.g) * ratio);
  const b = Math.round(start.b + (end.b - start.b) * ratio);
  return `rgb(${r}, ${g}, ${b})`;
}

function normalizeScore(value, minValue, maxValue, invert = false) {
  if (!isNumber(value) || !isNumber(minValue) || !isNumber(maxValue)) {
    return 0.5;
  }
  if (minValue === maxValue) {
    return 0.5;
  }
  const ratio = clamp((value - minValue) / (maxValue - minValue), 0, 1);
  return invert ? 1 - ratio : ratio;
}

function sampleArchiveItems(items, limit = 36) {
  if (items.length <= limit) {
    return items;
  }
  const sampled = [];
  const step = (items.length - 1) / (limit - 1);
  for (let index = 0; index < limit; index += 1) {
    sampled.push(items[Math.round(index * step)]);
  }
  return sampled;
}

function archiveTooltipHtml(item, isRepresentative = false) {
  return `
    <strong>Pareto #${formatInteger(item.index)}</strong>${isRepresentative ? "<br />\u4ee3\u8868\u89e3" : ""}
    <br />\u51b3\u7b56\u5206\u6570: ${formatNumber(item.decisionScore, 4)}
    <br />MHC: ${formatNumber(item.mhc, 3)}
    <br />CR: ${formatNumber(item.cr, 3)}
    <br />DR: ${formatNumber(item.dr, 3)}
    <br />AR: ${formatNumber(item.ar, 3)}
  `;
}


function formatSignedDelta(delta, digits = 3) {
  if (!isNumber(delta)) {
    return "-";
  }
  const formatted = formatNumber(Math.abs(delta), digits);
  if (Math.abs(delta) < 1e-12) {
    return formatted;
  }
  return `${delta > 0 ? "+" : "-"}${formatted}`;
}

function buildComparisonCard(metric) {
  return `
    <article class="comparison-card ${metric.highlight ? "representative" : ""}">
      <span class="comparison-label">${metric.label}</span>
      <strong class="comparison-value">${metric.value}</strong>
      <div class="comparison-meta">
        <span>\u524d\u6cbf\u6700\u4f73: ${metric.best}</span>
        <span>\u524d\u6cbf\u5747\u503c: ${metric.average}</span>
        <span class="comparison-delta ${metric.deltaTone}">\u4e0e\u6700\u4f73\u5dee\u8ddd: ${metric.delta}</span>
      </div>
    </article>
  `;
}


async function renderMoParetoScatterChart(container = elements.fitnessChart) {
  if (!container) {
    return;
  }
  container.innerHTML = "";

  const row = getSelectedRow();
  if (!row) {
    container.innerHTML = `<div class="empty-state">\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684 Pareto \u6863\u6848\u3002</div>`;
    return;
  }
  if (!row.paretoArchivePath) {
    container.innerHTML = `<div class="empty-state">\u5f53\u524d\u8fd0\u884c\u6ca1\u6709 Pareto \u6863\u6848\u8def\u5f84\u3002</div>`;
    return;
  }

  container.innerHTML = `<div class="solution-layout-loading">\u6b63\u5728\u52a0\u8f7d Pareto \u524d\u6cbf...</div>`;
  try {
    const payload = await getArchivePayload(row);
    if (state.selectedRunIndex !== row.runIndex) {
      return;
    }

    const items = toArchiveItems(payload);
    if (!items.length) {
      container.innerHTML = `<div class="empty-state">\u5f53\u524d\u6863\u6848\u4e3a\u7a7a\uff0c\u65e0\u6cd5\u7ed8\u5236 Pareto \u6563\u70b9\u56fe\u3002</div>`;
      return;
    }

    container.innerHTML = "";
    const svg = createChartSvg({ width: 760, height: 320 });
    const frame = getPlotFrame(svg, { top: 18, right: 28, bottom: 46, left: 62 });
    const xDomain = paddedDomain(items.map((item) => item.mhc));
    const yDomain = paddedDomain(items.map((item) => item.cr));
    const drValues = items.map((item) => item.dr).filter(isNumber);
    const representative = findRepresentativeArchiveItem(payload, row);

    drawGrid(svg, frame, {
      xDomain,
      yDomain,
      xTicks: 5,
      yTicks: 5,
      xFormatter: (value) => formatNumber(value, 0),
      yFormatter: (value) => formatNumber(value, 0),
      xLabel: "MHC (min)",
      yLabel: "CR (max)",
    });

    items.forEach((item) => {
      if (!isNumber(item.mhc) || !isNumber(item.cr)) {
        return;
      }
      const selected = representative && item.index === representative.index;
      const cx = scaleValue(item.mhc, xDomain[0], xDomain[1], frame.left, frame.right);
      const cy = scaleValue(item.cr, yDomain[0], yDomain[1], frame.bottom, frame.top);
      const point = createSvgNode("circle", {
        cx,
        cy,
        r: selected ? 7.5 : 4.8,
        fill: interpolateSeriesColor(item.dr, Math.min(...drValues), Math.max(...drValues)),
        opacity: selected ? 0.98 : 0.78,
        stroke: selected ? "#ef6c39" : "rgba(23, 34, 45, 0.18)",
        "stroke-width": selected ? 2.8 : 1.2,
      });
      bindTooltip(point, () => archiveTooltipHtml(item, selected));
      svg.appendChild(point);
    });

    container.appendChild(svg);
    container.appendChild(createLegend([
      ["orange", "\u4ee3\u8868\u89e3\u9ad8\u4eae"],
      ["green", "\u989c\u8272\u8d8a\u6df1\u8868\u793a DR \u8d8a\u9ad8"],
    ]));
    appendChartNote(container, `\u5f53\u524d\u6863\u6848\u5171 ${formatInteger(items.length)} \u4e2a\u975e\u652f\u914d\u89e3\u3002`);
  } catch (error) {
    if (state.selectedRunIndex !== row.runIndex) {
      return;
    }
    container.innerHTML = `<div class="empty-state">${escapeHtml(error.message || "Pareto \u524d\u6cbf\u52a0\u8f7d\u5931\u8d25\u3002")}</div>`;
  }
}


async function renderMoRepresentativeComparisonChart(container = elements.timingChart) {
  if (!container) {
    return;
  }
  container.innerHTML = "";

  const row = getSelectedRow();
  if (!row || !row.paretoArchivePath) {
    container.innerHTML = `<div class="empty-state">\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u4ee3\u8868\u89e3\u5bf9\u6bd4\u4fe1\u606f\u3002</div>`;
    return;
  }

  container.innerHTML = `<div class="solution-layout-loading">\u6b63\u5728\u6bd4\u8f83\u4ee3\u8868\u89e3\u4e0e\u6863\u6848...</div>`;
  try {
    const payload = await getArchivePayload(row);
    if (state.selectedRunIndex !== row.runIndex) {
      return;
    }

    const items = toArchiveItems(payload);
    const representative = findRepresentativeArchiveItem(payload, row);
    if (!items.length || !representative) {
      container.innerHTML = `<div class="empty-state">\u5f53\u524d\u6863\u6848\u4e3a\u7a7a\uff0c\u65e0\u6cd5\u751f\u6210\u4ee3\u8868\u89e3\u5bf9\u6bd4\u5361\u3002</div>`;
      return;
    }

    const buildMetric = ({ label, key, direction = "min", digits = 3, highlight = false }) => {
      const values = items.map((item) => item[key]).filter(isNumber);
      const repValue = representative[key];
      const bestValue = direction === "min" ? Math.min(...values) : Math.max(...values);
      const averageValue = average(values);
      const rawDelta = direction === "min" ? repValue - bestValue : bestValue - repValue;
      return {
        label,
        value: formatNumber(repValue, digits),
        best: formatNumber(bestValue, digits),
        average: formatNumber(averageValue, digits),
        delta: formatSignedDelta(rawDelta, digits),
        deltaTone: Math.abs(rawDelta) < 1e-12 ? "good" : rawDelta > 0 ? "warn" : "neutral",
        highlight,
      };
    };

    const metrics = [
      buildMetric({ label: "\u51b3\u7b56\u5206\u6570", key: "decisionScore", direction: "min", digits: 4, highlight: true }),
      buildMetric({ label: "MHC", key: "mhc", direction: "min" }),
      buildMetric({ label: "CR", key: "cr", direction: "max" }),
      buildMetric({ label: "DR", key: "dr", direction: "max" }),
      buildMetric({ label: "AR", key: "ar", direction: "max" }),
      {
        label: "Pareto \u6863\u6848\u89c4\u6a21",
        value: formatInteger(items.length),
        best: formatInteger(items.length),
        average: formatInteger(items.length),
        delta: "-",
        deltaTone: "neutral",
        highlight: false,
      },
    ];

    container.innerHTML = `<div class="comparison-grid">${metrics.map(buildComparisonCard).join("")}</div>`;
    appendChartNote(container, `\u4ee3\u8868\u89e3\u5e8f\u53f7 #${formatInteger(representative.index)}\uff0c\u5176\u4f59\u7edf\u8ba1\u5747\u6765\u81ea\u5f53\u524d\u9009\u4e2d\u8fd0\u884c\u7684 Pareto \u6863\u6848\u3002`);
  } catch (error) {
    if (state.selectedRunIndex !== row.runIndex) {
      return;
    }
    container.innerHTML = `<div class="empty-state">${escapeHtml(error.message || "\u4ee3\u8868\u89e3\u5bf9\u6bd4\u52a0\u8f7d\u5931\u8d25\u3002")}</div>`;
  }
}


async function renderMoParallelCoordinatesChart(container = elements.histogramChart) {
  if (!container) {
    return;
  }
  container.innerHTML = "";

  const row = getSelectedRow();
  if (!row || !row.paretoArchivePath) {
    container.innerHTML = `<div class="empty-state">\u5f53\u524d\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u5e73\u884c\u5750\u6807\u6570\u636e\u3002</div>`;
    return;
  }

  container.innerHTML = `<div class="solution-layout-loading">\u6b63\u5728\u52a0\u8f7d\u5e73\u884c\u5750\u6807\u56fe...</div>`;
  try {
    const payload = await getArchivePayload(row);
    if (state.selectedRunIndex !== row.runIndex) {
      return;
    }

    const items = toArchiveItems(payload);
    const representative = findRepresentativeArchiveItem(payload, row);
    if (!items.length) {
      container.innerHTML = `<div class="empty-state">\u5f53\u524d\u6863\u6848\u4e3a\u7a7a\uff0c\u65e0\u6cd5\u7ed8\u5236\u5e73\u884c\u5750\u6807\u56fe\u3002</div>`;
      return;
    }

    const sampledItems = sampleArchiveItems(items, 42);
    const domains = {
      mhc: paddedDomain(items.map((item) => item.mhc)),
      cr: paddedDomain(items.map((item) => item.cr)),
      dr: paddedDomain(items.map((item) => item.dr)),
      ar: paddedDomain(items.map((item) => item.ar)),
    };
    const axes = [
      { key: "mhc", label: "MHC", invert: true },
      { key: "cr", label: "CR", invert: false },
      { key: "dr", label: "DR", invert: false },
      { key: "ar", label: "AR", invert: false },
    ];

    container.innerHTML = "";
    const svg = createChartSvg({ width: 760, height: 320 });
    const frame = getPlotFrame(svg, { top: 24, right: 28, bottom: 46, left: 60 });
    const axisXs = axes.map((_, index) => scaleValue(index, 0, axes.length - 1, frame.left, frame.right));

    generateTicks(0, 1, 5).forEach((tick) => {
      const y = scaleValue(tick, 0, 1, frame.bottom, frame.top);
      svg.appendChild(createSvgNode("line", {
        x1: frame.left,
        x2: frame.right,
        y1: y,
        y2: y,
        class: "chart-grid-line",
      }));
      svg.appendChild(createSvgNode("text", {
        x: frame.left - 12,
        y: y + 4,
        "text-anchor": "end",
        class: "tick-text",
      }, formatNumber(tick, 2)));
    });

    axes.forEach((axis, index) => {
      const x = axisXs[index];
      svg.appendChild(createSvgNode("line", {
        x1: x,
        x2: x,
        y1: frame.top,
        y2: frame.bottom,
        class: "chart-axis",
      }));
      svg.appendChild(createSvgNode("text", {
        x,
        y: frame.bottom + 24,
        "text-anchor": "middle",
        class: "chart-label",
      }, axis.label));
    });

    sampledItems.forEach((item) => {
      const selected = representative && item.index === representative.index;
      const points = axes
        .map((axis, axisIndex) => {
          const [minValue, maxValue] = domains[axis.key];
          const score = normalizeScore(item[axis.key], minValue, maxValue, axis.invert);
          const y = scaleValue(score, 0, 1, frame.bottom, frame.top);
          return `${axisXs[axisIndex]},${y}`;
        })
        .join(" ");
      const polyline = createSvgNode("polyline", {
        points,
        fill: "none",
        stroke: selected ? "#ef6c39" : "rgba(63, 110, 207, 0.28)",
        "stroke-width": selected ? 3 : 1.4,
        opacity: selected ? 0.98 : 0.8,
      });
      bindTooltip(polyline, () => archiveTooltipHtml(item, selected));
      svg.appendChild(polyline);
    });

    container.appendChild(svg);
    container.appendChild(createLegend([
      ["orange", "\u4ee3\u8868\u89e3"],
      ["blue", "\u5176\u4f59 Pareto \u89e3"],
    ]));
    appendChartNote(container, "\u7eb5\u8f74\u4e3a\u5f52\u4e00\u5316\u540e\u7684\u4f18\u5ea6\u5206\u6570\uff0c\u8d8a\u9760\u4e0a\u8868\u793a\u8be5\u76ee\u6807\u8d8a\u4f18\uff1bMHC \u5df2\u6309\u6700\u5c0f\u5316\u65b9\u5411\u53cd\u8f6c\u3002");
  } catch (error) {
    if (state.selectedRunIndex !== row.runIndex) {
      return;
    }
    container.innerHTML = `<div class="empty-state">${escapeHtml(error.message || "\u5e73\u884c\u5750\u6807\u56fe\u52a0\u8f7d\u5931\u8d25\u3002")}</div>`;
  }
}


function renderMoRunOverviewChart(container = elements.scatterChart) {
  if (!container) {
    return;
  }
  const rows = getRowsForViewMode("mo", state.filteredRows).filter((row) => isNumber(currentPrimaryMetricValue(row)));
  container.innerHTML = "";

  if (!rows.length) {
    container.innerHTML = `<div class="empty-state">\u6ca1\u6709\u8db3\u591f\u7684\u6570\u636e\u7ed8\u5236\u591a\u76ee\u6807\u8fd0\u884c\u6982\u89c8\u3002</div>`;
    return;
  }

  const bestSoFar = [];
  let currentBest = Infinity;
  rows.forEach((row) => {
    const decisionValue = currentPrimaryMetricValue(row);
    if (isNumber(decisionValue)) {
      currentBest = Math.min(currentBest, decisionValue);
    }
    bestSoFar.push({ runIndex: row.runIndex, decisionScore: Number.isFinite(currentBest) ? currentBest : null });
  });

  const svg = createChartSvg({ width: 760, height: 320 });
  const frame = getPlotFrame(svg, { top: 18, right: 28, bottom: 46, left: 62 });
  const xDomain = [rows[0].runIndex, rows[rows.length - 1].runIndex];
  const yDomain = paddedDomain(rows.map((row) => currentPrimaryMetricValue(row)));

  drawGrid(svg, frame, {
    xDomain,
    yDomain,
    xTicks: 6,
    yTicks: 5,
    xFormatter: (value) => `#${Math.round(value)}`,
    yFormatter: (value) => formatNumber(value, 3),
    xLabel: "\u8fd0\u884c\u5e8f\u53f7",
    yLabel: "\u51b3\u7b56\u5206\u6570",
  });

  const lineActual = buildLinePath(rows, frame, xDomain, yDomain, (row) => row.runIndex, (row) => currentPrimaryMetricValue(row));
  const lineBest = buildLinePath(bestSoFar, frame, xDomain, yDomain, (row) => row.runIndex, (row) => row.decisionScore);

  svg.appendChild(createSvgNode("path", { d: lineBest, class: "series-secondary" }));
  svg.appendChild(createSvgNode("path", { d: lineActual, class: "series-primary" }));

  rows.forEach((row) => {
    const value = currentPrimaryMetricValue(row);
    if (!isNumber(value)) {
      return;
    }
    const cx = scaleValue(row.runIndex, xDomain[0], xDomain[1], frame.left, frame.right);
    const cy = scaleValue(value, yDomain[0], yDomain[1], frame.bottom, frame.top);
    const radius = clamp(4 + (row.paretoSize || 0) * 0.5, 4, 12);
    const selected = row.runIndex === state.selectedRunIndex;
    const point = createSvgNode("circle", {
      cx,
      cy,
      r: selected ? radius + 1.2 : radius,
      fill: getValidityColor(row.aspectRatioValid),
      opacity: selected ? 0.98 : 0.82,
      stroke: selected ? "white" : "rgba(23, 34, 45, 0.18)",
      "stroke-width": selected ? 2.4 : 1.2,
    });
    bindTooltip(point, () => tooltipHtml(row));
    point.addEventListener("click", () => selectRow(row.runIndex));
    svg.appendChild(point);
  });

  container.appendChild(svg);
  container.appendChild(createLegend([
    ["orange", "\u5355\u6b21\u51b3\u7b56\u5206\u6570"],
    ["green", "\u7d2f\u8ba1\u6700\u4f73"],
  ]));
  appendChartNote(container, "\u5706\u70b9\u5927\u5c0f\u8868\u793a Pareto \u6863\u6848\u89c4\u6a21\uff0c\u989c\u8272\u8868\u793a\u5bbd\u9ad8\u6bd4\u662f\u5426\u6ee1\u8db3\u3002");
}


function renderFitnessChart() {
  const container = elements.fitnessChart;
  const rows = getRowsForViewMode("so", state.filteredRows);
  container.innerHTML = "";

  if (!rows.length) {
    container.innerHTML = `<div class="empty-state">没有足够的数据绘制适应度趋势。</div>`;
    return;
  }

  const bestSoFar = [];
  let currentBest = Infinity;

  rows.forEach((row) => {
    if (isNumber(row.fitness)) {
      currentBest = Math.min(currentBest, row.fitness);
    }
    bestSoFar.push({
      runIndex: row.runIndex,
      fitness: Number.isFinite(currentBest) ? currentBest : null,
    });
  });

  const svg = createChartSvg({ width: 760, height: 320 });
  const frame = getPlotFrame(svg, { top: 18, right: 28, bottom: 46, left: 58 });
  const values = rows.map((row) => row.fitness).filter(isNumber);
  const combined = [...values, ...bestSoFar.map((item) => item.fitness).filter(isNumber)];
  const yDomain = paddedDomain(combined);
  const xDomain = [rows[0].runIndex, rows[rows.length - 1].runIndex];

  drawGrid(svg, frame, {
    xDomain,
    yDomain,
    xTicks: 6,
    yTicks: 5,
    xFormatter: (value) => `#${Math.round(value)}`,
    yFormatter: (value) => formatNumber(value, 0),
    xLabel: "运行序号",
    yLabel: "适应度值",
  });

  const lineActual = buildLinePath(rows, frame, xDomain, yDomain, (row) => row.runIndex, (row) => row.fitness);
  const lineBest = buildLinePath(bestSoFar, frame, xDomain, yDomain, (row) => row.runIndex, (row) => row.fitness);

  svg.appendChild(createSvgNode("path", { d: lineBest, class: "series-secondary" }));
  svg.appendChild(createSvgNode("path", { d: lineActual, class: "series-primary" }));

  rows.forEach((row) => {
    if (!isNumber(row.fitness)) {
      return;
    }

    const cx = scaleValue(row.runIndex, xDomain[0], xDomain[1], frame.left, frame.right);
    const cy = scaleValue(row.fitness, yDomain[0], yDomain[1], frame.bottom, frame.top);
    const selected = row.runIndex === state.selectedRunIndex;
    const point = createSvgNode("circle", {
      cx,
      cy,
      r: selected ? 6.5 : 4.2,
      class: "point-primary",
      opacity: selected ? 1 : 0.82,
      stroke: selected ? "white" : "none",
      "stroke-width": selected ? 2 : 0,
    });

    bindTooltip(point, () => tooltipHtml(row));
    point.addEventListener("click", () => selectRow(row.runIndex));
    svg.appendChild(point);
  });

  container.appendChild(svg);
  container.appendChild(createLegend([
    ["orange", "单次适应度"],
    ["green", "累计最优前沿"],
  ]));
}

function renderTimingChart() {
  const container = elements.timingChart;
  const rows = getRowsForViewMode("so", state.filteredRows).filter(
    (row) => isNumber(row.runtimeSeconds) || isNumber(row.bestResultSeconds),
  );
  container.innerHTML = "";

  if (!rows.length) {
    container.innerHTML = `<div class="empty-state">没有足够的数据绘制耗时对比。</div>`;
    return;
  }

  const svg = createChartSvg({ width: 760, height: 320 });
  const frame = getPlotFrame(svg, { top: 18, right: 28, bottom: 46, left: 58 });
  const xDomain = [rows[0].runIndex, rows[rows.length - 1].runIndex];
  const yMax = Math.max(
    ...rows.map((row) => row.runtimeSeconds || 0),
    ...rows.map((row) => row.bestResultSeconds || 0),
  );
  const yDomain = [0, yMax * 1.08 || 1];

  drawGrid(svg, frame, {
    xDomain,
    yDomain,
    xTicks: 6,
    yTicks: 5,
    xFormatter: (value) => `#${Math.round(value)}`,
    yFormatter: (value) => `${formatNumber(value, 0)}s`,
    xLabel: "运行序号",
    yLabel: "秒",
  });

  const totalPath = buildLinePath(
    rows,
    frame,
    xDomain,
    yDomain,
    (row) => row.runIndex,
    (row) => row.runtimeSeconds,
  );
  const bestPath = buildLinePath(
    rows,
    frame,
    xDomain,
    yDomain,
    (row) => row.runIndex,
    (row) => row.bestResultSeconds,
  );

  svg.appendChild(createSvgNode("path", { d: totalPath, class: "series-primary" }));
  svg.appendChild(createSvgNode("path", { d: bestPath, class: "series-tertiary" }));

  rows.forEach((row) => {
    if (isNumber(row.runtimeSeconds)) {
      const point = createPoint({
        row,
        frame,
        xDomain,
        yDomain,
        x: row.runIndex,
        y: row.runtimeSeconds,
        colorClass: "point-primary",
        radius: row.runIndex === state.selectedRunIndex ? 6.5 : 4,
      });
      svg.appendChild(point);
    }

    if (isNumber(row.bestResultSeconds)) {
      const point = createPoint({
        row,
        frame,
        xDomain,
        yDomain,
        x: row.runIndex,
        y: row.bestResultSeconds,
        colorClass: "point-secondary",
        radius: row.runIndex === state.selectedRunIndex ? 6 : 3.8,
      });
      svg.appendChild(point);
    }
  });

  container.appendChild(svg);
  container.appendChild(createLegend([
    ["orange", "总运行时间"],
    ["blue", "达到当前最好解耗时"],
  ]));
}

function renderHistogramChart() {
  const container = elements.histogramChart;
  const values = getRowsForViewMode("so", state.filteredRows).map((row) => row.fitness).filter(isNumber);
  container.innerHTML = "";

  if (values.length < 2) {
    container.innerHTML = `<div class="empty-state">至少需要两条数据才能查看分布。</div>`;
    return;
  }

  const bins = buildHistogramBins(values);
  const svg = createChartSvg({ width: 760, height: 320 });
  const frame = getPlotFrame(svg, { top: 18, right: 28, bottom: 54, left: 58 });
  const yDomain = [0, Math.max(...bins.map((bin) => bin.count)) * 1.15];

  drawGrid(svg, frame, {
    xDomain: [bins[0].start, bins[bins.length - 1].end],
    yDomain,
    xTicks: 6,
    yTicks: 5,
    xFormatter: (value) => formatNumber(value, 0),
    yFormatter: (value) => formatInteger(value),
    xLabel: "适应度值区间",
    yLabel: "次数",
  });

  const chartWidth = frame.right - frame.left;
  const barGap = 6;
  const barWidth = Math.max((chartWidth - barGap * (bins.length - 1)) / bins.length, 10);

  bins.forEach((bin, index) => {
    const x = frame.left + index * (barWidth + barGap);
    const y = scaleValue(bin.count, yDomain[0], yDomain[1], frame.bottom, frame.top);
    const bar = createSvgNode("rect", {
      x,
      y,
      width: barWidth,
      height: Math.max(frame.bottom - y, 0),
      rx: 8,
      class: "hist-bar",
    });
    bindTooltip(
      bar,
      () => `<strong>${formatNumber(bin.start, 1)} - ${formatNumber(bin.end, 1)}</strong>共 ${formatInteger(bin.count)} 次`,
    );
    svg.appendChild(bar);
  });

  container.appendChild(svg);
}

function renderScatterChart() {
  const container = elements.scatterChart;
  const rows = getRowsForViewMode("so", state.filteredRows).filter(
    (row) => isNumber(row.runtimeSeconds) && isNumber(row.fitness),
  );
  container.innerHTML = "";

  if (!rows.length) {
    container.innerHTML = `<div class="empty-state">没有足够的数据绘制散点图。</div>`;
    return;
  }

  const svg = createChartSvg({ width: 760, height: 320 });
  const frame = getPlotFrame(svg, { top: 18, right: 28, bottom: 46, left: 58 });
  const xDomain = paddedDomain(rows.map((row) => row.runtimeSeconds));
  const yDomain = paddedDomain(rows.map((row) => row.fitness));

  drawGrid(svg, frame, {
    xDomain,
    yDomain,
    xTicks: 6,
    yTicks: 5,
    xFormatter: (value) => `${formatNumber(value, 0)}s`,
    yFormatter: (value) => formatNumber(value, 0),
    xLabel: "运行时间（秒）",
    yLabel: "适应度值",
  });

  rows.forEach((row) => {
    const cx = scaleValue(row.runtimeSeconds, xDomain[0], xDomain[1], frame.left, frame.right);
    const cy = scaleValue(row.fitness, yDomain[0], yDomain[1], frame.bottom, frame.top);
    const radius = clamp(4 + (row.gbestUpdates || 0) * 0.35, 4, 11);
    const fill = getValidityColor(row.aspectRatioValid);
    const selected = row.runIndex === state.selectedRunIndex;
    const point = createSvgNode("circle", {
      cx,
      cy,
      r: selected ? radius + 1.8 : radius,
      fill,
      opacity: selected ? 0.98 : 0.82,
      stroke: selected ? "white" : "rgba(23, 34, 45, 0.18)",
      "stroke-width": selected ? 2.4 : 1.2,
    });

    bindTooltip(point, () => tooltipHtml(row));
    point.addEventListener("click", () => selectRow(row.runIndex));
    svg.appendChild(point);
  });

  container.appendChild(svg);
  container.appendChild(createLegend([
    ["green", "宽高比满足"],
    ["red", "宽高比不满足"],
    ["gold", "宽高比未知"],
  ]));
}

function formatRowIdentity(row) {
  return `${escapeHtml(row.instance || "-")} / ${escapeHtml(row.algorithm || "-")}`;
}

function buildTableMetricCellHtml(label, value, meta = "") {
  const metaHtml = meta ? `<span class="table-cell-meta">${meta}</span>` : "";
  return `
    <div class="table-cell-stack">
      <span class="table-cell-label">${label}</span>
      <strong class="table-cell-value">${value}</strong>
      ${metaHtml}
    </div>
  `;
}

function buildMixedPrimaryMetricCell(row) {
  if (getRowViewMode(row) === "mo") {
    return buildTableMetricCellHtml("\u51b3\u7b56\u5206\u6570", formatNumber(currentPrimaryMetricValue(row), 4), formatRowIdentity(row));
  }
  return buildTableMetricCellHtml("\u9002\u5e94\u5ea6", formatNumber(row.fitness, 3), formatRowIdentity(row));
}

function buildMixedSecondaryMetricCell(row) {
  if (getRowViewMode(row) === "mo") {
    return buildTableMetricCellHtml(
      "Pareto \u6863\u6848",
      formatInteger(row.paretoSize),
      `\u4ee3\u8868\u89e3 MHC ${formatNumber(row.repMhc, 3)}`,
    );
  }

  return buildTableMetricCellHtml(
    "gbest \u66f4\u65b0",
    formatInteger(row.gbestUpdates),
    `\u6700\u597d\u89e3\u8017\u65f6 ${formatSeconds(row.bestResultSeconds, 1)}`,
  );
}

function renderTable() {
  ensureTableSortKey();
  const rows = sortRowsForTable(state.filteredRows);
  const columns = getTableColumns();
  const tableMode = getRowsViewMode(rows);
  elements.resultsTable.innerHTML = "";

  if (!rows.length) {
    elements.tableMeta.textContent = "\u5f53\u524d\u7b5b\u9009\u6761\u4ef6\u4e0b\u6ca1\u6709\u6570\u636e\u3002";
    elements.resultsTable.innerHTML = `<div class="empty-state">\u8c03\u6574\u7b5b\u9009\u6761\u4ef6\u540e\u518d\u8bd5\u4e00\u6b21\u3002</div>`;
    return;
  }

  if (tableMode === "mixed") {
    const soCount = getRowsForViewMode("so", rows).length;
    const moCount = getRowsForViewMode("mo", rows).length;
    elements.tableMeta.textContent = `\u663e\u793a ${formatInteger(rows.length)} / ${formatInteger(state.rows.length)} \u6761\u8bb0\u5f55 \u00b7 \u5355\u76ee\u6807 ${formatInteger(soCount)} \u6761 \u00b7 \u591a\u76ee\u6807 ${formatInteger(moCount)} \u6761`;
  } else {
    elements.tableMeta.textContent = `\u663e\u793a ${formatInteger(rows.length)} / ${formatInteger(state.rows.length)} \u6761\u8bb0\u5f55`;
  }

  const table = document.createElement("table");
  const headCells = columns
    .map((column) => {
      const isActive = state.tableSort.key === column.key;
      const arrow = isActive ? (state.tableSort.direction === "asc" ? "ASC" : "DESC") : "--";
      const activeClass = isActive ? "active" : "";
      return `
        <th>
          <button class="table-sort ${activeClass}" type="button" data-sort-key="${column.key}">
            <span>${column.label}</span>
            <span class="table-sort-arrow">${arrow}</span>
          </button>
        </th>
      `;
    })
    .join("");

  table.innerHTML = `
    <thead>
      <tr>
        ${headCells}
      </tr>
    </thead>
  `;

  table.querySelectorAll(".table-sort").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.getAttribute("data-sort-key");
      if (!key) {
        return;
      }
      toggleTableSort(key);
    });
  });

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.runIndex === state.selectedRunIndex) {
      tr.classList.add("selected");
    }

    tr.innerHTML = tableMode === "mixed"
      ? `
        <td class="mono">#${row.runIndex}</td>
        <td>${createViewModeBadgeText(getRowViewMode(row))}</td>
        <td>${escapeHtml(formatDateCell(row.date))}</td>
        <td>${buildMixedPrimaryMetricCell(row)}</td>
        <td>${formatSeconds(row.runtimeSeconds, 1)}</td>
        <td>${buildMixedSecondaryMetricCell(row)}</td>
        <td>${createBadgeText(row.aspectRatioValid)}</td>
      `
      : tableMode === "mo"
      ? `
        <td class="mono">#${row.runIndex}</td>
        <td>${escapeHtml(formatDateCell(row.date))}</td>
        <td>${formatNumber(row.decisionScore, 4)}</td>
        <td>${formatInteger(row.paretoSize)}</td>
        <td>${formatNumber(row.repMhc, 3)}</td>
        <td>${formatNumber(row.repCr, 3)}</td>
        <td>${formatNumber(row.repDr, 3)}</td>
        <td>${formatNumber(row.repAr, 3)}</td>
        <td>${createBadgeText(row.aspectRatioValid)}</td>
      `
      : `
        <td class="mono">#${row.runIndex}</td>
        <td>${escapeHtml(formatDateCell(row.date))}</td>
        <td>${formatNumber(row.fitness, 3)}</td>
        <td>${formatSeconds(row.runtimeSeconds, 1)}</td>
        <td>${formatSeconds(row.bestResultSeconds, 1)}</td>
        <td>${formatInteger(row.gbestUpdates)}</td>
        <td>${createBadgeText(row.aspectRatioValid)}</td>
      `;

    tr.addEventListener("click", () => selectRow(row.runIndex));
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  elements.resultsTable.appendChild(table);
}


function toggleTableSort(key) {
  if (state.tableSort.key === key) {
    state.tableSort.direction = state.tableSort.direction === "asc" ? "desc" : "asc";
  } else {
    state.tableSort = {
      key,
      direction: defaultTableSortDirection(key),
    };
  }
  renderTable();
}


function sortRowsForTable(rows) {
  ensureTableSortKey();
  const sorted = [...rows];
  const key = state.tableSort.key;
  const direction = state.tableSort.direction;

  sorted.sort((left, right) => {
    const result = compareTableRows(left, right, key);
    if (result !== 0) {
      return direction === "asc" ? result : -result;
    }
    return right.runIndex - left.runIndex;
  });

  return sorted;
}


function compareTableRows(left, right, key) {
  if (key === "date") {
    return compareDateValue(left.date, right.date);
  }

  if (key === "aspectRatioValid") {
    return compareValidity(left.aspectRatioValid, right.aspectRatioValid);
  }

  if (key === "viewMode") {
    return getRowViewMode(left).localeCompare(getRowViewMode(right));
  }

  if (key === "primaryMetric") {
    return compareNumericValue(currentPrimaryMetricValue(left), currentPrimaryMetricValue(right));
  }

  if (key === "secondaryMetric") {
    return compareNumericValue(currentSecondaryMetricValue(left), currentSecondaryMetricValue(right));
  }

  return compareNumericValue(left[key], right[key]);
}

function compareNumericValue(left, right) {
  const leftValid = isNumber(left);
  const rightValid = isNumber(right);

  if (leftValid && rightValid) {
    return left - right;
  }
  if (leftValid) {
    return -1;
  }
  if (rightValid) {
    return 1;
  }
  return 0;
}

function compareDateValue(left, right) {
  const leftText = String(left || "").trim();
  const rightText = String(right || "").trim();
  const leftTime = Date.parse(leftText);
  const rightTime = Date.parse(rightText);
  const leftValid = Number.isFinite(leftTime);
  const rightValid = Number.isFinite(rightTime);

  if (leftValid && rightValid) {
    return leftTime - rightTime;
  }
  if (leftValid) {
    return -1;
  }
  if (rightValid) {
    return 1;
  }
  return leftText.localeCompare(rightText);
}

function compareValidity(left, right) {
  const rank = (value) => {
    if (value === true) return 2;
    if (value === false) return 1;
    return 0;
  };
  return rank(left) - rank(right);
}

function getDetailTabsForRow(row) {
  if (!row) {
    return [];
  }

  const tabs = [{ key: "overview", label: "\u6982\u89c8" }];
  if (getRowViewMode(row) === "mo") {
    tabs.push({ key: "pareto", label: "Pareto" });
  }
  tabs.push(
    { key: "layout", label: "\u5e03\u5c40" },
    { key: "analysis", label: "\u5206\u6790" },
    { key: "notes", label: "\u5907\u6ce8" },
  );
  return tabs;
}

function ensureActiveDetailTab(row) {
  const tabs = getDetailTabsForRow(row);
  if (!tabs.length) {
    state.activeDetailTab = "overview";
    return;
  }

  if (!tabs.some((tab) => tab.key === state.activeDetailTab)) {
    state.activeDetailTab = "overview";
  }
}

function selectDetailTab(tabKey) {
  state.activeDetailTab = tabKey;
  renderDetail();
  syncDynamicDetailElements();
  renderChartsOnly();
}

function buildDetailTabButton(tab) {
  const isActive = tab.key === state.activeDetailTab;
  return `
    <button
      type="button"
      class="detail-tab-button ${isActive ? "active" : ""}"
      data-detail-tab="${tab.key}"
    >
      ${tab.label}
    </button>
  `;
}

function buildSingleObjectiveOverviewTab(row) {
  return `
    <section class="detail-block detail-mode-block">
      <div class="detail-mode-header">
        ${createViewModeBadgeText("so")}
        <p>\u5f53\u524d\u9875\u9762\u53ea\u4fdd\u7559\u5355\u76ee\u6807\u8bed\u4e49\uff1a\u6700\u4f18\u89e3\u3001\u9002\u5e94\u5ea6\u3001\u6536\u655b\u76f8\u5173\u65f6\u95f4\u548c gbest \u66f4\u65b0\u3002</p>
      </div>
    </section>

    <dl class="detail-grid-list">
      <div class="detail-item">
        <dt>\u8fd0\u884c\u7f16\u53f7</dt>
        <dd class="mono">#${row.runIndex}</dd>
      </div>
      <div class="detail-item">
        <dt>\u5b9e\u4f8b / \u7b97\u6cd5</dt>
        <dd>${formatRowIdentity(row)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u9002\u5e94\u5ea6</dt>
        <dd>${formatNumber(row.fitness, 3)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u8fed\u4ee3\u6b21\u6570</dt>
        <dd>${formatInteger(row.iterations)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u603b\u8fd0\u884c\u65f6\u95f4</dt>
        <dd>${formatSeconds(row.runtimeSeconds, 3)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u8fbe\u5230\u6700\u597d\u89e3\u65f6\u95f4</dt>
        <dd>${formatSeconds(row.bestResultSeconds, 3)}</dd>
      </div>
      <div class="detail-item">
        <dt>gbest \u66f4\u65b0\u6b21\u6570</dt>
        <dd>${formatInteger(row.gbestUpdates)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u5bbd\u9ad8\u6bd4\u662f\u5426\u6ee1\u8db3</dt>
        <dd>${createBadge(row.aspectRatioValid)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u5f00\u59cb\u65f6\u95f4</dt>
        <dd>${escapeHtml(formatDateTime(row.startTime))}</dd>
      </div>
      <div class="detail-item">
        <dt>\u7ed3\u675f\u65f6\u95f4</dt>
        <dd>${escapeHtml(formatDateTime(row.endTime))}</dd>
      </div>
    </dl>
  `;
}

function buildMultiObjectiveOverviewTab(row) {
  return `
    <section class="detail-block detail-mode-block">
      <div class="detail-mode-header">
        ${createViewModeBadgeText("mo")}
        <p>\u5f53\u524d\u9875\u9762\u53ea\u4fdd\u7559\u591a\u76ee\u6807\u8bed\u4e49\uff1a\u4ee3\u8868\u89e3\u3001Pareto \u6863\u6848\u3001\u51b3\u7b56\u5206\u6570\u548c\u8fd0\u884c\u6458\u8981\u3002</p>
      </div>
    </section>

    <dl class="detail-grid-list">
      <div class="detail-item">
        <dt>\u8fd0\u884c\u7f16\u53f7</dt>
        <dd class="mono">#${row.runIndex}</dd>
      </div>
      <div class="detail-item">
        <dt>\u5b9e\u4f8b / \u7b97\u6cd5</dt>
        <dd>${formatRowIdentity(row)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u51b3\u7b56\u5206\u6570</dt>
        <dd>${formatNumber(row.decisionScore ?? row.fitness, 4)}</dd>
      </div>
      <div class="detail-item">
        <dt>Pareto \u6863\u6848\u5927\u5c0f</dt>
        <dd>${formatInteger(row.paretoSize)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u8fed\u4ee3\u6b21\u6570</dt>
        <dd>${formatInteger(row.iterations)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u603b\u8fd0\u884c\u65f6\u95f4</dt>
        <dd>${formatSeconds(row.runtimeSeconds, 3)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u4ee3\u8868\u89e3\u66f4\u65b0\u6b21\u6570</dt>
        <dd>${formatInteger(row.gbestUpdates)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u5bbd\u9ad8\u6bd4\u662f\u5426\u6ee1\u8db3</dt>
        <dd>${createBadge(row.aspectRatioValid)}</dd>
      </div>
      <div class="detail-item">
        <dt>\u5f00\u59cb\u65f6\u95f4</dt>
        <dd>${escapeHtml(formatDateTime(row.startTime))}</dd>
      </div>
      <div class="detail-item">
        <dt>\u7ed3\u675f\u65f6\u95f4</dt>
        <dd>${escapeHtml(formatDateTime(row.endTime))}</dd>
      </div>
    </dl>

    <section class="detail-block">
      <h3>\u4ee3\u8868\u89e3</h3>
      <dl class="detail-grid-list">
        <div class="detail-item">
          <dt>\u4ee3\u8868\u89e3 MHC</dt>
          <dd>${formatNumber(row.repMhc, 3)}</dd>
        </div>
        <div class="detail-item">
          <dt>\u4ee3\u8868\u89e3 CR</dt>
          <dd>${formatNumber(row.repCr, 3)}</dd>
        </div>
        <div class="detail-item">
          <dt>\u4ee3\u8868\u89e3 DR</dt>
          <dd>${formatNumber(row.repDr, 3)}</dd>
        </div>
        <div class="detail-item">
          <dt>\u4ee3\u8868\u89e3 AR</dt>
          <dd>${formatNumber(row.repAr, 3)}</dd>
        </div>
        <div class="detail-item">
          <dt>\u6863\u6848\u8def\u5f84</dt>
          <dd class="mono">${escapeHtml(row.paretoArchivePath || "-")}</dd>
        </div>
        <div class="detail-item">
          <dt>\u6863\u6848\u89c4\u6a21</dt>
          <dd>${formatInteger(row.paretoSize)}</dd>
        </div>
      </dl>
    </section>
  `;
}

function buildLayoutTabHtml() {
  return `
    <section class="detail-block">
      <h3>\u5e03\u5c40\u53ef\u89c6\u5316</h3>
      <div id="solutionLayoutHost" class="solution-layout-host">
        <div class="solution-layout-loading">\u6b63\u5728\u52a0\u8f7d\u5e03\u5c40...</div>
      </div>
    </section>
  `;
}

function buildNotesTabHtml(row) {
  return `
    <section class="detail-block">
      <h3>\u5907\u6ce8</h3>
      <p>${escapeHtml(row.remark || "\u65e0\u5907\u6ce8")}</p>
    </section>

    <section class="detail-block">
      <h3>\u539f\u59cb\u89e3</h3>
      <pre class="mono">${escapeHtml(row.solution || "\u6ca1\u6709\u53ef\u5c55\u793a\u7684\u89e3\u6587\u672c\u3002")}</pre>
    </section>
  `;
}

function buildAnalysisTabHtml() {
  return `
    <section class="analysis-chart-grid">
      <article class="panel chart-panel compact-chart-panel compact-chart-panel-secondary">
        <div class="panel-header">
          <div>
            <h3 id="chartPanel3Title">\u5206\u6790\u56fe A</h3>
            <p id="chartPanel3Subtitle">\u56fe\u8868\u52a0\u8f7d\u4e2d...</p>
          </div>
        </div>
        <div id="histogramChart" class="chart-canvas chart-canvas-compact"></div>
      </article>

      <article class="panel chart-panel compact-chart-panel compact-chart-panel-secondary">
        <div class="panel-header">
          <div>
            <h3 id="chartPanel4Title">\u5206\u6790\u56fe B</h3>
            <p id="chartPanel4Subtitle">\u56fe\u8868\u52a0\u8f7d\u4e2d...</p>
          </div>
        </div>
        <div id="scatterChart" class="chart-canvas chart-canvas-compact"></div>
      </article>
    </section>
  `;
}

function buildParetoTabHtml() {
  return `
    <section class="detail-block">
      <h3>Pareto \u6863\u6848</h3>
      <div id="paretoArchiveHost" class="solution-layout-host">
        <div class="solution-layout-loading">\u6b63\u5728\u52a0\u8f7d Pareto \u6863\u6848...</div>
      </div>
    </section>
  `;
}

function renderDetail() {
  const row = getSelectedRow();

  if (!row) {
    elements.detailContent.innerHTML = `<div class="empty-state">\u5f53\u524d\u6ca1\u6709\u53ef\u67e5\u770b\u7684\u8fd0\u884c\u8be6\u60c5\u3002</div>`;
    return;
  }

  ensureActiveDetailTab(row);
  const tabs = getDetailTabsForRow(row);
  const mode = getRowViewMode(row);
  const panels = [
    {
      key: "overview",
      html: mode === "mo" ? buildMultiObjectiveOverviewTab(row) : buildSingleObjectiveOverviewTab(row),
    },
    ...(mode === "mo" ? [{ key: "pareto", html: buildParetoTabHtml() }] : []),
    { key: "layout", html: buildLayoutTabHtml() },
    { key: "analysis", html: buildAnalysisTabHtml() },
    { key: "notes", html: buildNotesTabHtml(row) },
  ];

  elements.detailContent.innerHTML = `
    <div class="detail-tabs-shell">
      <nav class="detail-tabs" aria-label="\u8fd0\u884c\u8be6\u60c5\u6807\u7b7e">
        ${tabs.map(buildDetailTabButton).join("")}
      </nav>
      <div class="detail-tab-panels">
        ${panels
          .map(
            (panel) => `
              <section class="detail-tab-panel ${panel.key === state.activeDetailTab ? "active" : "hidden"}" data-tab-panel="${panel.key}">
                ${panel.html}
              </section>
            `,
          )
          .join("")}
      </div>
    </div>
  `;

  elements.detailContent.querySelectorAll("[data-detail-tab]").forEach((button) => {
    button.addEventListener("click", () => selectDetailTab(button.getAttribute("data-detail-tab")));
  });

  renderSolutionLayout(row);
  if (mode === "mo") {
    renderParetoArchive(row);
  }
}

async function renderParetoArchive(row) {
  const host = document.getElementById("paretoArchiveHost");
  if (!host) {
    return;
  }

  if (!row?.paretoArchivePath) {
    host.innerHTML = `<div class="empty-state">\u5f53\u524d\u8bb0\u5f55\u6ca1\u6709 Pareto \u6863\u6848\u8def\u5f84\u3002</div>`;
    return;
  }

  host.innerHTML = `<div class="solution-layout-loading">\u6b63\u5728\u52a0\u8f7d Pareto \u6863\u6848...</div>`;
  try {
    const payload = await getArchivePayload(row);
    if (row.runIndex !== state.selectedRunIndex) {
      return;
    }
    host.innerHTML = buildParetoArchiveHtml(payload);
  } catch (error) {
    if (row.runIndex !== state.selectedRunIndex) {
      return;
    }
    host.innerHTML = `<div class="empty-state">${escapeHtml(error.message || "\u8bfb\u53d6 Pareto \u6863\u6848\u5931\u8d25\u3002")}</div>`;
  }
}



function buildParetoArchiveHtml(payload) {
  const items = Array.isArray(payload?.items) ? payload.items : [];
  if (!items.length) {
    return `<div class="empty-state">Pareto 档案为空。</div>`;
  }

  const representativeIndex = Number(payload?.representativeArchiveIndex || 0);
  const rows = items
    .map((item) => {
      const selectedClass = Number(item.index) === representativeIndex ? ' class="selected"' : "";
      return `
        <tr${selectedClass}>
          <td class="mono">#${formatInteger(item.index)}</td>
          <td>${formatNumber(item.decisionScore, 3)}</td>
          <td>${formatNumber(item.mhc, 3)}</td>
          <td>${formatNumber(item.cr, 3)}</td>
          <td>${formatNumber(item.dr, 3)}</td>
          <td>${formatNumber(item.ar, 3)}</td>
          <td>${formatInteger(item.dInf)}</td>
        </tr>
      `;
    })
    .join("");

  return `
    <div class="detail-item detail-inline-note">
      <dt>档案路径</dt>
      <dd class="mono">${escapeHtml(payload.path || "-")}</dd>
    </div>
    <div class="table-host pareto-table-host"><table>
      <thead>
        <tr>
          <th>序号</th>
          <th>决策分数</th>
          <th>MHC</th>
          <th>CR</th>
          <th>DR</th>
          <th>AR</th>
          <th>dInf</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table></div>
  `;
}

function selectRow(runIndex) {
  state.selectedRunIndex = runIndex;
  render();
}
function createChartSvg({ width, height }) {
  const svg = createSvgNode("svg", {
    class: "chart-svg",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-hidden": "true",
  });
  svg.appendChild(
    createSvgNode("rect", {
      x: 0,
      y: 0,
      width,
      height,
      rx: 24,
      class: "chart-frame",
    }),
  );
  return svg;
}

function getPlotFrame(svg, padding) {
  const [, , width, height] = svg.getAttribute("viewBox").split(" ").map(Number);
  return {
    left: padding.left,
    right: width - padding.right,
    top: padding.top,
    bottom: height - padding.bottom,
  };
}

function drawGrid(
  svg,
  frame,
  { xDomain, yDomain, xTicks, yTicks, xFormatter, yFormatter, xLabel, yLabel },
) {
  const xValues = generateTicks(xDomain[0], xDomain[1], xTicks);
  const yValues = generateTicks(yDomain[0], yDomain[1], yTicks);

  xValues.forEach((value) => {
    const x = scaleValue(value, xDomain[0], xDomain[1], frame.left, frame.right);
    svg.appendChild(
      createSvgNode("line", {
        x1: x,
        x2: x,
        y1: frame.top,
        y2: frame.bottom,
        class: "chart-grid-line",
      }),
    );
    svg.appendChild(
      createSvgNode(
        "text",
        {
          x,
          y: frame.bottom + 22,
          "text-anchor": "middle",
          class: "tick-text",
        },
        xFormatter(value),
      ),
    );
  });

  yValues.forEach((value) => {
    const y = scaleValue(value, yDomain[0], yDomain[1], frame.bottom, frame.top);
    svg.appendChild(
      createSvgNode("line", {
        x1: frame.left,
        x2: frame.right,
        y1: y,
        y2: y,
        class: "chart-grid-line",
      }),
    );
    svg.appendChild(
      createSvgNode(
        "text",
        {
          x: frame.left - 10,
          y: y + 4,
          "text-anchor": "end",
          class: "tick-text",
        },
        yFormatter(value),
      ),
    );
  });

  svg.appendChild(
    createSvgNode("line", {
      x1: frame.left,
      x2: frame.right,
      y1: frame.bottom,
      y2: frame.bottom,
      class: "chart-axis",
    }),
  );

  svg.appendChild(
    createSvgNode("line", {
      x1: frame.left,
      x2: frame.left,
      y1: frame.top,
      y2: frame.bottom,
      class: "chart-axis",
    }),
  );

  svg.appendChild(
    createSvgNode(
      "text",
      {
        x: (frame.left + frame.right) / 2,
        y: frame.bottom + 40,
        "text-anchor": "middle",
        class: "chart-label",
      },
      xLabel,
    ),
  );

  svg.appendChild(
    createSvgNode(
      "text",
      {
        x: 20,
        y: (frame.top + frame.bottom) / 2,
        "text-anchor": "middle",
        class: "chart-label",
        transform: `rotate(-90 20 ${(frame.top + frame.bottom) / 2})`,
      },
      yLabel,
    ),
  );
}

function buildLinePath(rows, frame, xDomain, yDomain, getX, getY) {
  const commands = [];

  rows.forEach((row) => {
    const xValue = getX(row);
    const yValue = getY(row);
    if (!isNumber(xValue) || !isNumber(yValue)) {
      return;
    }
    const x = scaleValue(xValue, xDomain[0], xDomain[1], frame.left, frame.right);
    const y = scaleValue(yValue, yDomain[0], yDomain[1], frame.bottom, frame.top);
    commands.push(`${commands.length ? "L" : "M"} ${x} ${y}`);
  });

  return commands.join(" ");
}

function createPoint({ row, frame, xDomain, yDomain, x, y, colorClass, radius }) {
  const cx = scaleValue(x, xDomain[0], xDomain[1], frame.left, frame.right);
  const cy = scaleValue(y, yDomain[0], yDomain[1], frame.bottom, frame.top);
  const selected = row.runIndex === state.selectedRunIndex;
  const point = createSvgNode("circle", {
    cx,
    cy,
    r: radius,
    class: colorClass,
    opacity: selected ? 1 : 0.82,
    stroke: selected ? "white" : "none",
    "stroke-width": selected ? 2 : 0,
  });

  bindTooltip(point, () => tooltipHtml(row));
  point.addEventListener("click", () => selectRow(row.runIndex));
  return point;
}

function buildHistogramBins(values) {
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const count = clamp(Math.round(Math.sqrt(values.length)), 6, 12);
  const width = maxValue === minValue ? 1 : (maxValue - minValue) / count;
  const bins = Array.from({ length: count }, (_, index) => ({
    start: minValue + index * width,
    end: index === count - 1 ? maxValue : minValue + (index + 1) * width,
    count: 0,
  }));

  values.forEach((value) => {
    const ratio = width === 0 ? 0 : (value - minValue) / width;
    const index = clamp(Math.floor(ratio), 0, bins.length - 1);
    bins[index].count += 1;
  });

  return bins;
}

function createLegend(items) {
  const wrapper = document.createElement("div");
  wrapper.className = "legend-row";
  items.forEach(([tone, label]) => {
    const chip = document.createElement("span");
    chip.className = `legend-chip ${tone}`;
    chip.textContent = label;
    wrapper.appendChild(chip);
  });
  return wrapper;
}

function paddedDomain(values) {
  const numeric = values.filter(isNumber);
  const minValue = Math.min(...numeric);
  const maxValue = Math.max(...numeric);

  if (minValue === maxValue) {
    const offset = minValue === 0 ? 1 : Math.abs(minValue) * 0.1;
    return [minValue - offset, maxValue + offset];
  }

  const padding = (maxValue - minValue) * 0.08;
  return [minValue - padding, maxValue + padding];
}

function scaleValue(value, domainStart, domainEnd, rangeStart, rangeEnd) {
  if (domainStart === domainEnd) {
    return (rangeStart + rangeEnd) / 2;
  }
  const ratio = (value - domainStart) / (domainEnd - domainStart);
  return rangeStart + ratio * (rangeEnd - rangeStart);
}

function generateTicks(minValue, maxValue, count) {
  if (count <= 1 || minValue === maxValue) {
    return [minValue];
  }

  const step = (maxValue - minValue) / (count - 1);
  return Array.from({ length: count }, (_, index) => minValue + step * index);
}

function createSvgNode(tagName, attributes, textContent = "") {
  const node = document.createElementNS(SVG_NS, tagName);
  Object.entries(attributes || {}).forEach(([key, value]) => {
    node.setAttribute(key, String(value));
  });
  if (textContent) {
    node.textContent = textContent;
  }
  return node;
}

function bindTooltip(target, htmlBuilder) {
  target.addEventListener("mouseenter", (event) => {
    elements.tooltip.innerHTML = htmlBuilder();
    elements.tooltip.classList.remove("hidden");
    moveTooltip(event);
  });

  target.addEventListener("mousemove", moveTooltip);
  target.addEventListener("mouseleave", hideTooltip);
}

function moveTooltip(event) {
  elements.tooltip.style.left = `${event.pageX + 14}px`;
  elements.tooltip.style.top = `${event.pageY + 14}px`;
}

function hideTooltip() {
  elements.tooltip.classList.add("hidden");
}

function tooltipHtml(row) {
  if (getRowViewMode(row) === "mo") {
    return `
      <strong>\u8fd0\u884c #${row.runIndex}</strong>
      \u51b3\u7b56\u5206\u6570: ${formatNumber(row.decisionScore, 4)}<br />
      Pareto \u6863\u6848: ${formatInteger(row.paretoSize)}<br />
      \u4ee3\u8868\u89e3 MHC: ${formatNumber(row.repMhc, 3)}<br />
      \u4ee3\u8868\u89e3 CR: ${formatNumber(row.repCr, 3)}<br />
      \u4ee3\u8868\u89e3 DR: ${formatNumber(row.repDr, 3)}
    `;
  }

  return `
    <strong>\u8fd0\u884c #${row.runIndex}</strong>
    \u9002\u5e94\u5ea6: ${formatNumber(row.fitness, 3)}<br />
    \u8fd0\u884c\u65f6\u95f4: ${formatSeconds(row.runtimeSeconds, 1)}<br />
    \u6700\u4f73\u89e3\u8017\u65f6: ${formatSeconds(row.bestResultSeconds, 1)}<br />
    gbest \u66f4\u65b0: ${formatInteger(row.gbestUpdates)}
  `;
}


async function renderSolutionLayout(row) {
  const host = elements.detailContent.querySelector("#solutionLayoutHost");
  if (!host) {
    return;
  }

  const runIndex = row.runIndex;
  const csvPath = state.selectedCsv;

  host.innerHTML = `<div class="solution-layout-loading">正在按 render 规则生成布局...</div>`;

  try {
    const layout = await requestJson(
      `/api/layout?csv=${encodeURIComponent(csvPath)}&runIndex=${runIndex}`,
    );

    if (state.selectedRunIndex !== runIndex) {
      return;
    }

    host.innerHTML = buildLayoutSvg(layout);
  } catch (error) {
    if (state.selectedRunIndex !== runIndex) {
      return;
    }

    host.innerHTML = `<div class="solution-layout-error">${escapeHtml(
      error.message || "布局生成失败。",
    )}</div>`;
  }
}

function buildLayoutSvg(layout) {
  const rectangles = Array.isArray(layout.rectangles) ? layout.rectangles : [];
  if (!rectangles.length) {
    return `<div class="solution-layout-error">\u5f53\u524d\u89e3\u6ca1\u6709\u53ef\u7ed8\u5236\u7684\u8bbe\u65bd\u5e03\u5c40\u3002</div>`;
  }

  const width = Number(layout.layoutWidth);
  const height = Number(layout.layoutHeight);
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    return `<div class="solution-layout-error">\u5e03\u5c40\u5c3a\u5bf8\u65e0\u6548\uff0c\u65e0\u6cd5\u7ed8\u5236\u3002</div>`;
  }

  const uid = `layout-${Math.random().toString(36).slice(2, 9)}`;
  const floorGradientId = `${uid}-floor`;
  const gridPatternId = `${uid}-grid`;
  const strokeWidth = Math.max(Math.min(width, height) * 0.0032, 0.35);

  const ordered = [...rectangles];

  const rectNodes = ordered
    .map(
      (rect) => `
        <rect
          x="${rect.x}"
          y="${rect.y}"
          width="${rect.width}"
          height="${rect.height}"
          fill="${rect.fillColor}"
          stroke="${rect.edgeColor}"
          stroke-width="${strokeWidth}"
        ></rect>
      `,
    )
    .join("");

  const textNodes = ordered
    .map((rect) => {
      const textX = Number(rect.x) + Number(rect.width) / 2;
      const textY = height - (Number(rect.y) + Number(rect.height) / 2);
      const dynamicSize = Math.min(
        Math.max(Math.min(Number(rect.width), Number(rect.height)) * 0.32, 2.4),
        12,
      );
      return `
        <text
          x="${textX}"
          y="${textY}"
          text-anchor="middle"
          dominant-baseline="middle"
          font-size="${dynamicSize}"
          font-weight="800"
          fill="${rect.textColor}"
          class="layout-label"
        >${rect.label}</text>
      `;
    })
    .join("");

  return `
    <div class="solution-layout-meta">
      <span class="solution-layout-pill">\u5b9e\u4f8b: ${escapeHtml(layout.instance || "-")}</span>
      <span class="solution-layout-pill">Cost: ${formatLayoutValue(layout.cost, 2)}</span>
      <span class="solution-layout-pill">MHC: ${formatLayoutValue(layout.mhc, 2)}</span>
      <span class="solution-layout-pill">d_inf: ${formatLayoutValue(layout.dInf, 0)}</span>
      <span class="solution-layout-pill">\u53ef\u884c\u6027: ${layout.isFeasible ? "\u53ef\u884c" : "\u4e0d\u53ef\u884c"}</span>
    </div>
    <div class="solution-layout-canvas">
      <svg
        class="solution-layout-svg"
        viewBox="0 0 ${width} ${height}"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Facility layout"
      >
        <defs>
          <linearGradient id="${floorGradientId}" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#fffdf8"></stop>
            <stop offset="100%" stop-color="#f2eee5"></stop>
          </linearGradient>
          <pattern id="${gridPatternId}" width="6" height="6" patternUnits="userSpaceOnUse">
            <path d="M 6 0 L 0 0 0 6" fill="none" stroke="rgba(23,34,45,0.08)" stroke-width="0.18"></path>
          </pattern>
        </defs>

        <rect x="0" y="0" width="${width}" height="${height}" fill="url(#${floorGradientId})"></rect>
        <rect x="0" y="0" width="${width}" height="${height}" fill="url(#${gridPatternId})" opacity="0.35"></rect>
        <rect x="0" y="0" width="${width}" height="${height}" class="layout-border"></rect>

        <g transform="translate(0 ${height}) scale(1 -1)">
          ${rectNodes}
        </g>
        ${textNodes}
      </svg>
    </div>
  `;
}

function formatLayoutValue(value, digits = 2) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: 0,
  }).format(numeric);
}

function showStatus(message, isError) {
  elements.statusBanner.textContent = message;
  elements.statusBanner.classList.remove("hidden");
  elements.statusBanner.style.color = isError ? "var(--danger)" : "var(--ink)";
  elements.statusBanner.style.background = isError
    ? "rgba(255, 236, 231, 0.88)"
    : "rgba(255, 248, 231, 0.88)";
}

function hideStatus() {
  elements.statusBanner.classList.add("hidden");
}

function getValidityColor(value) {
  if (value === true) return "#0e8578";
  if (value === false) return "#c45249";
  return "#d4a019";
}

function createBadgeText(value) {
  if (value === true) return "满足";
  if (value === false) return "不满足";
  return "未知";
}

function createBadge(value) {
  if (value === true) return `<span class="badge valid">满足</span>`;
  if (value === false) return `<span class="badge invalid">不满足</span>`;
  return `<span class="badge unknown">未知</span>`;
}

function formatSeconds(value, fractionDigits = 1) {
  if (!isNumber(value)) return "-";
  return `${formatNumber(value, fractionDigits)}s`;
}

function formatDateCell(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
  }).format(date);
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(date);
}

function formatNumber(value, fractionDigits = 2) {
  if (!isNumber(value)) return "-";
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: fractionDigits,
    minimumFractionDigits: 0,
  }).format(value);
}

function formatInteger(value) {
  if (!isNumber(value)) return "-";
  return new Intl.NumberFormat("zh-CN", {
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value) {
  if (!isNumber(value)) return "-";
  return new Intl.NumberFormat("zh-CN", {
    style: "percent",
    maximumFractionDigits: 1,
  }).format(value);
}

function average(values) {
  const numeric = values.filter(isNumber);
  if (!numeric.length) return null;
  return numeric.reduce((sum, value) => sum + value, 0) / numeric.length;
}

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function clamp(value, minValue, maxValue) {
  return Math.min(Math.max(value, minValue), maxValue);
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function debounce(fn, delay) {
  let timeoutId = null;
  return (...args) => {
    window.clearTimeout(timeoutId);
    timeoutId = window.setTimeout(() => fn(...args), delay);
  };
}





async function loadInstances() {
  try {
    const payload = await requestJson("/api/instances");
    state.instances = Array.isArray(payload.instances) ? payload.instances : [];
    renderInstanceOptions();
    syncGeneratorDefaultInstance();
  } catch (error) {
    state.instances = [];
    renderInstanceOptions();
  }
}

function renderInstanceOptions() {
  if (!elements.instanceSelect) {
    return;
  }

  const currentValue = (elements.instanceSelect.value || "").trim();
  elements.instanceSelect.innerHTML = "";

  state.instances.forEach((instance) => {
    const option = document.createElement("option");
    option.value = instance;
    option.textContent = instance;
    elements.instanceSelect.appendChild(option);
  });

  if (currentValue && state.instances.includes(currentValue)) {
    elements.instanceSelect.value = currentValue;
  }
}

function syncGeneratorDefaultInstance() {
  if (!elements.instanceSelect) {
    return;
  }

  const current = (elements.instanceSelect.value || "").trim();
  if (current) {
    return;
  }

  const selectedRow = getSelectedRow();
  const candidate = (selectedRow?.instance || state.rows[0]?.instance || state.instances[0] || "").trim();
  if (!candidate) {
    return;
  }

  if (!state.instances.includes(candidate)) {
    const option = document.createElement("option");
    option.value = candidate;
    option.textContent = `${candidate} (CSV)`;
    elements.instanceSelect.appendChild(option);
  }

  elements.instanceSelect.value = candidate;
}

function getSelectedRow() {
  if (state.selectedRunIndex === null) {
    return state.filteredRows[0] || state.rows[0] || null;
  }

  return (
    state.filteredRows.find((item) => item.runIndex === state.selectedRunIndex) ||
    state.rows.find((item) => item.runIndex === state.selectedRunIndex) ||
    null
  );
}

function getSelectedInstance() {
  return (elements.instanceSelect?.value || "").trim();
}

function setGeneratorBusy(isBusy) {
  [
    elements.generateFromRunButton,
    elements.generateFromDocButton,
    elements.generateManualButton,
    elements.downloadSvgButton,
    elements.downloadPngButton,
  ].forEach((button) => {
    if (button) {
      button.disabled = isBusy;
    }
  });
}

async function generateLayoutFromRun() {
  const row = getSelectedRow();
  if (!row) {
    showStatus("当前没有可用的运行记录。", true);
    return;
  }

  await generateLayoutByPayload(
    {
      csv: state.selectedCsv,
      runIndex: row.runIndex,
    },
    "正在按当前运行生成布局...",
  );
}

async function generateLayoutFromDocument() {
  const instance = getSelectedInstance();
  const docPath = (elements.docPathInput?.value || "").trim();
  const extractIndexValue = Number.parseInt(elements.extractIndexInput?.value || "0", 10);
  const extractIndex = Number.isFinite(extractIndexValue) && extractIndexValue >= 0 ? extractIndexValue : 0;

  if (!instance) {
    showStatus("请先选择实例。", true);
    return;
  }
  if (!docPath) {
    showStatus("请填写文档路径。", true);
    return;
  }

  await generateLayoutByPayload(
    {
      instance,
      docPath,
      extractIndex,
    },
    "正在从文档提取解并生成布局...",
  );
}

async function generateLayoutFromManual() {
  const instance = getSelectedInstance();
  const solution = (elements.manualSolutionInput?.value || "").trim();

  if (!instance) {
    showStatus("请先选择实例。", true);
    return;
  }
  if (!solution) {
    showStatus("请先输入解。", true);
    return;
  }

  await generateLayoutByPayload(
    {
      instance,
      solution,
    },
    "正在根据手动输入生成布局...",
  );
}

async function generateLayoutByPayload(payload, loadingMessage) {
  if (!elements.generatorLayoutHost) {
    return;
  }

  setGeneratorBusy(true);
  elements.generatorLayoutHost.innerHTML = `<div class="solution-layout-loading">${escapeHtml(loadingMessage)}</div>`;

  try {
    let layout = null;
    const hasRunIndex = Number.isInteger(payload?.runIndex) && payload.runIndex > 0;

    if (hasRunIndex) {
      const csv = payload?.csv || state.selectedCsv || "";
      const query = `/api/layout?csv=${encodeURIComponent(csv)}&runIndex=${payload.runIndex}`;
      layout = await requestJson(query);
    } else {
      layout = await requestJson("/api/layout", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
    }

    state.generatorLayout = layout;
    renderGeneratorMeta(layout);
    elements.generatorLayoutHost.innerHTML = buildLayoutSvg(layout);
    hideStatus();
  } catch (error) {
    state.generatorLayout = null;
    renderGeneratorMeta(null);

    const rawMessage = error?.message || "Layout generation failed.";
    const message = rawMessage.includes("501")
      ? "Current service does not support POST /api/layout. Restart dashboard server: py -3.11 -m src.dashboard.server --port 8765"
      : rawMessage;

    elements.generatorLayoutHost.innerHTML = `<div class="solution-layout-error">${escapeHtml(message)}</div>`;
    showStatus(message, true);
  } finally {
    setGeneratorBusy(false);
  }
}


function renderGeneratorMeta(layout) {
  if (!elements.generatorMeta) {
    return;
  }

  if (!layout) {
    elements.generatorMeta.innerHTML = "";
    return;
  }

  const sourceMode = layout.source?.mode || "unknown";
  const sourceLabel =
    sourceMode === "csv"
      ? "CSV运行"
      : sourceMode === "document"
      ? "文档提取"
      : sourceMode === "manual"
      ? "手动输入"
      : "未知";

  const pills = [
    `来源: ${sourceLabel}`,
    `实例: ${escapeHtml(layout.instance || "-")}`,
    `可行性: ${layout.isFeasible ? "可行" : "不可行"}`,
    `Cost: ${formatLayoutValue(layout.cost, 2)}`,
    `MHC: ${formatLayoutValue(layout.mhc, 2)}`,
    `d_inf: ${formatLayoutValue(layout.dInf, 0)}`,
  ];

  if (sourceMode === "document") {
    pills.push(`候选解数: ${formatLayoutValue(layout.source?.candidateCount, 0)}`);
    pills.push(`提取索引: ${formatLayoutValue(layout.source?.extractIndex, 0)}`);
  }

  elements.generatorMeta.innerHTML = pills
    .map((pill) => `<span class="solution-layout-pill">${pill}</span>`)
    .join("");
}

function getLayoutDownloadBaseName() {
  const layout = state.generatorLayout || {};
  const instance = (layout.instance || getSelectedInstance() || "layout").replace(/[^a-zA-Z0-9_-]+/g, "_");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-").replace("T", "_").slice(0, 19);
  return `${instance}_${timestamp}`;
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function downloadCurrentSvg() {
  const svg = elements.generatorLayoutHost?.querySelector("svg.solution-layout-svg");
  if (!svg) {
    showStatus("请先生成布局图再导出。", true);
    return;
  }

  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(svg);
  if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {
    source = source.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
  }

  const blob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  downloadBlob(blob, `${getLayoutDownloadBaseName()}.svg`);
}

async function downloadCurrentPng() {
  const svg = elements.generatorLayoutHost?.querySelector("svg.solution-layout-svg");
  if (!svg) {
    showStatus("请先生成布局图再导出。", true);
    return;
  }

  const serializer = new XMLSerializer();
  let source = serializer.serializeToString(svg);
  if (!source.includes('xmlns="http://www.w3.org/2000/svg"')) {
    source = source.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
  }

  const svgBlob = new Blob([source], { type: "image/svg+xml;charset=utf-8" });
  const svgUrl = URL.createObjectURL(svgBlob);

  const viewBox = (svg.getAttribute("viewBox") || "0 0 1000 1000").split(" ").map(Number);
  const width = Number.isFinite(viewBox[2]) && viewBox[2] > 0 ? viewBox[2] : 1000;
  const height = Number.isFinite(viewBox[3]) && viewBox[3] > 0 ? viewBox[3] : 1000;

  try {
    const image = await new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("SVG 转 PNG 失败。"));
      img.src = svgUrl;
    });

    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("浏览器不支持 Canvas 导出。");
    }

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob((result) => {
        if (!result) {
          reject(new Error("PNG 编码失败。"));
          return;
        }
        resolve(result);
      }, "image/png");
    });

    downloadBlob(blob, `${getLayoutDownloadBaseName()}.png`);
  } catch (error) {
    showStatus(error.message || "PNG 导出失败。", true);
  } finally {
    URL.revokeObjectURL(svgUrl);
  }
}
