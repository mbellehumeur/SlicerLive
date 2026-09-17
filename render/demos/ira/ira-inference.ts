// Medical Inference dialog for IRA — probe Cast resource servers and run one.
import {
  CAST_INFERENCE_SERVERS,
  extractImagingStudyFiles,
  probeCastInferenceServers,
  renderInferenceServerInfoCardHtml,
  type CastInferenceServerDef,
  type CastProductStatusProbe,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import { idcS3, s3ListKeys } from "../../vendor/idc_tools/index.js";
import {
  getIraCastClient,
  getIraCastSessionInfo,
  getIraImagingStudyOpenContext,
  publishInferenceJob,
  setIraStatusUpdateHandler,
} from "./ira-cast.ts";
import {
  TOTAL_SEG_GROUP_LABELS,
  defaultTaskId,
  getTotalSegTask,
  groupsForModality,
  searchClasses,
  taskOptionLabel,
  taskSupportsFast,
  tasksInGroup,
  type TotalSegModality,
  type TotalSegTask,
} from "./totalseg-tasks.ts";
import { resolveOpenStudyTotalSegModality } from "./totalseg-modality.ts";

export type InferenceServerDef = CastInferenceServerDef;
export type InferenceProbeResult = CastProductStatusProbe;
export const INFERENCE_SERVERS: InferenceServerDef[] = [...CAST_INFERENCE_SERVERS];

const MAX_INFERENCE_DICOM_FILES = 300;

type RowState = {
  def: InferenceServerDef;
  probe: InferenceProbeResult | null;
  probing: boolean;
};

type VolumeFile = { url: string; fileName: string };

let selectedId: string | null = null;
let rows: RowState[] = [];
let installed = false;

/** Remember last floating-window position across open/close. */
let floatPos: { left: number; top: number } | null = null;

function $(id: string): HTMLElement | null {
  return document.getElementById(id);
}

function clampFloatPos(
  left: number,
  top: number,
  panel: HTMLElement,
): { left: number; top: number } {
  const margin = 8;
  const w = panel.offsetWidth || 480;
  const h = panel.offsetHeight || 320;
  const maxL = Math.max(margin, window.innerWidth - w - margin);
  const maxT = Math.max(margin, window.innerHeight - h - margin);
  return {
    left: Math.min(maxL, Math.max(margin, left)),
    top: Math.min(maxT, Math.max(margin, top)),
  };
}

function applyFloatPos(host: HTMLElement, panel: HTMLElement): void {
  if (!floatPos) return;
  const next = clampFloatPos(floatPos.left, floatPos.top, panel);
  floatPos = next;
  host.style.left = `${next.left}px`;
  host.style.top = `${next.top}px`;
  host.style.right = "auto";
}

function installInferenceWindowDrag(): void {
  const host = $("iraInferenceModal");
  const handle = host?.querySelector(
    "[data-ira-inference-drag]",
  ) as HTMLElement | null;
  const panel = host?.querySelector(".ira-modal-inference") as HTMLElement | null;
  if (!host || !handle || !panel || handle.dataset.iraDragBound === "1") return;
  handle.dataset.iraDragBound = "1";

  let dragging = false;
  let startX = 0;
  let startY = 0;
  let originLeft = 0;
  let originTop = 0;

  const onMove = (ev: PointerEvent) => {
    if (!dragging) return;
    const next = clampFloatPos(
      originLeft + (ev.clientX - startX),
      originTop + (ev.clientY - startY),
      panel,
    );
    floatPos = next;
    host.style.left = `${next.left}px`;
    host.style.top = `${next.top}px`;
    host.style.right = "auto";
  };

  const onUp = (ev: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("ira-dragging");
    try {
      handle.releasePointerCapture(ev.pointerId);
    } catch {
      /* ignore */
    }
    window.removeEventListener("pointermove", onMove);
    window.removeEventListener("pointerup", onUp);
    window.removeEventListener("pointercancel", onUp);
  };

  handle.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    const t = ev.target as HTMLElement | null;
    if (t?.closest("button, a, input, select, textarea, label")) return;
    dragging = true;
    handle.classList.add("ira-dragging");
    startX = ev.clientX;
    startY = ev.clientY;
    const rect = host.getBoundingClientRect();
    originLeft = rect.left;
    originTop = rect.top;
    host.style.left = `${originLeft}px`;
    host.style.top = `${originTop}px`;
    host.style.right = "auto";
    try {
      handle.setPointerCapture(ev.pointerId);
    } catch {
      /* ignore */
    }
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    ev.preventDefault();
  });
}

function setInferenceDialogMinimized(minimized: boolean): void {
  const panel = $("iraInferenceModal")?.querySelector(
    ".ira-modal-inference",
  ) as HTMLElement | null;
  const btn = $("iraInferenceMinimizeBtn") as HTMLButtonElement | null;
  if (!panel) return;
  panel.classList.toggle("is-minimized", minimized);
  if (minimized) {
    panel.style.height = "";
    panel.style.width = "";
  }
  if (btn) {
    btn.setAttribute("aria-expanded", minimized ? "false" : "true");
    btn.setAttribute("aria-label", minimized ? "Restore" : "Minimize");
    btn.title = minimized ? "Restore" : "Minimize";
    btn.textContent = minimized ? "▢" : "−";
  }
}

function toggleInferenceDialogMinimized(): void {
  const panel = $("iraInferenceModal")?.querySelector(
    ".ira-modal-inference",
  ) as HTMLElement | null;
  if (!panel) return;
  setInferenceDialogMinimized(!panel.classList.contains("is-minimized"));
}

function volumeFilesFromOpenStudy(): VolumeFile[] {
  const ctx = getIraImagingStudyOpenContext();
  const files = extractImagingStudyFiles(ctx) as Array<Record<string, unknown>>;
  const out: VolumeFile[] = [];
  for (const f of files) {
    const role = String(f.role || "").toLowerCase();
    if (role === "seg") continue;
    const url = String(f.url || "").trim();
    if (!url) continue;
    const fileName = String(f.fileName || f.name || "volume").trim() || "volume";
    out.push({ url, fileName });
  }
  return out;
}

function looksLikeNiftiVolumeFile(file: VolumeFile): boolean {
  return (
    /\.nii(\.gz)?$/i.test(file.fileName) ||
    /\.nii(\.gz)?(\?|$)/i.test(file.url)
  );
}

/** Prefer nifti-send when open study is NIfTI; else the server catalog event. */
function hubEventForInferenceJob(
  catalogEvent: string,
  files: VolumeFile[]
): "nifti-send" | "dicom-send" {
  if (files.some(looksLikeNiftiVolumeFile)) return "nifti-send";
  return catalogEvent === "nifti-send" ? "nifti-send" : "dicom-send";
}

/** Parse `https://{bucket}.s3….amazonaws.com/{prefix}/` into bucket + CRDC prefix. */
function parseIdcPrefixUrl(
  urlStr: string
): { bucket: string; prefix: string } | null {
  try {
    const u = new URL(urlStr);
    const host = u.hostname.toLowerCase();
    const m = host.match(/^([^.]+)\.s3[.-]/);
    if (!m) return null;
    const path = u.pathname.replace(/^\/+/, "").replace(/\/+$/, "");
    // Prefix-only: single path segment (CRDC uuid), no object leaf.
    if (!path || path.includes("/")) return null;
    if (/\.(dcm|nii|gz|zip)$/i.test(path)) return null;
    return { bucket: m[1], prefix: path };
  } catch {
    return null;
  }
}

function looksLikePrefixEntry(file: VolumeFile): boolean {
  if (file.url.endsWith("/") || file.fileName.endsWith("/")) return true;
  return Boolean(parseIdcPrefixUrl(file.url));
}

/**
 * Expand IDC series prefix URLs to per-instance .dcm file entries for dicom-send.
 * Non-prefix URLs (NIfTI, explicit .dcm) pass through unchanged.
 */
async function expandVolumeFilesForSend(
  files: VolumeFile[]
): Promise<VolumeFile[]> {
  const out: VolumeFile[] = [];
  for (const file of files) {
    if (!looksLikePrefixEntry(file)) {
      out.push(file);
      continue;
    }
    const parsed = parseIdcPrefixUrl(file.url.replace(/\/?$/, "/"));
    const bucket =
      parsed?.bucket ||
      (() => {
        try {
          const host = new URL(file.url).hostname.toLowerCase();
          return host.match(/^([^.]+)\.s3[.-]/)?.[1] || "idc-open-data";
        } catch {
          return "idc-open-data";
        }
      })();
    const prefix =
      parsed?.prefix ||
      file.fileName.replace(/\/+$/, "") ||
      (() => {
        try {
          return new URL(file.url).pathname.replace(/^\/+|\/+$/g, "").split("/")[0];
        } catch {
          return "";
        }
      })();
    if (!prefix) {
      throw new Error(`Cannot expand IDC prefix URL: ${file.url}`);
    }
    const keys = await s3ListKeys(prefix.endsWith("/") ? prefix : `${prefix}/`, bucket);
    if (!keys.length) {
      throw new Error(`No DICOM objects under S3 prefix ${bucket}/${prefix}/`);
    }
    if (keys.length > MAX_INFERENCE_DICOM_FILES) {
      throw new Error(
        `Series has ${keys.length} DICOM files (max ${MAX_INFERENCE_DICOM_FILES})`
      );
    }
    const base = idcS3(bucket);
    for (const key of keys) {
      const fileName = key.split("/").pop() || key;
      out.push({ url: `${base}${key}`, fileName });
    }
  }
  return out;
}

function availabilityLabel(probe: InferenceProbeResult | null, probing: boolean): string {
  if (probing) return "Checking…";
  if (!probe) return "Unknown";
  if (probe.error) return "Error";
  if (probe.online) {
    const job = probe.job ? ` · job ${probe.job}` : "";
    return `Online${job}`;
  }
  return "Offline";
}

/** Online first, then probing/unknown, then offline — catalog order within each group. */
function orderedRows(): RowState[] {
  const catalogIndex = (def: InferenceServerDef) =>
    INFERENCE_SERVERS.findIndex((s) => s.id === def.id);
  const sortKey = (row: RowState): number => {
    if (row.probe?.online) return 0;
    if (row.probing || !row.probe) return 1;
    return 2;
  };
  return [...rows].sort((a, b) => {
    const d = sortKey(a) - sortKey(b);
    if (d !== 0) return d;
    return catalogIndex(a.def) - catalogIndex(b.def);
  });
}

let totalSegUiBound = false;
let totalSegModality: TotalSegModality | null = "CT";
let totalSegAssumed = false;
let totalSegUnsupportedReason = "";

function fillSelect(
  select: HTMLSelectElement,
  options: Array<{ value: string; label: string }>,
  selected: string,
): void {
  select.replaceChildren();
  for (const opt of options) {
    const el = document.createElement("option");
    el.value = opt.value;
    el.textContent = opt.label;
    if (opt.value === selected) el.selected = true;
    select.append(el);
  }
  if (!select.value && options[0]) select.value = options[0].value;
}

function syncTotalSegTaskSelect(preferredTaskId?: string): void {
  const groupSel = $("iraTotalSegGroup") as HTMLSelectElement | null;
  const taskSel = $("iraTotalSegTask") as HTMLSelectElement | null;
  if (!groupSel || !taskSel || !totalSegModality) {
    if (taskSel) taskSel.replaceChildren();
    return;
  }
  const group = groupSel.value || "total";
  const tasks = tasksInGroup(group, totalSegModality);
  const preferred =
    preferredTaskId && tasks.some((t) => t.id === preferredTaskId)
      ? preferredTaskId
      : tasks[0]?.id || "";
  fillSelect(
    taskSel,
    tasks.map((t) => ({ value: t.id, label: taskOptionLabel(t) })),
    preferred,
  );
  const taskRow = taskSel.closest(".ira-totalseg-row") as HTMLElement | null;
  if (taskRow) taskRow.hidden = tasks.length <= 1;
}

function syncTotalSegGroupSelect(preferredTaskId?: string): void {
  const groupSel = $("iraTotalSegGroup") as HTMLSelectElement | null;
  if (!groupSel || !totalSegModality) {
    if (groupSel) groupSel.replaceChildren();
    return;
  }
  const groups = groupsForModality(totalSegModality);
  const preferredTask = preferredTaskId
    ? getTotalSegTask(preferredTaskId)
    : undefined;
  const preferredGroup =
    preferredTask && preferredTask.modality === totalSegModality
      ? preferredTask.group
      : groups.includes("total")
        ? "total"
        : groups[0] || "";
  fillSelect(
    groupSel,
    groups.map((g) => ({
      value: g,
      label: TOTAL_SEG_GROUP_LABELS[g] || g,
    })),
    preferredGroup,
  );
  syncTotalSegTaskSelect(preferredTaskId || defaultTaskId(totalSegModality));
}

function clearTotalSegHits(): void {
  const hits = $("iraTotalSegHits");
  if (!hits) return;
  hits.replaceChildren();
  hits.hidden = true;
}

function selectTotalSegTask(task: TotalSegTask): void {
  if (!totalSegModality || task.modality !== totalSegModality) return;
  syncTotalSegGroupSelect(task.id);
  clearTotalSegHits();
  const find = $("iraTotalSegFind") as HTMLInputElement | null;
  if (find) find.value = "";
}

function renderTotalSegHits(query: string): void {
  const hitsEl = $("iraTotalSegHits");
  if (!hitsEl || !totalSegModality) {
    clearTotalSegHits();
    return;
  }
  const hits = searchClasses(query, totalSegModality);
  hitsEl.replaceChildren();
  if (!hits.length) {
    hitsEl.hidden = true;
    return;
  }
  for (const hit of hits) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ira-totalseg-hit";
    btn.setAttribute("role", "option");
    const cls = document.createElement("div");
    cls.className = "ira-totalseg-hit-class";
    cls.textContent = hit.className;
    const taskLine = document.createElement("div");
    taskLine.className = "ira-totalseg-hit-task";
    taskLine.textContent = `${TOTAL_SEG_GROUP_LABELS[hit.task.group] || hit.task.group} · ${taskOptionLabel(hit.task)} (${hit.task.id})`;
    btn.append(cls, taskLine);
    btn.addEventListener("click", () => selectTotalSegTask(hit.task));
    hitsEl.append(btn);
  }
  hitsEl.hidden = false;
}

function currentTotalSegTaskId(): string | undefined {
  const taskSel = $("iraTotalSegTask") as HTMLSelectElement | null;
  const raw = String(taskSel?.value || "").trim();
  return raw || undefined;
}

function refreshTotalSegModalityUi(opts?: { resetSelection?: boolean }): void {
  const prevModality = totalSegModality;
  const prevTask = currentTotalSegTaskId();
  const resolved = resolveOpenStudyTotalSegModality();
  totalSegUnsupportedReason = "";
  totalSegAssumed = false;
  if (resolved.modality) {
    totalSegModality = resolved.modality;
    totalSegAssumed = Boolean(resolved.assumed);
  } else {
    totalSegModality = null;
    totalSegUnsupportedReason = resolved.reason;
  }

  const modEl = $("iraTotalSegModality");
  if (modEl) {
    if (totalSegModality) {
      modEl.textContent = totalSegAssumed
        ? `${totalSegModality} (assumed)`
        : totalSegModality;
    } else {
      modEl.textContent = "unsupported";
    }
  }

  const modalityChanged = prevModality !== totalSegModality;
  const find = $("iraTotalSegFind") as HTMLInputElement | null;
  if (find) {
    find.disabled = !totalSegModality;
    if (modalityChanged || opts?.resetSelection) {
      find.value = "";
      clearTotalSegHits();
    }
  } else {
    clearTotalSegHits();
  }

  if (totalSegModality) {
    const keep =
      !opts?.resetSelection &&
      prevTask &&
      getTotalSegTask(prevTask)?.modality === totalSegModality
        ? prevTask
        : defaultTaskId(totalSegModality);
    syncTotalSegGroupSelect(keep);
  } else {
    const groupSel = $("iraTotalSegGroup") as HTMLSelectElement | null;
    const taskSel = $("iraTotalSegTask") as HTMLSelectElement | null;
    groupSel?.replaceChildren();
    taskSel?.replaceChildren();
  }
}

function ensureTotalSegUiBindings(): void {
  if (totalSegUiBound) return;
  totalSegUiBound = true;
  const groupSel = $("iraTotalSegGroup") as HTMLSelectElement | null;
  const taskSel = $("iraTotalSegTask") as HTMLSelectElement | null;
  const find = $("iraTotalSegFind") as HTMLInputElement | null;
  groupSel?.addEventListener("change", () => syncTotalSegTaskSelect());
  find?.addEventListener("input", () => {
    renderTotalSegHits(find.value);
  });
  find?.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") clearTotalSegHits();
  });
  // Keep task select interactive; no extra handler needed beyond group change.
  void taskSel;
}

function syncTotalSegOptionsVisibility(): void {
  const panel = $("iraTotalSegOptions");
  if (!panel) return;
  const show = selectedId === "totalseg";
  panel.hidden = !show;
  if (show) {
    ensureTotalSegUiBindings();
    // Preserve group/task across probes and detail refreshes while a job runs.
    refreshTotalSegModalityUi({ resetSelection: false });
  }
}

function readTotalSegOptions(): { task: string; fast: boolean } {
  const select = $("iraTotalSegTask") as HTMLSelectElement | null;
  const fallback = totalSegModality
    ? defaultTaskId(totalSegModality)
    : "total";
  const raw = String(select?.value || fallback).trim();
  const known = getTotalSegTask(raw);
  const task = known ? raw : fallback;
  return {
    task,
    // Prefer --fast when TotalSeg allows it for this task (catalog supportsFast).
    fast: taskSupportsFast(task),
  };
}

function renderList(): void {
  const list = $("iraInferenceList");
  if (!list) return;
  list.replaceChildren();
  for (const row of orderedRows()) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ira-inference-row";
    if (row.def.id === selectedId) btn.classList.add("is-selected");
    btn.dataset.id = row.def.id;

    const title = document.createElement("div");
    title.className = "ira-inference-row-title";
    title.textContent = row.def.title;

    const desc = document.createElement("div");
    desc.className = "ira-inference-desc";
    desc.textContent = row.def.summary || row.def.capabilities || "";

    const meta = document.createElement("div");
    meta.className = "ira-inference-row-meta";
    const status = document.createElement("span");
    status.className = "ira-inference-avail";
    if (row.probe?.online) status.classList.add("is-online");
    else if (row.probe && !row.probing) status.classList.add("is-offline");
    status.textContent = availabilityLabel(row.probe, row.probing);
    meta.append(status);
    const loc = document.createElement("span");
    loc.className = "ira-inference-location";
    loc.textContent = row.def.location;
    meta.append(loc);

    btn.append(title, desc, meta);
    btn.addEventListener("click", () => {
      selectedId = row.def.id;
      renderList();
      updateDetail();
    });
    list.append(btn);
  }
}

function updateDetail(): void {
  const detail = $("iraInferenceDetail");
  const runBtn = $("iraInferenceRunBtn") as HTMLButtonElement | null;
  const note = $("iraInferenceNote");
  const row = rows.find((r) => r.def.id === selectedId);
  if (!detail || !runBtn) return;

  if (!row) {
    detail.className = "ira-inference-detail";
    detail.textContent = "Select an inference server.";
    runBtn.disabled = true;
    if (note) note.textContent = "";
    syncTotalSegOptionsVisibility();
    return;
  }

  const opLines: string[] = [];
  if (row.probe?.error) opLines.push(`Probe error: ${row.probe.error}`);
  if (row.probe?.items?.length) {
    for (const it of row.probe.items) {
      const k = String((it as { key?: unknown }).key ?? "").trim();
      const v = String((it as { value?: unknown }).value ?? "").trim();
      if (k && v && k !== "availability") opLines.push(`${k}: ${v}`);
    }
  }

  detail.className = "ira-inference-detail ira-inference-detail-card";
  detail.innerHTML =
    renderInferenceServerInfoCardHtml(row.def) +
    (opLines.length
      ? `<pre class="ira-inference-op-notes">${opLines
          .map((l) =>
            l
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;")
          )
          .join("\n")}</pre>`
      : "");

  const info = getIraCastSessionInfo();
  const connected = info.connection === "connected";
  const files = volumeFilesFromOpenStudy();
  const online = Boolean(row.probe?.online);
  const needsNifti = row.def.hubEvent === "nifti-send";
  const hasNifti = files.some(looksLikeNiftiVolumeFile);

  // Refresh TotalSeg modality/task UI before Run enablement checks.
  syncTotalSegOptionsVisibility();

  let disableReason = "";
  if (!connected) disableReason = "Connect to Cast (open from the worklist) to run inference.";
  else if (!online) disableReason = "Server is offline — refresh status or start the resource server.";
  else if (!files.length) disableReason = "Open a study first (ImagingStudy context with volume files).";
  else if (needsNifti && !hasNifti) {
    disableReason =
      "This server expects NIfTI; the open study only has DICOM/prefix URLs. Use Lung Screening or Total Segmentator, or send NIfTI.";
  } else if (row.def.id === "totalseg" && !totalSegModality) {
    disableReason =
      totalSegUnsupportedReason ||
      "TotalSegmentator requires a CT or MR study.";
  }

  runBtn.disabled = Boolean(disableReason);
  if (note) {
    if (disableReason) {
      note.textContent = disableReason;
    } else if (row.def.id === "totalseg") {
      note.textContent = totalSegAssumed
        ? "Assuming CT — open an MR study for MR tasks."
        : "";
    } else {
      note.textContent = "Ready to run on the open study.";
    }
  }
}

function appendLog(line: string): void {
  const pre = $("iraInferenceLog");
  if (!pre) return;
  const text = String(line || "").replace(/\s+$/, "");
  if (!text) return;
  // Server status-update lines already include ``[HH:MM:SS]``; don't double-stamp.
  const stamped = /^\[[^\]]+\]\s/.test(text)
    ? text
    : `[${new Date().toLocaleTimeString()}] ${text}`;
  pre.textContent = `${pre.textContent || ""}${pre.textContent ? "\n" : ""}${stamped}`;
  pre.scrollTop = pre.scrollHeight;
}

let closeAfterJobTimer: number | null = null;
/** When true, keep the dialog open after ``Job finished`` (errors / empty SEG). */
let keepDialogOpenAfterJob = false;

function scheduleCloseAfterJobFinished(): void {
  if (closeAfterJobTimer != null) window.clearTimeout(closeAfterJobTimer);
  // Brief pause so the final status line is visible before the window disappears.
  closeAfterJobTimer = window.setTimeout(() => {
    closeAfterJobTimer = null;
    closeMedicalInferenceDialog();
  }, 700);
}

function onInferenceStatusUpdate(message: string, level: string): void {
  appendLog(message);
  if (
    level === "error" ||
    /No segmentation to save/i.test(message) ||
    /Segmentation failed/i.test(message) ||
    /no output produced/i.test(message) ||
    /Failed to publish/i.test(message)
  ) {
    keepDialogOpenAfterJob = true;
  }
  if (/\bJob finished\b/i.test(message)) {
    if (!keepDialogOpenAfterJob) {
      scheduleCloseAfterJobFinished();
    }
    keepDialogOpenAfterJob = false;
  }
}

/** Prefer keeping the current selection if it is online; else pick the first online server. */
function selectPreferredOnlineServer(): void {
  const online = orderedRows().filter((r) => r.probe?.online);
  if (!online.length) return;
  if (selectedId && online.some((r) => r.def.id === selectedId)) return;
  selectedId = online[0]!.def.id;
}

async function probeAll(): Promise<void> {
  const info = getIraCastSessionInfo();
  const client = getIraCastClient();
  if (info.connection !== "connected" || !client) {
    for (const row of rows) {
      row.probing = false;
      row.probe = { online: false, items: [], error: "Not connected to Cast" };
    }
    renderList();
    updateDetail();
    appendLog("Not connected — cannot probe inference servers.");
    return;
  }

  for (const row of rows) {
    row.probing = true;
    row.probe = null;
  }
  renderList();
  updateDetail();
  appendLog("Probing inference servers…");

  try {
    const results = await probeCastInferenceServers(client, {
      subscriberName: info.subscriberName,
      subscriberProductName: "SLICERLIVE-IRA",
      subscriberActor: "ID",
      topic: info.topic || undefined,
    });
    const byId = new Map(results.map((r) => [r.server.id, r.probe]));
    for (const row of rows) {
      row.probe = byId.get(row.def.id) || {
        online: false,
        items: [],
        error: "No STATUS response",
      };
      row.probing = false;
    }
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    for (const row of rows) {
      row.probe = { online: false, items: [], error };
      row.probing = false;
    }
  }

  selectPreferredOnlineServer();
  renderList();
  updateDetail();
  const online = rows.filter((r) => r.probe?.online).map((r) => r.def.title);
  appendLog(
    online.length
      ? `Online: ${online.join(", ")}`
      : "No inference servers responded online.",
  );
}

async function runSelected(): Promise<void> {
  const row = rows.find((r) => r.def.id === selectedId);
  if (!row) return;
  const runBtn = $("iraInferenceRunBtn") as HTMLButtonElement | null;
  if (runBtn) runBtn.disabled = true;
  keepDialogOpenAfterJob = false;
  if (closeAfterJobTimer != null) {
    window.clearTimeout(closeAfterJobTimer);
    closeAfterJobTimer = null;
  }
  appendLog(`Running ${row.def.title} (${row.def.product})…`);
  try {
    const rawFiles = volumeFilesFromOpenStudy();
    appendLog(
      rawFiles.some(looksLikePrefixEntry)
        ? "Expanding IDC series prefix to DICOM instance URLs…"
        : `Sending ${rawFiles.length} volume file(s)…`
    );
    const files = await expandVolumeFilesForSend(rawFiles);
    const hubEvent = hubEventForInferenceJob(row.def.hubEvent, files);
    appendLog(`Publishing ${files.length} file(s) as ${hubEvent}…`);
    const contextExtras =
      row.def.id === "totalseg" ? readTotalSegOptions() : undefined;
    if (contextExtras) {
      appendLog(
        `TotalSeg options: task=${contextExtras.task}, fast=${contextExtras.fast}`,
      );
    }
    const result = await publishInferenceJob({
      product: row.def.product,
      hubEvent,
      files,
      contextExtras,
    });
    appendLog(result.ok ? `Published ${hubEvent} → ${row.def.product}` : `Publish failed: ${result.detail}`);
    if (result.ok) appendLog("Watch this log for status-update lines from the server.");
    else keepDialogOpenAfterJob = true;
  } catch (err) {
    keepDialogOpenAfterJob = true;
    appendLog(`Run error: ${err instanceof Error ? err.message : String(err)}`);
  }
  updateDetail();
}

export function openMedicalInferenceDialog(): void {
  const modal = $("iraInferenceModal");
  if (!modal) {
    console.error("[ira] Medical Inference modal missing (#iraInferenceModal)");
    return;
  }
  rows = INFERENCE_SERVERS.map((def) => ({ def, probe: null, probing: false }));
  selectedId = null;
  const log = $("iraInferenceLog");
  if (log) log.textContent = "";
  installInferenceWindowDrag();
  modal.hidden = false;
  const panel = modal.querySelector(".ira-modal-inference") as HTMLElement | null;
  if (panel) applyFloatPos(modal, panel);
  try {
    ensureTotalSegUiBindings();
    renderList();
    updateDetail();
  } catch (err) {
    console.error("[ira] Medical Inference dialog render failed", err);
  }
  setIraStatusUpdateHandler(onInferenceStatusUpdate);
  void probeAll();
}

export function closeMedicalInferenceDialog(): void {
  if (closeAfterJobTimer != null) {
    window.clearTimeout(closeAfterJobTimer);
    closeAfterJobTimer = null;
  }
  const modal = $("iraInferenceModal");
  if (modal) modal.hidden = true;
  setIraStatusUpdateHandler(null);
}

/** Wire buttons once (called from installIraHeader). */
export function installMedicalInferenceUi(): void {
  if (installed) return;
  installed = true;
  ensureTotalSegUiBindings();
  installInferenceWindowDrag();
  $("iraInferenceRefreshBtn")?.addEventListener("click", () => {
    void probeAll();
  });
  $("iraInferenceRunBtn")?.addEventListener("click", () => {
    void runSelected();
  });
  $("iraInferenceMinimizeBtn")?.addEventListener("click", () => {
    toggleInferenceDialogMinimized();
  });
}
