// Medical Inference dialog for IRA — probe Cast resource servers and run one.
import {
  CAST_INFERENCE_SERVERS,
  extractImagingStudyFiles,
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
  probeInferenceProduct,
  setIraStatusUpdateHandler,
} from "./ira-cast.ts";
import {
  TOTAL_SEG_DEFAULT_TASK,
  TOTAL_SEG_TASK_OPTIONS,
} from "./totalseg-tasks.ts";

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

function ensureTotalSegTaskSelect(): void {
  const select = $("iraTotalSegTask") as HTMLSelectElement | null;
  if (!select || select.options.length > 0) return;
  for (const opt of TOTAL_SEG_TASK_OPTIONS) {
    const el = document.createElement("option");
    el.value = opt.value;
    el.textContent = opt.title;
    if (opt.value === TOTAL_SEG_DEFAULT_TASK) el.selected = true;
    select.append(el);
  }
}

function syncTotalSegOptionsVisibility(): void {
  const panel = $("iraTotalSegOptions");
  if (!panel) return;
  const show = selectedId === "totalseg";
  panel.hidden = !show;
  if (show) ensureTotalSegTaskSelect();
}

function readTotalSegOptions(): { task: string; fast: boolean } {
  const select = $("iraTotalSegTask") as HTMLSelectElement | null;
  const raw = String(select?.value || TOTAL_SEG_DEFAULT_TASK).trim();
  const known = TOTAL_SEG_TASK_OPTIONS.some((o) => o.value === raw);
  return {
    task: known ? raw : TOTAL_SEG_DEFAULT_TASK,
    fast: true,
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
  const hasNifti = files.some((f) => /\.nii(\.gz)?$/i.test(f.fileName) || /\.nii(\.gz)?(\?|$)/i.test(f.url));

  let disableReason = "";
  if (!connected) disableReason = "Connect to Cast (open from the worklist) to run inference.";
  else if (!online) disableReason = "Server is offline — refresh status or start the resource server.";
  else if (!files.length) disableReason = "Open a study first (ImagingStudy context with volume files).";
  else if (needsNifti && !hasNifti) {
    disableReason =
      "This server expects NIfTI; the open study only has DICOM/prefix URLs. Use Lung Screening or Total Segmentator, or send NIfTI.";
  }

  runBtn.disabled = Boolean(disableReason);
  if (note) {
    if (disableReason) {
      note.textContent = disableReason;
    } else if (row.def.id === "totalseg") {
      // Options panel replaces the ready message for TotalSeg.
      note.textContent = "";
    } else {
      note.textContent = "Ready to run on the open study.";
    }
  }
  syncTotalSegOptionsVisibility();
}

function appendLog(line: string): void {
  const pre = $("iraInferenceLog");
  if (!pre) return;
  const stamp = new Date().toLocaleTimeString();
  pre.textContent = `${pre.textContent || ""}${pre.textContent ? "\n" : ""}[${stamp}] ${line}`;
  pre.scrollTop = pre.scrollHeight;
}

let closeAfterJobTimer: number | null = null;

function scheduleCloseAfterJobFinished(): void {
  if (closeAfterJobTimer != null) window.clearTimeout(closeAfterJobTimer);
  // Brief pause so the final status line is visible before the window disappears.
  closeAfterJobTimer = window.setTimeout(() => {
    closeAfterJobTimer = null;
    closeMedicalInferenceDialog();
  }, 700);
}

function onInferenceStatusUpdate(message: string, _level: string): void {
  appendLog(message);
  if (/\bJob finished\b/i.test(message)) {
    scheduleCloseAfterJobFinished();
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
  if (info.connection !== "connected" || !getIraCastClient()) {
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

  await Promise.all(
    rows.map(async (row) => {
      try {
        row.probe = await probeInferenceProduct(row.def.product);
      } catch (err) {
        row.probe = {
          online: false,
          items: [],
          error: err instanceof Error ? err.message : String(err),
        };
      } finally {
        row.probing = false;
      }
    }),
  );
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
  appendLog(`Running ${row.def.title} (${row.def.product})…`);
  try {
    const rawFiles = volumeFilesFromOpenStudy();
    appendLog(
      rawFiles.some(looksLikePrefixEntry)
        ? "Expanding IDC series prefix to DICOM instance URLs…"
        : `Sending ${rawFiles.length} volume file(s)…`
    );
    const files = await expandVolumeFilesForSend(rawFiles);
    appendLog(`Publishing ${files.length} file(s) as ${row.def.hubEvent}…`);
    const contextExtras =
      row.def.id === "totalseg" ? readTotalSegOptions() : undefined;
    if (contextExtras) {
      appendLog(`TotalSeg options: task=${contextExtras.task}, fast`);
    }
    const result = await publishInferenceJob({
      product: row.def.product,
      hubEvent: row.def.hubEvent,
      files,
      contextExtras,
    });
    appendLog(result.ok ? `Published ${row.def.hubEvent} → ${row.def.product}` : `Publish failed: ${result.detail}`);
    if (result.ok) appendLog("Watch this log for status-update lines from the server.");
  } catch (err) {
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
    ensureTotalSegTaskSelect();
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
  ensureTotalSegTaskSelect();
  installInferenceWindowDrag();
  $("iraInferenceRefreshBtn")?.addEventListener("click", () => {
    void probeAll();
  });
  $("iraInferenceRunBtn")?.addEventListener("click", () => {
    void runSelected();
  });
}
