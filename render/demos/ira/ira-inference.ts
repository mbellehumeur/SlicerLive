// Medical Inference dialog for IRA — probe Cast resource servers and run one.
import {
  extractImagingStudyFiles,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import {
  INFERENCE_SERVERS,
  type InferenceServerDef,
} from "../../../../worklist/inference-servers.ts";
import {
  getIraCastClient,
  getIraCastSessionInfo,
  getIraImagingStudyOpenContext,
  publishInferenceJob,
  probeInferenceProduct,
  setIraStatusUpdateHandler,
  type InferenceProbeResult,
} from "./ira-cast.ts";

export type { InferenceServerDef };
export { INFERENCE_SERVERS };

type RowState = {
  def: InferenceServerDef;
  probe: InferenceProbeResult | null;
  probing: boolean;
};

let selectedId: string | null = null;
let rows: RowState[] = [];
let installed = false;

function $(id: string): HTMLElement | null {
  return document.getElementById(id);
}

function volumeFilesFromOpenStudy(): Array<{ url: string; fileName: string }> {
  const ctx = getIraImagingStudyOpenContext();
  const files = extractImagingStudyFiles(ctx) as Array<Record<string, unknown>>;
  const out: Array<{ url: string; fileName: string }> = [];
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

function renderList(): void {
  const list = $("iraInferenceList");
  if (!list) return;
  list.replaceChildren();
  for (const row of rows) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ira-inference-row";
    if (row.def.id === selectedId) btn.classList.add("is-selected");
    btn.dataset.id = row.def.id;

    const title = document.createElement("div");
    title.className = "ira-inference-row-title";
    title.textContent = row.def.title;

    const loc = document.createElement("div");
    loc.className = "ira-inference-location";
    loc.textContent = row.def.location;

    const meta = document.createElement("div");
    meta.className = "ira-inference-row-meta";
    const status = document.createElement("span");
    status.className = "ira-inference-avail";
    if (row.probe?.online) status.classList.add("is-online");
    else if (row.probe && !row.probing) status.classList.add("is-offline");
    status.textContent = availabilityLabel(row.probe, row.probing);
    meta.append(status);
    const prod = document.createElement("span");
    prod.className = "ira-inference-product";
    prod.textContent = row.def.product;
    meta.append(prod);

    const caps = document.createElement("div");
    caps.className = "ira-inference-caps";
    caps.textContent = row.def.capabilities;

    btn.append(title, loc, meta, caps);
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
    detail.textContent = "Select an inference server.";
    runBtn.disabled = true;
    if (note) note.textContent = "";
    return;
  }

  const lines = [
    row.def.title,
    `Location: ${row.def.location}`,
    `Product: ${row.def.product}`,
    `Status: ${availabilityLabel(row.probe, row.probing)}`,
    row.def.capabilities,
  ];
  if (row.probe?.error) lines.push(`Probe error: ${row.probe.error}`);
  if (row.probe?.items?.length) {
    for (const it of row.probe.items) {
      const k = String((it as { key?: unknown }).key ?? "").trim();
      const v = String((it as { value?: unknown }).value ?? "").trim();
      if (k && v && k !== "availability") lines.push(`${k}: ${v}`);
    }
  }
  detail.textContent = lines.join("\n");

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
  if (note) note.textContent = disableReason || "Ready to run on the open study.";
}

function appendLog(line: string): void {
  const pre = $("iraInferenceLog");
  if (!pre) return;
  const stamp = new Date().toLocaleTimeString();
  pre.textContent = `${pre.textContent || ""}${pre.textContent ? "\n" : ""}[${stamp}] ${line}`;
  pre.scrollTop = pre.scrollHeight;
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
  const files = volumeFilesFromOpenStudy();
  const runBtn = $("iraInferenceRunBtn") as HTMLButtonElement | null;
  if (runBtn) runBtn.disabled = true;
  appendLog(`Running ${row.def.title} (${row.def.product})…`);
  try {
    const result = await publishInferenceJob({
      product: row.def.product,
      hubEvent: row.def.hubEvent,
      files,
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
  selectedId = rows[0]?.def.id ?? null;
  const log = $("iraInferenceLog");
  if (log) log.textContent = "";
  modal.hidden = false;
  try {
    renderList();
    updateDetail();
  } catch (err) {
    console.error("[ira] Medical Inference dialog render failed", err);
  }
  setIraStatusUpdateHandler((message) => appendLog(message));
  void probeAll();
}

export function closeMedicalInferenceDialog(): void {
  const modal = $("iraInferenceModal");
  if (modal) modal.hidden = true;
  setIraStatusUpdateHandler(null);
}

/** Wire buttons once (called from installIraHeader). */
export function installMedicalInferenceUi(): void {
  if (installed) return;
  installed = true;
  $("iraInferenceRefreshBtn")?.addEventListener("click", () => {
    void probeAll();
  });
  $("iraInferenceRunBtn")?.addEventListener("click", () => {
    void runSelected();
  });
}
