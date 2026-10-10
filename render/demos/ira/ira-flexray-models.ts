/** FleXray Low / High quality list from the public demo manifest (JSON only — no ONNX). */

export const FLEXRAY_MANIFEST_URL =
  "https://flexray.csail.mit.edu/demo/demo_manifest.json";

export type FlexrayModelChoice = {
  id: string;
  sizeBytes: number;
  optionLabel: string;
  isDefault: boolean;
};

export function formatFlexrayDownloadSize(bytes: number): string {
  const n = Number(bytes) || 0;
  if (n >= 1e9) {
    const gb = n / 1e9;
    return `${gb >= 10 ? Math.round(gb) : gb.toFixed(1).replace(/\.0$/, "")} GB`;
  }
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${n} B`;
}

function listCheckpoints(manifest: unknown): Array<{ id: string; sizeBytes: number; isDefault: boolean }> {
  const rec = manifest && typeof manifest === "object" ? (manifest as Record<string, unknown>) : {};
  const defaultModel =
    rec.model && typeof rec.model === "object"
      ? (rec.model as Record<string, unknown>)
      : null;
  const defaultId = String(defaultModel?.id || "").trim();
  const out: Array<{ id: string; sizeBytes: number; isDefault: boolean }> = [];
  const seen = new Set<string>();
  const push = (raw: Record<string, unknown> | null, isDefault: boolean) => {
    if (!raw) return;
    const id = String(raw.id || "").trim();
    const url = String(raw.url || "").trim();
    if (!id || !url || seen.has(id)) return;
    seen.add(id);
    out.push({
      id,
      sizeBytes: Number(raw.size_bytes) || 0,
      isDefault: isDefault || id === defaultId,
    });
  };
  push(defaultModel, true);
  const ensemble =
    rec.ensemble && typeof rec.ensemble === "object"
      ? (rec.ensemble as Record<string, unknown>)
      : null;
  const members = Array.isArray(ensemble?.members) ? ensemble.members : [];
  for (const member of members) {
    if (member && typeof member === "object") {
      push(member as Record<string, unknown>, false);
    }
  }
  return out;
}

export function listFlexrayModelChoices(manifest: unknown): FlexrayModelChoice[] {
  const all = listCheckpoints(manifest);
  const sum = (items: typeof all) => items.reduce((s, m) => s + m.sizeBytes, 0);
  const low = all.filter((m) => m.isDefault);
  const lowModels = low.length ? low : all.slice(0, 1);
  const lowBytes = sum(lowModels);
  const highBytes = sum(all);
  return [
    {
      id: "low",
      sizeBytes: lowBytes,
      isDefault: true,
      optionLabel: `Low — ${formatFlexrayDownloadSize(lowBytes)}`,
    },
    {
      id: "high",
      sizeBytes: highBytes,
      isDefault: false,
      optionLabel: `High — ${formatFlexrayDownloadSize(highBytes)}`,
    },
  ];
}

export async function fetchFlexrayModelChoices(): Promise<FlexrayModelChoice[]> {
  const res = await fetch(FLEXRAY_MANIFEST_URL, { mode: "cors" });
  if (!res.ok) {
    throw new Error(`FleXray manifest HTTP ${res.status}`);
  }
  return listFlexrayModelChoices(await res.json());
}
