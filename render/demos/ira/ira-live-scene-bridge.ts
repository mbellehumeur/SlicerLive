/**
 * IRA LiveScene document bridge — keeps a mrson-shaped scene cache for STATUS /
 * Scene View, and applies Cast scene-update ops onto the segroulette scene.
 *
 * Full DisplayableManager hosting is deferred; this is the use-case adapter.
 */
import { applyOp, type Op } from "../../liveops.ts";
import type { MrsonNode } from "../../mrson.ts";
import type { IraLiveSceneDoc } from "./ira-cast.ts";

export const IRA_VOL_NODE_ID = "vol1";
export const IRA_SEG_NODE_ID = "seg1";

export type IraSegDisplaySegment = {
  num: number;
  name: string;
  color: [number, number, number];
  visible: boolean;
  opacity?: number;
};

export type IraSceneSyncInput = {
  /** Preserve blobBase / url / unrelated nodes from an opened scene doc. */
  base?: IraLiveSceneDoc | null;
  dims: [number, number, number];
  ijkToRAS: number[];
  modality?: string;
  volumeName?: string;
  /** Volume-render opacity 0..1 (IRA local VR scale). */
  volumeOpacity?: number;
  /** VR transfer-function shift along the scalar axis (HU offset). */
  volumeShift?: number;
  /** Active VR preset id (Slicer CT-* name), or '' / null for default W/L. */
  volumePreset?: string | null;
  /** Volume-render crop enabled. */
  cropEnabled?: boolean;
  /** Crop ROI box visibility. */
  roiVisible?: boolean;
  segments: IraSegDisplaySegment[];
  labelsVisible: boolean;
  hasSeg: boolean;
};

type SegEntry = {
  id?: string;
  name?: string;
  labelValue: number;
  color: number[];
  visible?: boolean;
  opacity?: number;
};

function cloneNodes(
  nodes: Record<string, unknown> | undefined,
): Record<string, MrsonNode> {
  const out: Record<string, MrsonNode> = {};
  if (!nodes || typeof nodes !== "object") return out;
  for (const [id, node] of Object.entries(nodes)) {
    if (!node || typeof node !== "object") continue;
    out[id] = structuredClone(node) as MrsonNode;
  }
  return out;
}

/** Build / refresh the STATUS scene document from the current IRA study. */
export function buildIraLiveSceneDoc(input: IraSceneSyncInput): IraLiveSceneDoc {
  const nodes = cloneNodes(input.base?.nodes);
  const volName = String(input.volumeName || input.modality || "Volume").trim() || "Volume";

  const prevVol = nodes[IRA_VOL_NODE_ID] as Record<string, unknown> | undefined;
  const volOpacity =
    typeof input.volumeOpacity === "number" && Number.isFinite(input.volumeOpacity)
      ? Math.max(0, Math.min(1, input.volumeOpacity))
      : typeof prevVol?.opacity === "number" && Number.isFinite(prevVol.opacity)
        ? Math.max(0, Math.min(1, Number(prevVol.opacity)))
        : 1;
  const volShift =
    typeof input.volumeShift === "number" && Number.isFinite(input.volumeShift)
      ? Number(input.volumeShift)
      : typeof prevVol?.shift === "number" && Number.isFinite(prevVol.shift)
        ? Number(prevVol.shift)
        : 0;
  const volPreset =
    input.volumePreset !== undefined
      ? String(input.volumePreset || "").trim()
      : String(prevVol?.preset ?? "").trim();
  const cropEnabled =
    typeof input.cropEnabled === "boolean"
      ? input.cropEnabled
      : typeof prevVol?.cropEnabled === "boolean"
        ? Boolean(prevVol.cropEnabled)
        : false;
  const roiVisible =
    typeof input.roiVisible === "boolean"
      ? input.roiVisible
      : typeof prevVol?.roiVisible === "boolean"
        ? Boolean(prevVol.roiVisible)
        : false;
  nodes[IRA_VOL_NODE_ID] = {
    ...(prevVol || {}),
    type: "volume",
    id: IRA_VOL_NODE_ID,
    name: String(prevVol?.name || volName),
    dims: [...input.dims],
    ijkToRAS: [...input.ijkToRAS],
    modality: input.modality || "CT",
    opacity: volOpacity,
    shift: volShift,
    preset: volPreset,
    cropEnabled,
    roiVisible,
  } as MrsonNode;

  if (input.hasSeg && input.segments.length) {
    const segments: SegEntry[] = input.segments.map((s) => {
      const op = Math.max(
        0,
        Math.min(1, Number(s.opacity ?? 1)),
      );
      return {
        id: `Segment_${s.num}`,
        name: s.name,
        labelValue: s.num,
        color: [s.color[0], s.color[1], s.color[2], op],
        opacity: op,
        visible: s.visible !== false,
      };
    });
    const prev = nodes[IRA_SEG_NODE_ID] as Record<string, unknown> | undefined;
    nodes[IRA_SEG_NODE_ID] = {
      ...(prev || {}),
      type: "segmentation",
      id: IRA_SEG_NODE_ID,
      name: String(prev?.name || "SEG"),
      frame: "RAS",
      dims: [...input.dims],
      ijkToRAS: [...input.ijkToRAS],
      refs: { source: [IRA_VOL_NODE_ID] },
      visible: true,
      opacity: typeof prev?.opacity === "number" ? prev.opacity : 1,
      labelsVisible: input.labelsVisible,
      fill2D: (prev?.fill2D as object) || { visible: true, opacity: 0.5 },
      outline2D: (prev?.outline2D as object) || { visible: true, opacity: 1 },
      segments,
    } as MrsonNode;
  } else {
    delete nodes[IRA_SEG_NODE_ID];
  }

  return {
    ...(input.base || {}),
    blobBase: String(input.base?.blobBase || "").trim() || "ira://local/",
    nodes,
  };
}

export type SceneUpdateApplyResult = {
  labelsVisible?: boolean;
  segmentVisibility: Map<number, boolean>;
  /** Per-segment opacity 0..1 when segments were patched. */
  segmentOpacity: Map<number, number>;
  /** Global segmentation opacity 0..1 when patched. */
  opacity?: number;
  /** Volume-render opacity 0..1 when patched on a volume node. */
  volumeOpacity?: number;
  /** VR scalar-axis shift (HU) when patched on a volume node. */
  volumeShift?: number;
  /** VR preset id when patched on a volume node ('' = default W/L). */
  volumePreset?: string;
  /** Volume crop enabled when patched on a volume node. */
  cropEnabled?: boolean;
  /** Crop ROI box visibility when patched on a volume node. */
  roiVisible?: boolean;
  /** True when the scene document nodes were mutated. */
  changed: boolean;
};

/** Apply mrson ops from a scene-update resource onto a scene doc; collect visibility intents. */
export function applySceneUpdateOps(
  doc: IraLiveSceneDoc,
  ops: Op[],
): { doc: IraLiveSceneDoc; result: SceneUpdateApplyResult } {
  const nodes = cloneNodes(doc.nodes);
  const map = new Map<string, MrsonNode>(Object.entries(nodes));
  let changed = false;
  const segmentVisibility = new Map<number, boolean>();
  const segmentOpacity = new Map<number, number>();
  let labelsVisible: boolean | undefined;
  let opacity: number | undefined;
  let volumeOpacity: number | undefined;
  let volumeShift: number | undefined;
  let volumePreset: string | undefined;
  let cropEnabled: boolean | undefined;
  let roiVisible: boolean | undefined;

  for (const op of ops) {
    if (!op || typeof op !== "object" || !op.op || !op.id) continue;
    const before = map.get(op.id);
    const res = applyOp(map, op);
    if (res.changed) changed = true;
    const after = map.get(op.id);
    if (!after) continue;

    if (after.type === "volume") {
      const volOp = Number((after as { opacity?: unknown }).opacity);
      if (Number.isFinite(volOp)) {
        volumeOpacity = Math.max(0, Math.min(1, volOp));
      }
      const volSh = Number((after as { shift?: unknown }).shift);
      if (Number.isFinite(volSh)) {
        volumeShift = volSh;
      }
      if (
        op.op === "patch" &&
        typeof (op as { path?: string }).path === "string" &&
        (op as { path: string }).path === "#/preset"
      ) {
        volumePreset = String((op as { value?: unknown }).value ?? "").trim();
      }
      if (
        op.op === "patch" &&
        typeof (op as { path?: string }).path === "string" &&
        (op as { path: string }).path === "#/cropEnabled"
      ) {
        cropEnabled = Boolean((op as { value?: unknown }).value);
      }
      if (
        op.op === "patch" &&
        typeof (op as { path?: string }).path === "string" &&
        (op as { path: string }).path === "#/roiVisible"
      ) {
        roiVisible = Boolean((op as { value?: unknown }).value);
      }
      continue;
    }

    if (after.type !== "segmentation") continue;

    if (typeof after.labelsVisible === "boolean") {
      labelsVisible = after.labelsVisible;
    }
    const opVal = Number((after as { opacity?: unknown }).opacity);
    if (Number.isFinite(opVal)) {
      opacity = Math.max(0, Math.min(1, opVal));
    }

    const segs = after.segments as SegEntry[] | undefined;
    if (Array.isArray(segs)) {
      for (const s of segs) {
        const lv = Number(s.labelValue);
        if (!Number.isFinite(lv)) continue;
        segmentVisibility.set(lv, s.visible !== false);
        const fromField = Number(s.opacity);
        const fromColor = Number(s.color?.[3]);
        const so = Number.isFinite(fromField)
          ? fromField
          : Number.isFinite(fromColor)
            ? fromColor
            : 1;
        segmentOpacity.set(lv, Math.max(0, Math.min(1, so)));
      }
    }

    // Path-only patch of a single segment visible flag
    if (
      op.op === "patch" &&
      typeof (op as { path?: string }).path === "string" &&
      /#\/segments\/\d+\/visible/.test((op as { path: string }).path)
    ) {
      const m = (op as { path: string }).path.match(/#\/segments\/(\d+)\/visible/);
      if (m && Array.isArray(segs)) {
        const idx = Number(m[1]);
        const s = segs[idx];
        if (s) segmentVisibility.set(Number(s.labelValue), Boolean((op as { value: unknown }).value));
      }
    }

    void before;
  }

  const nextNodes: Record<string, unknown> = {};
  for (const [id, n] of map) nextNodes[id] = n;
  return {
    doc: { ...doc, nodes: nextNodes },
    result: {
      labelsVisible,
      segmentVisibility,
      segmentOpacity,
      opacity,
      volumeOpacity,
      volumeShift,
      volumePreset,
      cropEnabled,
      roiVisible,
      changed,
    },
  };
}

/** Parse Cast scene-update context into ops (+ optional nested display event). */
export function opsFromSceneUpdateContext(context: unknown): Op[] {
  if (!Array.isArray(context)) return [];
  const ops: Op[] = [];
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const row = item as { key?: unknown; resource?: unknown };
    const key = String(row.key || "").toLowerCase();
    if (key !== "scene" && key !== "sceneupdate") continue;
    const resource = row.resource;
    if (!resource || typeof resource !== "object") continue;
    const res = resource as {
      ops?: unknown;
      event?: unknown;
      sourceId?: unknown;
      display?: unknown;
    };

    if (Array.isArray(res.ops)) {
      for (const op of res.ops) {
        if (op && typeof op === "object" && typeof (op as Op).op === "string") {
          ops.push(op as Op);
        }
      }
    }

    // Nested SegmentationDisplayModified → synthetic patches
    const evName = String(res.event || "").trim();
    if (evName === "SegmentationDisplayModified" && res.display && typeof res.display === "object") {
      const sourceId = String(res.sourceId || IRA_SEG_NODE_ID).trim() || IRA_SEG_NODE_ID;
      const disp = res.display as Record<string, unknown>;
      if ("labelsVisible" in disp) {
        ops.push({
          op: "patch",
          id: sourceId,
          path: "#/labelsVisible",
          value: Boolean(disp.labelsVisible),
        });
      }
      if (Array.isArray(disp.segments)) {
        ops.push({
          op: "patch",
          id: sourceId,
          path: "#/segments",
          value: disp.segments,
        });
      }
      if ("visible" in disp) {
        ops.push({
          op: "patch",
          id: sourceId,
          path: "#/visible",
          value: Boolean(disp.visible),
        });
      }
      if ("opacity" in disp) {
        ops.push({
          op: "patch",
          id: sourceId,
          path: "#/opacity",
          value: disp.opacity,
        });
      }
    }
  }
  return ops;
}

/** Find the primary segmentation node id in a scene doc. */
export function primarySegNodeId(doc: IraLiveSceneDoc | null | undefined): string | null {
  if (!doc?.nodes || typeof doc.nodes !== "object") return null;
  if (doc.nodes[IRA_SEG_NODE_ID]) return IRA_SEG_NODE_ID;
  for (const [id, n] of Object.entries(doc.nodes)) {
    if (n && typeof n === "object" && (n as { type?: string }).type === "segmentation") {
      return id;
    }
  }
  return null;
}
