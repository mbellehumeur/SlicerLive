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
export const IRA_CAM_NODE_ID = "cam1";
export const IRA_LAYOUT_NODE_ID = "layout1";

/** Slicer layoutName ↔ IRA MPR plane (Red/Yellow/Green + Oblique). */
export type IraSliceLayoutName = "Red" | "Yellow" | "Green" | "Oblique";
export type IraSlicePlaneName = "axial" | "sagittal" | "coronal" | "oblique";

export const IRA_SLICE_LAYOUT_BY_PLANE: Record<
  IraSlicePlaneName,
  IraSliceLayoutName
> = {
  axial: "Red",
  sagittal: "Yellow",
  coronal: "Green",
  oblique: "Oblique",
};

export const IRA_SLICE_PLANE_BY_LAYOUT: Record<
  IraSliceLayoutName,
  IraSlicePlaneName
> = {
  Red: "axial",
  Yellow: "sagittal",
  Green: "coronal",
  Oblique: "oblique",
};

export const IRA_SLICE_ORIENTATION_BY_PLANE: Record<IraSlicePlaneName, string> = {
  axial: "Axial",
  sagittal: "Sagittal",
  coronal: "Coronal",
  oblique: "Oblique",
};

export function iraSliceNodeId(layout: IraSliceLayoutName): string {
  return `iraSlice-${layout}`;
}

export const IRA_SLICE_NODE_IDS = (
  Object.keys(IRA_SLICE_PLANE_BY_LAYOUT) as IraSliceLayoutName[]
).map(iraSliceNodeId);

/** LiveScene-compatible MPR slice frame for STATUS / scene-update. */
export type IraSliceFrame = {
  layoutName: IraSliceLayoutName;
  orientation: string;
  /** Row-major 4×4: columns = u, v, n; last column = plane centre RAS. */
  sliceToRAS: number[];
  /** [fovX, fovY, slab] mm. */
  fieldOfView: [number, number, number];
};

/** Follower intent extracted from a put/patch slice view node. */
export type IraSliceApplyIntent = {
  layoutName: IraSliceLayoutName;
  plane: IraSlicePlaneName;
  centerRAS: [number, number, number];
  fieldOfView?: [number, number, number];
  basis: {
    uDir: [number, number, number];
    vDir: [number, number, number];
    nDir: [number, number, number];
  };
};

/** Build sliceToRAS (row-major) from basis columns + centre. */
export function sliceToRASFromBasis(
  uDir: [number, number, number],
  vDir: [number, number, number],
  nDir: [number, number, number],
  center: [number, number, number],
): number[] {
  return [
    uDir[0],
    vDir[0],
    nDir[0],
    center[0],
    uDir[1],
    vDir[1],
    nDir[1],
    center[1],
    uDir[2],
    vDir[2],
    nDir[2],
    center[2],
    0,
    0,
    0,
    1,
  ];
}

export type IraSegDisplaySegment = {
  num: number;
  name: string;
  color: [number, number, number];
  visible: boolean;
  opacity?: number;
};

/** 3D camera pose (RAS / world), matching mrson camera node fields. */
export type IraCameraPose = {
  position: [number, number, number];
  focalPoint: [number, number, number];
  viewUp: [number, number, number];
  viewAngle: number;
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
  /**
   * Half-span of the VR shift slider (HU), typically max(200, (vmax-vmin)/2).
   * Published so remote clients (reporting) can match IRA’s slider range.
   */
  volumeShiftRange?: number;
  /** Active VR preset id (Slicer CT-* name), or '' / null for default W/L. */
  volumePreset?: string | null;
  /** Volume-render crop enabled. */
  cropEnabled?: boolean;
  /** Crop ROI box visibility. */
  roiVisible?: boolean;
  segments: IraSegDisplaySegment[];
  labelsVisible: boolean;
  /** Global segmentation opacity 0..1. */
  opacity?: number;
  hasSeg: boolean;
  /** Current 3D camera pose. */
  camera?: IraCameraPose | null;
  /** MPR slice frames (Red/Yellow/Green/Oblique LiveScene view nodes). */
  slices?: IraSliceFrame[] | null;
  /** Double-click maximized viewport cell id, or null when 2×2 is restored. */
  maximizedCell?: string | null;
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
  const volShiftRange =
    typeof input.volumeShiftRange === "number" &&
    Number.isFinite(input.volumeShiftRange) &&
    input.volumeShiftRange > 0
      ? Number(input.volumeShiftRange)
      : typeof prevVol?.shiftRange === "number" &&
          Number.isFinite(prevVol.shiftRange) &&
          Number(prevVol.shiftRange) > 0
        ? Number(prevVol.shiftRange)
        : 500;
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
    shiftRange: volShiftRange,
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
    const segOpacity =
      typeof input.opacity === "number" && Number.isFinite(input.opacity)
        ? Math.max(0, Math.min(1, input.opacity))
        : typeof prev?.opacity === "number"
          ? Number(prev.opacity)
          : 1;
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
      opacity: segOpacity,
      labelsVisible: input.labelsVisible,
      fill2D: (prev?.fill2D as object) || { visible: true, opacity: 0.5 },
      outline2D: (prev?.outline2D as object) || { visible: true, opacity: 1 },
      segments,
    } as MrsonNode;
  } else {
    delete nodes[IRA_SEG_NODE_ID];
  }

  if (input.camera) {
    const prevCam = nodes[IRA_CAM_NODE_ID] as Record<string, unknown> | undefined;
    const c = input.camera;
    nodes[IRA_CAM_NODE_ID] = {
      ...(prevCam || {}),
      type: "camera",
      id: IRA_CAM_NODE_ID,
      name: String(prevCam?.name || "Camera"),
      position: [...c.position],
      focalPoint: [...c.focalPoint],
      viewUp: [...c.viewUp],
      viewAngle: Number.isFinite(c.viewAngle) ? c.viewAngle : 30,
    } as MrsonNode;
  }

  if (Array.isArray(input.slices)) {
    for (const frame of input.slices) {
      if (!frame?.layoutName || !Array.isArray(frame.sliceToRAS)) continue;
      const id = iraSliceNodeId(frame.layoutName);
      const prevSlice = nodes[id] as Record<string, unknown> | undefined;
      const fov = frame.fieldOfView;
      nodes[id] = {
        ...(prevSlice || {}),
        type: "view",
        kind: "slice",
        id,
        name: frame.layoutName,
        layoutName: frame.layoutName,
        orientation: String(frame.orientation || "").trim() || frame.layoutName,
        sliceToRAS: [...frame.sliceToRAS],
        fieldOfView: [
          Number(fov?.[0]) || 0,
          Number(fov?.[1]) || 0,
          Number(fov?.[2]) || 0,
        ],
      } as MrsonNode;
    }
  }

  const prevLayout = nodes[IRA_LAYOUT_NODE_ID] as Record<string, unknown> | undefined;
  const maximizedCell =
    input.maximizedCell !== undefined
      ? input.maximizedCell
        ? String(input.maximizedCell).trim() || null
        : null
      : prevLayout?.maximizedCell != null &&
          String(prevLayout.maximizedCell).trim()
        ? String(prevLayout.maximizedCell).trim()
        : null;
  nodes[IRA_LAYOUT_NODE_ID] = {
    ...(prevLayout || {}),
    type: "layout",
    id: IRA_LAYOUT_NODE_ID,
    name: String(prevLayout?.name || "Layout"),
    arrangementName: String(prevLayout?.arrangementName || "fourUp"),
    maximizedCell,
  } as MrsonNode;

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
  /** 3D camera pose when a camera node was patched / put. */
  camera?: IraCameraPose;
  /** MPR slice frames when view/slice nodes were put / patched. */
  slices: IraSliceApplyIntent[];
  /**
   * Maximized viewport cell when a layout node was applied.
   * `null` = restore multi-up; `undefined` = layout not in this update.
   */
  maximizedCell?: string | null;
  /** True when the scene document nodes were mutated. */
  changed: boolean;
};

function vec3From(value: unknown): [number, number, number] | null {
  if (!Array.isArray(value) || value.length < 3) return null;
  const a = Number(value[0]);
  const b = Number(value[1]);
  const c = Number(value[2]);
  if (![a, b, c].every(Number.isFinite)) return null;
  return [a, b, c];
}

function norm3(v: [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function layoutNameFromSliceNode(
  node: Record<string, unknown>,
): IraSliceLayoutName | null {
  const raw = String(node.layoutName || node.name || "").trim();
  if (raw === "Red" || raw === "Yellow" || raw === "Green" || raw === "Oblique") {
    return raw;
  }
  const id = String(node.id || "");
  if (id.endsWith("-Red")) return "Red";
  if (id.endsWith("-Yellow")) return "Yellow";
  if (id.endsWith("-Green")) return "Green";
  if (id.endsWith("-Oblique")) return "Oblique";
  const orient = String(node.orientation || "").trim().toLowerCase();
  if (orient === "axial") return "Red";
  if (orient === "sagittal") return "Yellow";
  if (orient === "coronal") return "Green";
  if (orient === "oblique") return "Oblique";
  return null;
}

/** Parse a LiveScene view/slice node into an IRA apply intent. */
export function sliceIntentFromNode(
  node: Record<string, unknown> | undefined,
): IraSliceApplyIntent | null {
  if (!node || node.type !== "view" || node.kind !== "slice") return null;
  const layoutName = layoutNameFromSliceNode(node);
  if (!layoutName) return null;
  const m = node.sliceToRAS;
  if (!Array.isArray(m) || m.length < 16) return null;
  const nums = m.map((x) => Number(x));
  if (!nums.every(Number.isFinite)) return null;
  const col = (c: number): [number, number, number] => [
    nums[c]!,
    nums[4 + c]!,
    nums[8 + c]!,
  ];
  const centerRAS = col(3);
  const uDir = norm3(col(0));
  const vDir = norm3(col(1));
  const nDir = norm3(col(2));
  const fovRaw = node.fieldOfView;
  let fieldOfView: [number, number, number] | undefined;
  if (Array.isArray(fovRaw) && fovRaw.length >= 2) {
    const fx = Number(fovRaw[0]);
    const fy = Number(fovRaw[1]);
    const fz = Number(fovRaw[2]);
    if (Number.isFinite(fx) && Number.isFinite(fy) && fx > 0 && fy > 0) {
      fieldOfView = [fx, fy, Number.isFinite(fz) ? fz : 0];
    }
  }
  return {
    layoutName,
    plane: IRA_SLICE_PLANE_BY_LAYOUT[layoutName],
    centerRAS,
    fieldOfView,
    basis: { uDir, vDir, nDir },
  };
}

function cameraPoseFromNode(
  node: Record<string, unknown> | undefined,
): IraCameraPose | undefined {
  if (!node || node.type !== "camera") return undefined;
  const position = vec3From(node.position);
  const focalPoint = vec3From(node.focalPoint);
  const viewUp = vec3From(node.viewUp);
  if (!position || !focalPoint || !viewUp) return undefined;
  const viewAngle = Number(node.viewAngle);
  return {
    position,
    focalPoint,
    viewUp,
    viewAngle: Number.isFinite(viewAngle) ? viewAngle : 30,
  };
}

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
  let camera: IraCameraPose | undefined;
  const slicesByLayout = new Map<IraSliceLayoutName, IraSliceApplyIntent>();
  let maximizedCell: string | null | undefined;

  for (const op of ops) {
    if (!op || typeof op !== "object" || !op.op || !op.id) continue;
    const before = map.get(op.id);
    const res = applyOp(map, op);
    if (res.changed) changed = true;
    const after = map.get(op.id);
    if (!after) continue;

    if (after.type === "layout") {
      const raw = (after as { maximizedCell?: unknown }).maximizedCell;
      maximizedCell =
        raw == null || raw === "" ? null : String(raw).trim() || null;
      continue;
    }

    if (after.type === "camera") {
      const pose = cameraPoseFromNode(after as Record<string, unknown>);
      if (pose) camera = pose;
      continue;
    }

    if (after.type === "view" && (after as { kind?: string }).kind === "slice") {
      const intent = sliceIntentFromNode(after as Record<string, unknown>);
      if (intent) slicesByLayout.set(intent.layoutName, intent);
      continue;
    }

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
      camera,
      slices: [...slicesByLayout.values()],
      maximizedCell,
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

/** Find the primary volume node id in a scene doc. */
export function primaryVolNodeId(doc: IraLiveSceneDoc | null | undefined): string | null {
  if (!doc?.nodes || typeof doc.nodes !== "object") return null;
  if (doc.nodes[IRA_VOL_NODE_ID]) return IRA_VOL_NODE_ID;
  for (const [id, n] of Object.entries(doc.nodes)) {
    if (n && typeof n === "object" && (n as { type?: string }).type === "volume") {
      return id;
    }
  }
  return null;
}

function stableJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return "";
  }
}

function displayFieldChanged(
  prevNode: Record<string, unknown> | undefined,
  nextNode: Record<string, unknown> | undefined,
  key: string,
): boolean {
  if (!prevNode) return nextNode != null && key in nextNode;
  return stableJson(prevNode[key]) !== stableJson(nextNode?.[key]);
}

/**
 * Build scene-update patch ops for display fields (seg visibility/opacity, VR, crop).
 * When ``prev`` is provided, only emit ops for fields that changed.
 */
export function buildIraDisplaySceneUpdateOps(
  doc: IraLiveSceneDoc | null | undefined,
  prev?: IraLiveSceneDoc | null,
): Op[] {
  if (!doc?.nodes || typeof doc.nodes !== "object") return [];
  const ops: Op[] = [];
  const full = !prev?.nodes;

  const segId = primarySegNodeId(doc) || IRA_SEG_NODE_ID;
  const seg = doc.nodes[segId] as Record<string, unknown> | undefined;
  const prevSeg = prev?.nodes?.[segId] as Record<string, unknown> | undefined;
  if (seg && typeof seg === "object" && seg.type === "segmentation") {
    if (full || displayFieldChanged(prevSeg, seg, "segments")) {
      if (Array.isArray(seg.segments)) {
        ops.push({
          op: "patch",
          id: segId,
          path: "#/segments",
          value: structuredClone(seg.segments),
        });
      }
    }
    if (full || displayFieldChanged(prevSeg, seg, "labelsVisible")) {
      if (typeof seg.labelsVisible === "boolean") {
        ops.push({
          op: "patch",
          id: segId,
          path: "#/labelsVisible",
          value: seg.labelsVisible,
        });
      }
    }
    if (full || displayFieldChanged(prevSeg, seg, "opacity")) {
      const o = Number(seg.opacity);
      if (Number.isFinite(o)) {
        ops.push({
          op: "patch",
          id: segId,
          path: "#/opacity",
          value: Math.max(0, Math.min(1, o)),
        });
      }
    }
  }

  const volId = primaryVolNodeId(doc) || IRA_VOL_NODE_ID;
  const vol = doc.nodes[volId] as Record<string, unknown> | undefined;
  const prevVol = prev?.nodes?.[volId] as Record<string, unknown> | undefined;
  if (vol && typeof vol === "object" && vol.type === "volume") {
    if (full || displayFieldChanged(prevVol, vol, "opacity")) {
      const o = Number(vol.opacity);
      if (Number.isFinite(o)) {
        ops.push({
          op: "patch",
          id: volId,
          path: "#/opacity",
          value: Math.max(0, Math.min(1, o)),
        });
      }
    }
    if (full || displayFieldChanged(prevVol, vol, "shift")) {
      const s = Number(vol.shift);
      if (Number.isFinite(s)) {
        ops.push({
          op: "patch",
          id: volId,
          path: "#/shift",
          value: s,
        });
      }
    }
    if (full || displayFieldChanged(prevVol, vol, "shiftRange")) {
      const r = Number(vol.shiftRange);
      if (Number.isFinite(r) && r > 0) {
        ops.push({
          op: "patch",
          id: volId,
          path: "#/shiftRange",
          value: r,
        });
      }
    }
    if (full || displayFieldChanged(prevVol, vol, "preset")) {
      ops.push({
        op: "patch",
        id: volId,
        path: "#/preset",
        value: String(vol.preset ?? "").trim(),
      });
    }
    if (full || displayFieldChanged(prevVol, vol, "cropEnabled")) {
      if (typeof vol.cropEnabled === "boolean") {
        ops.push({
          op: "patch",
          id: volId,
          path: "#/cropEnabled",
          value: vol.cropEnabled,
        });
      }
    }
    if (full || displayFieldChanged(prevVol, vol, "roiVisible")) {
      if (typeof vol.roiVisible === "boolean") {
        ops.push({
          op: "patch",
          id: volId,
          path: "#/roiVisible",
          value: vol.roiVisible,
        });
      }
    }
  }

  const camId = IRA_CAM_NODE_ID;
  const cam = doc.nodes[camId] as Record<string, unknown> | undefined;
  const prevCam = prev?.nodes?.[camId] as Record<string, unknown> | undefined;
  if (cam && typeof cam === "object" && cam.type === "camera") {
    const camChanged =
      full ||
      !prevCam ||
      displayFieldChanged(prevCam, cam, "position") ||
      displayFieldChanged(prevCam, cam, "focalPoint") ||
      displayFieldChanged(prevCam, cam, "viewUp") ||
      displayFieldChanged(prevCam, cam, "viewAngle");
    if (camChanged) {
      // put so followers without an existing cam1 node still apply the pose
      ops.push({
        op: "put",
        id: camId,
        node: structuredClone(cam) as MrsonNode,
      });
    }
  }

  for (const layout of Object.keys(
    IRA_SLICE_PLANE_BY_LAYOUT,
  ) as IraSliceLayoutName[]) {
    const id = iraSliceNodeId(layout);
    const slice = doc.nodes[id] as Record<string, unknown> | undefined;
    const prevSlice = prev?.nodes?.[id] as Record<string, unknown> | undefined;
    if (!slice || typeof slice !== "object" || slice.type !== "view") continue;
    if ((slice as { kind?: string }).kind !== "slice") continue;
    const sliceChanged =
      full ||
      !prevSlice ||
      displayFieldChanged(prevSlice, slice, "sliceToRAS") ||
      displayFieldChanged(prevSlice, slice, "fieldOfView") ||
      displayFieldChanged(prevSlice, slice, "orientation") ||
      displayFieldChanged(prevSlice, slice, "layoutName");
    if (sliceChanged) {
      ops.push({
        op: "put",
        id,
        node: structuredClone(slice) as MrsonNode,
      });
    }
  }

  const layoutId = IRA_LAYOUT_NODE_ID;
  const layout = doc.nodes[layoutId] as Record<string, unknown> | undefined;
  const prevLayout = prev?.nodes?.[layoutId] as Record<string, unknown> | undefined;
  if (layout && typeof layout === "object" && layout.type === "layout") {
    if (full || displayFieldChanged(prevLayout, layout, "maximizedCell")) {
      ops.push({
        op: "put",
        id: layoutId,
        node: structuredClone(layout) as MrsonNode,
      });
    }
  }

  return ops;
}
