/**
 * Ephemeral freehand “vanishing marker” strokes for IRA conference.
 * Published on Cast scene-update as cmd ops; never written into the durable scene node map.
 */

import type { Op } from "../../liveops.ts";
import type { Vec3 } from "../../mat4.ts";

export const VANISHING_NODE_ID = "_vanishing";
export const VANISHING_CMD = "vanishingStroke";

export const DEFAULT_VANISHING_TTL_MS = 3000;
export const DEFAULT_VANISHING_FADE_MS = 1000;
export const DEFAULT_VANISHING_WIDTH_PX = 3;
export const DEFAULT_VANISHING_COLOR: number[] = [1, 0.92, 0.2, 0.9];
/** Min interval between live "draw" publishes while the pointer moves. */
export const VANISHING_PUBLISH_INTERVAL_MS = 100;

export type VanishingPhase = "draw" | "end";

export type VanishingStrokeArgs = {
  strokeId: string;
  points: number[][];
  color: number[];
  widthPx?: number;
  phase: VanishingPhase;
  ttlMs?: number;
  author?: string;
  /** Slice plane the stroke was drawn on (optional; improves MPR overlay). */
  plane?: string;
  /** RAS mm along the plane normal when drawn. */
  mm?: number;
};

export type VanishingStrokeOp = {
  op: "cmd";
  id: typeof VANISHING_NODE_ID;
  cmd: typeof VANISHING_CMD;
  args: VanishingStrokeArgs;
};

export type VanishingStrokeView = {
  strokeId: string;
  points: Vec3[];
  color: number[];
  widthPx: number;
  /** null while still drawing; set on phase "end". */
  expiresAt: number | null;
  plane?: string;
  mm?: number;
  author?: string;
};

export function isVanishingStrokeOp(op: unknown): op is VanishingStrokeOp {
  if (!op || typeof op !== "object") return false;
  const o = op as Record<string, unknown>;
  if (o.op !== "cmd" || o.id !== VANISHING_NODE_ID || o.cmd !== VANISHING_CMD) {
    return false;
  }
  const args = o.args;
  if (!args || typeof args !== "object") return false;
  const a = args as Record<string, unknown>;
  return typeof a.strokeId === "string" && Array.isArray(a.points);
}

export function partitionVanishingOps(ops: Op[]): {
  vanishing: VanishingStrokeOp[];
  rest: Op[];
} {
  const vanishing: VanishingStrokeOp[] = [];
  const rest: Op[] = [];
  for (const op of ops) {
    if (isVanishingStrokeOp(op)) vanishing.push(op);
    else rest.push(op);
  }
  return { vanishing, rest };
}

/** Pull vanishingStroke cmds from a Cast scene-update context array. */
export function vanishingOpsFromSceneUpdateContext(
  context: unknown,
): VanishingStrokeOp[] {
  if (!Array.isArray(context)) return [];
  const out: VanishingStrokeOp[] = [];
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const row = item as { key?: unknown; resource?: unknown };
    const key = String(row.key || "").toLowerCase();
    if (key !== "scene" && key !== "sceneupdate") continue;
    const resource = row.resource;
    if (!resource || typeof resource !== "object") continue;
    const ops = (resource as { ops?: unknown }).ops;
    if (!Array.isArray(ops)) continue;
    for (const op of ops) {
      if (isVanishingStrokeOp(op)) out.push(op);
    }
  }
  return out;
}

/**
 * When not following the conference host, strip non-vanishing ops from scene-update
 * context so display/study patches are ignored but collaborative ink still arrives.
 */
export function filterSceneUpdateContextForVanishing(
  context: unknown[],
): unknown[] {
  const out: unknown[] = [];
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const row = item as { key?: unknown; resource?: unknown };
    const key = String(row.key || "").toLowerCase();
    if (key !== "scene" && key !== "sceneupdate") continue;
    const resource = row.resource;
    if (!resource || typeof resource !== "object") continue;
    const res = resource as Record<string, unknown>;
    const ops = Array.isArray(res.ops) ? res.ops.filter(isVanishingStrokeOp) : [];
    if (!ops.length) continue;
    out.push({
      key: row.key,
      resource: { ...res, ops },
    });
  }
  return out;
}

export function buildVanishingStrokeOp(
  args: VanishingStrokeArgs,
): VanishingStrokeOp {
  return {
    op: "cmd",
    id: VANISHING_NODE_ID,
    cmd: VANISHING_CMD,
    args: {
      strokeId: args.strokeId,
      points: args.points.map((p) => [
        Number(p[0]),
        Number(p[1]),
        Number(p[2]),
      ]),
      color: normalizeColor(args.color),
      widthPx:
        typeof args.widthPx === "number" && Number.isFinite(args.widthPx)
          ? Math.max(1, args.widthPx)
          : DEFAULT_VANISHING_WIDTH_PX,
      phase: args.phase === "end" ? "end" : "draw",
      ttlMs:
        typeof args.ttlMs === "number" && Number.isFinite(args.ttlMs)
          ? Math.max(0, args.ttlMs)
          : DEFAULT_VANISHING_TTL_MS,
      author: args.author,
      plane: args.plane,
      mm: typeof args.mm === "number" && Number.isFinite(args.mm) ? args.mm : undefined,
    },
  };
}

function normalizeColor(color: number[] | undefined): number[] {
  if (!Array.isArray(color) || color.length < 3) {
    return [...DEFAULT_VANISHING_COLOR];
  }
  const r = Number(color[0]);
  const g = Number(color[1]);
  const b = Number(color[2]);
  const a = color.length > 3 ? Number(color[3]) : DEFAULT_VANISHING_COLOR[3];
  return [
    Number.isFinite(r) ? r : DEFAULT_VANISHING_COLOR[0],
    Number.isFinite(g) ? g : DEFAULT_VANISHING_COLOR[1],
    Number.isFinite(b) ? b : DEFAULT_VANISHING_COLOR[2],
    Number.isFinite(a) ? a : DEFAULT_VANISHING_COLOR[3],
  ];
}

function asVec3(p: number[]): Vec3 | null {
  if (!Array.isArray(p) || p.length < 3) return null;
  const x = Number(p[0]);
  const y = Number(p[1]);
  const z = Number(p[2]);
  if (![x, y, z].every(Number.isFinite)) return null;
  return [x, y, z];
}

export function newStrokeId(): string {
  try {
    if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
      return crypto.randomUUID();
    }
  } catch {
    /* ignore */
  }
  return `vstroke-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** In-memory ephemeral stroke store with TTL fade. */
export class VanishingStrokeStore {
  private strokes = new Map<string, VanishingStrokeView>();
  private raf = 0;
  private onChange: (() => void) | null = null;
  private fadeMs: number;

  constructor(opts: { fadeMs?: number; onChange?: () => void } = {}) {
    this.fadeMs = opts.fadeMs ?? DEFAULT_VANISHING_FADE_MS;
    this.onChange = opts.onChange ?? null;
  }

  setOnChange(fn: (() => void) | null): void {
    this.onChange = fn;
  }

  clear(): void {
    this.strokes.clear();
    this.stopTick();
    this.onChange?.();
  }

  list(): VanishingStrokeView[] {
    return [...this.strokes.values()];
  }

  /** Alpha 0..1 for drawing (1 while live / early TTL; fades in the last fadeMs). */
  alphaOf(stroke: VanishingStrokeView, now = performance.now()): number {
    if (stroke.expiresAt == null) return 1;
    const remaining = stroke.expiresAt - now;
    if (remaining <= 0) return 0;
    if (remaining >= this.fadeMs) return 1;
    return Math.max(0, remaining / this.fadeMs);
  }

  applyOp(op: VanishingStrokeOp, now = performance.now()): boolean {
    const args = op.args;
    const points: Vec3[] = [];
    for (const p of args.points || []) {
      const v = asVec3(p);
      if (v) points.push(v);
    }
    if (!args.strokeId || points.length < 1) return false;

    const prev = this.strokes.get(args.strokeId);
    const color = normalizeColor(args.color ?? prev?.color);
    const widthPx =
      typeof args.widthPx === "number" && Number.isFinite(args.widthPx)
        ? Math.max(1, args.widthPx)
        : prev?.widthPx ?? DEFAULT_VANISHING_WIDTH_PX;

    let expiresAt: number | null = prev?.expiresAt ?? null;
    if (args.phase === "end") {
      const ttl =
        typeof args.ttlMs === "number" && Number.isFinite(args.ttlMs)
          ? Math.max(0, args.ttlMs)
          : DEFAULT_VANISHING_TTL_MS;
      expiresAt = now + ttl;
    } else if (args.phase === "draw") {
      // Keep drawing alive; clear any premature expiry from a race.
      expiresAt = null;
    }

    this.strokes.set(args.strokeId, {
      strokeId: args.strokeId,
      points,
      color,
      widthPx,
      expiresAt,
      plane: args.plane ?? prev?.plane,
      mm: typeof args.mm === "number" ? args.mm : prev?.mm,
      author: args.author ?? prev?.author,
    });
    this.onChange?.();
    this.ensureTick();
    return true;
  }

  applyOps(ops: VanishingStrokeOp[]): boolean {
    let any = false;
    for (const op of ops) {
      if (this.applyOp(op)) any = true;
    }
    return any;
  }

  private ensureTick(): void {
    if (this.raf) return;
    const step = () => {
      this.raf = 0;
      const now = performance.now();
      let changed = false;
      let needTick = false;
      for (const [id, s] of this.strokes) {
        if (s.expiresAt == null) {
          needTick = true;
          continue;
        }
        if (s.expiresAt <= now) {
          this.strokes.delete(id);
          changed = true;
          continue;
        }
        // Still fading or waiting to fade — keep ticking and redraw for alpha.
        needTick = true;
        if (s.expiresAt - now < this.fadeMs) changed = true;
      }
      if (changed) this.onChange?.();
      if (needTick) {
        this.raf = requestAnimationFrame(step);
      }
    };
    this.raf = requestAnimationFrame(step);
  }

  private stopTick(): void {
    if (this.raf) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
  }
}
