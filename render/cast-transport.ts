// CastTransport — LiveSync Transport that carries opaque mrson/LiveSync JSON over Cast
// ``scene-update`` instead of ``ws://:2132``. Bulk blobs travel separately via hub payloads.
import type { Transport } from "./transport.ts";

export const LIVESYNC_CONTEXT_KEY = "livesync";

export type CastSceneUpdatePublisher = {
  publishSceneUpdate: (wire: unknown) => void | Promise<void>;
  isConnected: () => boolean;
};

/** Encode a LiveSync wire message into a Cast scene-update context array. */
export function livesyncContextFromWire(wire: unknown): unknown[] {
  return [{ key: LIVESYNC_CONTEXT_KEY, resource: wire }];
}

/** Extract a LiveSync wire message from Cast scene-update context (array or dict). */
export function wireFromSceneUpdateContext(context: unknown): unknown | null {
  if (Array.isArray(context)) {
    for (const item of context) {
      if (!item || typeof item !== "object") continue;
      const row = item as { key?: unknown; resource?: unknown };
      if (String(row.key || "").toLowerCase() !== LIVESYNC_CONTEXT_KEY) continue;
      if (row.resource != null) return row.resource;
    }
    return null;
  }
  if (context && typeof context === "object") {
    const obj = context as Record<string, unknown>;
    if (obj[LIVESYNC_CONTEXT_KEY] != null) return obj[LIVESYNC_CONTEXT_KEY];
    if (obj.livesync != null) return obj.livesync;
  }
  return null;
}

/**
 * LiveSync transport over Cast scene-update.
 * Call {@link deliver} from the Cast message handler when an inbound livesync payload arrives.
 */
export class CastTransport implements Transport {
  onMessage?: (msg: unknown) => void;
  onOpen?: () => void;
  onClose?: () => void;

  private open = false;
  private closed = false;

  constructor(private readonly cast: CastSceneUpdatePublisher) {}

  get isOpen(): boolean {
    return this.open && this.cast.isConnected();
  }

  connect(): void {
    this.closed = false;
    if (this.cast.isConnected()) {
      this.open = true;
      this.onOpen?.();
      return;
    }
    this.open = false;
  }

  /** Notify LiveSync that the Cast bind socket is ready (call after subscribe succeeds). */
  notifyOpen(): void {
    if (this.closed) return;
    this.open = true;
    this.onOpen?.();
  }

  /** Notify LiveSync that Cast disconnected. */
  notifyClose(): void {
    const wasOpen = this.open;
    this.open = false;
    if (wasOpen) this.onClose?.();
  }

  /** Deliver one inbound LiveSync wire message from a Cast scene-update. */
  deliver(wire: unknown): void {
    if (this.closed || !this.open) return;
    this.onMessage?.(wire);
  }

  send(msg: unknown): void {
    if (!this.isOpen) return;
    void Promise.resolve(this.cast.publishSceneUpdate(msg)).catch((err) => {
      console.warn("[cast-transport] scene-update publish failed", err);
    });
  }

  close(): void {
    this.closed = true;
    this.open = false;
  }
}
