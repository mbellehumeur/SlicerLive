/**
 * Cast session for hub-mirror — deep-link ?topic=&id-token=, scene-update LiveSync.
 * Bulk chunks arrive via HubBlobs on scene-update (payloadIds → GET /api/hub/payloads).
 * Does not subscribe to imagingstudy-open / imagingstudy-close.
 */
import CastClient, {
  batchContextFiles,
  generateSubscriberName,
  getHubEventLower,
  isRunningInCloud,
  messageEventContext,
  messageSubscriberName,
  resolveCastFileMessage,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import {
  CastTransport,
  livesyncContextFromWire,
  wireFromSceneUpdateContext,
} from "../../cast-transport.ts";
import { setBlobFetch } from "../../zarr.ts";

const LOG = "[hub-mirror-cast]";

const HUB_CREDENTIALS = {
  client_id: "client_id_3d_Slicer",
  client_secret: "client_secret_3d_Slicer",
} as const;

const LOCAL_HUB = {
  name: "local",
  version: "1.0",
  hub_endpoint: "http://127.0.0.1:2018/api/hub",
  authorization_endpoint: "http://127.0.0.1:2018/oauth/authorize",
  token_endpoint: "http://127.0.0.1:2018/oauth/token",
  ...HUB_CREDENTIALS,
};

const ID_ACTOR = "ID";
const PRODUCT_NAME = "SLICERLIVE-HUB-MIRROR";
const HUB_BLOBS_EVENT = "HubBlobs";

/**
 * True when this page is served from the Cast hub `/hub-mirror/` mount.
 * Local demo servers (e.g. :8130/.../hub-mirror/) must NOT match — they have no /api/hub.
 */
function isServedFromCastHubMount(
  location: Location = globalThis.location
): boolean {
  const path = String(location?.pathname || "");
  if (!path.startsWith("/hub-mirror")) return false;
  if (isRunningInCloud(location)) return true;
  const port =
    location.port ||
    (location.protocol === "https:" ? "443" : "80");
  return port === "2018";
}

function shouldUseSameOriginHub(
  location: Location = globalThis.location
): boolean {
  return isServedFromCastHubMount(location) || isRunningInCloud(location);
}

/** Local hub for demos; same-origin only when mounted on Cast hub / in cloud. */
function resolveHubConfig(location: Location = globalThis.location) {
  if (!shouldUseSameOriginHub(location)) {
    return LOCAL_HUB;
  }
  const origin = location.origin;
  return {
    name: "hub",
    version: "1.0",
    hub_endpoint: `${origin}/api/hub`,
    authorization_endpoint: `${origin}/oauth/authorize`,
    token_endpoint: `${origin}/oauth/token`,
    ...HUB_CREDENTIALS,
  };
}

export type HubMirrorCast = {
  transport: CastTransport;
  client: ReturnType<typeof CastClient.newInstance>;
  topic: string;
  subscriberName: string;
  ready: Promise<void>;
  close: () => void;
};

/**
 * Install a blob fetch that serves hub-downloaded chunk files by content hash.
 * Never hits the network for the synthetic hub-mirror.local base (avoids DNS errors).
 */
export function installHubBlobFetch(hashToBytes: Map<string, Uint8Array>): void {
  setBlobFetch(async (url: string) => {
    const u = String(url || "");
    let hash = "";
    try {
      const path = new URL(u, "http://local/").pathname;
      const base = path.split("/").pop() || "";
      if (base.startsWith("sha256-")) {
        hash = base.replace(/\.bin$/i, "");
      }
    } catch {
      /* ignore */
    }
    if (!hash && u.startsWith("sha256-")) hash = u.replace(/\.bin$/i, "");
    if (!hash) {
      const m = u.match(/(sha256-[a-fA-F0-9]+)/);
      if (m) hash = m[1];
    }
    const bytes = hash ? hashToBytes.get(hash) : undefined;
    if (bytes) {
      return new Response(bytes, {
        status: 200,
        headers: { "Content-Type": "application/octet-stream" },
      });
    }
    const synthetic =
      u.includes("hub-mirror.local") || (hash && hash.startsWith("sha256-"));
    if (synthetic) {
      console.warn(
        `${LOG} missing blob cache entry for`,
        hash || u,
        `(cached=${hashToBytes.size})`
      );
      return new Response(null, { status: 404, statusText: "Blob not in hub cache" });
    }
    return fetch(url);
  });
}

/** Map transfer fileName (possibly ``sha256-….bin``) to content-hash key used in chunkHashes. */
function contentHashFromFileName(fileName: string): string {
  let name = String(fileName || "").trim();
  if (name.toLowerCase().endsWith(".bin")) {
    name = name.slice(0, -4);
  }
  return name;
}

function ingestBlobFiles(
  hashToBytes: Map<string, Uint8Array>,
  files: Array<{ fileName?: string; data?: ArrayBuffer | Uint8Array | string }>
): number {
  let n = 0;
  for (const entry of files) {
    const hash = contentHashFromFileName(String(entry.fileName || ""));
    if (!hash.startsWith("sha256-")) continue;
    const data = entry.data;
    let bytes: Uint8Array | null = null;
    if (data instanceof ArrayBuffer) bytes = new Uint8Array(data);
    else if (data instanceof Uint8Array) bytes = data;
    if (!bytes || !bytes.byteLength) continue;
    hashToBytes.set(hash, bytes);
    n++;
  }
  return n;
}

function isHubBlobsWire(wire: unknown): wire is {
  event: string;
  files?: Array<Record<string, unknown>>;
} {
  if (!wire || typeof wire !== "object") return false;
  const ev = String((wire as { event?: unknown }).event || "").trim();
  return ev === HUB_BLOBS_EVENT;
}

async function sleep(ms: number): Promise<void> {
  await new Promise((r) => globalThis.setTimeout(r, ms));
}

/** Fetch hub payloads with backoff — payloads may still be registering. */
async function fetchHubBlobFilesWithRetry(
  client: ReturnType<typeof CastClient.newInstance>,
  files: Array<Record<string, unknown>>,
  opts: { attempts?: number; baseDelayMs?: number } = {}
): Promise<
  Array<{ fileName?: string; data?: ArrayBuffer | Uint8Array | string }>
> {
  const attempts = opts.attempts ?? 8;
  const baseDelayMs = opts.baseDelayMs ?? 400;
  const pending = files.filter((f) => {
    const ids = f.payloadIds;
    return Array.isArray(ids) && ids.length > 0 && f.data == null;
  });
  if (!pending.length) return files as Array<{ fileName?: string; data?: ArrayBuffer | Uint8Array | string }>;

  let lastErr: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      const synthetic = {
        event: {
          "hub.event": "ImagingStudy-open",
          context: { files: pending.map((f) => ({ ...f })) },
        },
      };
      const resolved = (await resolveCastFileMessage(
        client,
        synthetic
      )) as Record<string, unknown>;
      const event = (resolved.event || synthetic.event) as Record<string, unknown>;
      return batchContextFiles(event) as Array<{
        fileName?: string;
        data?: ArrayBuffer | Uint8Array | string;
      }>;
    } catch (err) {
      lastErr = err;
      const delay = baseDelayMs * Math.pow(2, Math.min(i, 4));
      console.warn(
        `${LOG} payload fetch attempt ${i + 1}/${attempts} failed; retry in ${delay}ms`,
        err
      );
      await sleep(delay);
    }
  }
  throw lastErr instanceof Error
    ? lastErr
    : new Error(String(lastErr || "payload fetch failed"));
}

/**
 * Connect to Cast hub and return a CastTransport ready for LiveSync.
 * Requires ?topic= and ?id-token= (worklist deep-link).
 */
export async function connectHubMirrorCast(): Promise<HubMirrorCast> {
  const params = new URLSearchParams(globalThis.location.search);
  let topic =
    params.get("topic")?.trim() ||
    sessionStorage.getItem("hub-mirror.cast.topic")?.trim() ||
    "";
  let token =
    params.get("id-token")?.trim() ||
    sessionStorage.getItem("hub-mirror.cast.token")?.trim() ||
    "";
  if (!topic || !token) {
    throw new Error(
      "hub-mirror requires ?topic= and ?id-token= (open from worklist 3D Slicer web mirror button)"
    );
  }
  try {
    sessionStorage.setItem("hub-mirror.cast.topic", topic);
    sessionStorage.setItem("hub-mirror.cast.token", token);
  } catch {
    /* private mode */
  }

  try {
    const url = new URL(globalThis.location.href);
    url.searchParams.delete("id-token");
    globalThis.history.replaceState({}, "", url.toString());
  } catch {
    /* ignore */
  }

  const hub = resolveHubConfig();
  const subscriberName = generateSubscriberName("HubMirror");
  const hashToBytes = new Map<string, Uint8Array>();
  installHubBlobFetch(hashToBytes);

  let resolveReady!: () => void;
  let readySettled = false;
  const ready = new Promise<void>((r) => {
    resolveReady = () => {
      if (readySettled) return;
      readySettled = true;
      r();
    };
  });

  const client = CastClient.newInstance({
    hub,
    session: {
      subscriberName,
      productName: PRODUCT_NAME,
      productVersion: "0.1",
      actors: [ID_ACTOR],
      topic: topic || undefined,
      // Scene control + HubBlobs payload refs only — no imagingstudy-* subscription.
      events: ["scene-update"],
      lease: 7200,
      defaultTargetActor: "ID",
    },
    callbackUrl: `${globalThis.location.origin}/castCallback`,
    preserveSessionTopicFromToken: Boolean(topic),
    autoReconnect: true,
  });

  let castWsConnected = false;
  const transport = new CastTransport({
    isConnected: () => castWsConnected,
    publishSceneUpdate: async (wire: unknown) => {
      await client.publish({
        event: {
          "hub.topic": topic,
          "hub.event": "scene-update",
          context: livesyncContextFromWire(wire),
        },
        actor: ID_ACTOR,
        "target.actor": "ID",
        "target.product.name": "3DSLICER-ID",
      });
    },
  });

  /** Buffer livesync until HubBlobs ingest finishes (or empty-scene timeout). */
  let blobsReady = false;
  let ingestInFlight = false;
  const pendingWire: unknown[] = [];
  const flushPending = () => {
    blobsReady = true;
    for (const w of pendingWire.splice(0)) transport.deliver(w);
  };
  // Only release early if no HubBlobs ingest ever started (empty scene / export skip).
  globalThis.setTimeout(() => {
    if (!blobsReady && !ingestInFlight) {
      console.info(`${LOG} blob wait timeout — releasing livesync buffer`);
      flushPending();
    }
  }, 60000);

  client.onMessage(async (message: Record<string, unknown>) => {
    const eventName = getHubEventLower(message?.event);
    const from = messageSubscriberName(message);
    if (from && from === subscriberName) return;

    if (eventName !== "scene-update") return;

    const eventObj = (message.event || {}) as Record<string, unknown>;
    const rawContext = eventObj.context;
    const contextFiles = batchContextFiles(eventObj) as Array<Record<string, unknown>>;
    // Prefer full context (array or dict). messageEventContext historically returned [] for dicts.
    const wire = wireFromSceneUpdateContext(
      rawContext != null ? rawContext : messageEventContext(message)
    );
    const hubBlobs =
      isHubBlobsWire(wire) ||
      contextFiles.some((f) => {
        const ids = f.payloadIds;
        return Array.isArray(ids) && ids.length > 0;
      });

    if (hubBlobs) {
      ingestInFlight = true;
      try {
        const filesMeta =
          contextFiles.length > 0
            ? contextFiles
            : Array.isArray((wire as { files?: unknown })?.files)
              ? ((wire as { files: Array<Record<string, unknown>> }).files)
              : [];
        console.info(
          `${LOG} HubBlobs received — fetching ${filesMeta.length} payload(s)…`
        );
        if (!filesMeta.length) {
          flushPending();
          return;
        }
        const files = await fetchHubBlobFilesWithRetry(client, filesMeta);
        const n = ingestBlobFiles(hashToBytes, files);
        console.info(`${LOG} HubBlobs cached`, n);
        flushPending();
      } catch (err) {
        console.error(`${LOG} HubBlobs resolve failed`, err);
        flushPending();
      } finally {
        ingestInFlight = false;
      }
      return;
    }

    if (wire == null) return;

    const wireEvent = String((wire as { event?: unknown }).event || "");
    if (wireEvent === "Snapshot") {
      const nodes = (wire as { scene?: { nodes?: Record<string, unknown> } }).scene
        ?.nodes;
      console.info(
        `${LOG} Snapshot received nodes=`,
        nodes ? Object.keys(nodes).length : 0,
        blobsReady ? "(deliver)" : "(buffer until HubBlobs)"
      );
    }

    if (!blobsReady) {
      pendingWire.push(wire);
      return;
    }
    transport.deliver(wire);
  });

  client.onConnectionStateChange((wsState: string) => {
    console.info(`${LOG} ws state=`, wsState);
    if (wsState === "connected") {
      castWsConnected = true;
      transport.notifyOpen();
      resolveReady();
    } else if (wsState === "error" || wsState === "disconnected") {
      castWsConnected = false;
      transport.notifyClose();
    }
  });

  client.setToken(token);
  if (topic) client.setTopic(topic);
  const subStatus = await client.subscribe();
  console.info(`${LOG} subscribe status=`, subStatus);

  return {
    transport,
    client,
    topic,
    subscriberName,
    ready,
    close: () => {
      transport.close();
      void client.unsubscribe?.();
    },
  };
}

/**
 * Absolute synthetic httpBase for LiveScene.blobBase() (`new URL("blobs/", httpBase)`).
 * Bytes are served from the hub payload cache via setBlobFetch (hash in the URL path).
 */
export const HUB_MIRROR_BLOB_BASE = "http://hub-mirror.local/mrson/";
