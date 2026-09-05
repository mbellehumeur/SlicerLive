/**
 * Cast join for IRA — deep-link ?topic=&id-token= like OHIF CastService / worklist viewers.
 * Subscribes as Image Display (ID), answers status-request, emits status-request after
 * subscribe to catch up on an already-open worklist study, and forwards ImagingStudy-open.
 */
import CastClient, {
  castRadioStatusVisual,
  generateSubscriberName,
  isRequestEvent,
  isRunningInCloud,
  requestEventFor,
  resolveCastHubAdminUrl,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";

const LOG = "[ira-cast]";

const HUB_CREDENTIALS = {
  client_id: "client_id_3d_Slicer",
  client_secret: "client_secret_3d_Slicer",
} as const;

/** Standalone / local demos default (localhost Cast hub). */
const LOCAL_HUB = {
  name: "local",
  version: "1.0",
  hub_endpoint: "http://127.0.0.1:2018/api/hub",
  authorization_endpoint: "http://127.0.0.1:2018/oauth/authorize",
  token_endpoint: "http://127.0.0.1:2018/oauth/token",
  ...HUB_CREDENTIALS,
};

type HubConfig = typeof LOCAL_HUB;

/** True when served from the Cast hub `/slicerlive` SPA mount. */
function isHostedOnCastHub(
  location: Location = globalThis.location
): boolean {
  return String(location?.pathname || "").startsWith("/slicerlive");
}

function shouldUseSameOriginHub(
  location: Location = globalThis.location
): boolean {
  return isHostedOnCastHub(location) || isRunningInCloud(location);
}

/** Local hub for demos; same-origin endpoints when hosted on hub / in cloud. */
function resolveHubConfig(
  location: Location = globalThis.location
): HubConfig {
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

const EMPTY_CAST_CONTEXT = { "context.type": "", context: [] as unknown[] };
const ID_ACTOR = "ID";
const WORKLIST_ACTOR = "WORKLIST_CLIENT";
const SS_TOPIC = "ira.cast.topic";
const SS_TOKEN = "ira.cast.token";

/** Query keys that select a study; stripped on close / replaced on open. */
const STUDY_URL_KEYS = [
  "series",
  "bucket",
  "seg",
  "segBucket",
  "patient",
  "modality",
  "collection",
  "scene",
  "StudyInstanceUIDs",
  "SeriesInstanceUIDs",
  "studyUID",
  "seriesUID",
  "requestType",
] as const;

export type IraCastConnection =
  | "idle"
  | "connecting"
  | "connected"
  | "disconnected"
  | "error";

type StatusListener = (connection: IraCastConnection, detail: string) => void;
type ImagingStudyOpenHandler = (context: unknown[]) => void | Promise<void>;
type ImagingStudyCloseHandler = () => void | Promise<void>;

let client: ReturnType<typeof CastClient.newInstance> | null = null;
let topic = "";
let subscriberName = "";
let connection: IraCastConnection = "idle";
let connectionDetail = "Not connected";
let lastImagingStudyOpenContext: unknown[] = [];
/** idle → not attempted; pending → in flight; done → finished or intentionally skipped */
let statusRequestState: "idle" | "pending" | "done" = "idle";
let onStatus: StatusListener | null = null;
let onImagingStudyOpen: ImagingStudyOpenHandler | null = null;
let onImagingStudyClose: ImagingStudyCloseHandler | null = null;

export function setIraCastStatusListener(fn: StatusListener): void {
  onStatus = fn;
}

/** Register handler for inbound ImagingStudy-open (empty-start Cast load). */
export function setIraImagingStudyOpenHandler(
  fn: ImagingStudyOpenHandler | null
): void {
  onImagingStudyOpen = fn;
}

/** Register handler for inbound ImagingStudy-close (unload / return to empty). */
export function setIraImagingStudyCloseHandler(
  fn: ImagingStudyCloseHandler | null
): void {
  onImagingStudyClose = fn;
}

function persistCastCredentials(nextTopic: string, token: string): void {
  try {
    if (nextTopic) sessionStorage.setItem(SS_TOPIC, nextTopic);
    if (token) sessionStorage.setItem(SS_TOKEN, token);
  } catch {
    /* private mode / quota */
  }
}

function readPersistedCredentials(): { topic: string; token: string } {
  try {
    return {
      topic: sessionStorage.getItem(SS_TOPIC)?.trim() || "",
      token: sessionStorage.getItem(SS_TOKEN)?.trim() || "",
    };
  } catch {
    return { topic: "", token: "" };
  }
}

/** True when the URL still names a study (direct-S3, IID, or SlicerLive scene). */
export function iraUrlHasStudyParams(): boolean {
  const p = new URLSearchParams(globalThis.location.search);
  return Boolean(
    p.get("series") ||
      p.get("StudyInstanceUIDs") ||
      p.get("studyUID") ||
      p.get("scene")
  );
}

/**
 * Reload IRA preserving Cast topic/token, optionally with new study params.
 * Used for ImagingStudy-open / -close so the scene fully resets without tear-down.
 */
export function assignIraLocation(studyParams: Record<string, string> = {}): void {
  const url = new URL(globalThis.location.href);
  for (const key of STUDY_URL_KEYS) url.searchParams.delete(key);

  const persisted = readPersistedCredentials();
  const nextTopic = topic || persisted.topic;
  const nextToken =
    persisted.token ||
    String(client?.getHubConfig?.()?.token || "").trim();

  if (nextTopic) url.searchParams.set("topic", nextTopic);
  if (nextToken) url.searchParams.set("id-token", nextToken);

  for (const [key, value] of Object.entries(studyParams)) {
    const v = String(value || "").trim();
    if (v) url.searchParams.set(key, v);
  }

  globalThis.location.assign(url.toString());
}

export function getIraCastSessionInfo(): {
  topic: string;
  subscriberName: string;
  connection: IraCastConnection;
  connectionDetail: string;
  hubEndpoint: string;
} {
  return {
    topic,
    subscriberName:
      subscriberName || client?.getSessionConfig?.()?.subscriberName || "",
    connection,
    connectionDetail,
    hubEndpoint: client?.getHubConfig?.()?.hub_endpoint || resolveHubConfig().hub_endpoint,
  };
}

export function getIraHubAdminUrl(): string {
  const endpoint =
    client?.getHubConfig?.()?.hub_endpoint || resolveHubConfig().hub_endpoint;
  return resolveCastHubAdminUrl(endpoint) || endpoint;
}

export function setIraImagingStudyContext(context: unknown[]): void {
  lastImagingStudyOpenContext = Array.isArray(context) ? context : [];
}

function setConnection(next: IraCastConnection, detail: string): void {
  connection = next;
  connectionDetail = detail;
  onStatus?.(next, detail);
}

function stripCastParamsFromUrl(): void {
  if (!globalThis.history?.replaceState) return;
  const url = new URL(globalThis.location.href);
  if (!url.searchParams.has("topic") && !url.searchParams.has("id-token")) {
    return;
  }
  url.searchParams.delete("topic");
  url.searchParams.delete("id-token");
  globalThis.history.replaceState(null, "", url.toString());
}

function getHubEventLower(event: { "hub.event"?: string } | undefined): string {
  return String(event?.["hub.event"] || "")
    .trim()
    .toLowerCase();
}

function extractActorKeywords(actor: unknown): string[] {
  if (actor == null) return [];
  if (typeof actor === "string") {
    const keyword = actor.trim();
    return keyword ? [keyword] : [];
  }
  if (Array.isArray(actor)) {
    return actor.flatMap((entry) => extractActorKeywords(entry));
  }
  if (typeof actor === "object") {
    const obj = actor as { id?: unknown; key?: unknown; name?: unknown };
    const candidate = obj.id || obj.key || obj.name;
    if (candidate == null) return [];
    const keyword = String(candidate).trim();
    return keyword ? [keyword] : [];
  }
  const keyword = String(actor).trim();
  return keyword ? [keyword] : [];
}

function cloneContextArray(context: unknown): unknown[] {
  if (!Array.isArray(context)) return [];
  try {
    return structuredClone(context);
  } catch {
    return JSON.parse(JSON.stringify(context));
  }
}

function messageEventContext(message: Record<string, unknown>): unknown[] {
  const event = message?.event as { context?: unknown } | undefined;
  if (Array.isArray(event?.context)) return event.context as unknown[];
  return [];
}

function handleStatusRequest(message: Record<string, unknown>): boolean {
  const event = message?.event as
    | {
        "hub.event"?: string;
        context?: Record<string, unknown>;
        "hub.topic"?: string;
      }
    | undefined;
  if (!event || !isRequestEvent(getHubEventLower(event))) return false;

  const context =
    event.context && typeof event.context === "object" ? event.context : {};
  const correlationId = context.id;
  if (typeof correlationId !== "string" || !correlationId) return false;

  const requestedTargets = extractActorKeywords(message["target.actor"]);
  if (
    requestedTargets.length > 0 &&
    !requestedTargets.includes("*") &&
    !requestedTargets.includes(ID_ACTOR)
  ) {
    return false;
  }
  if (context.dataType !== "STATUS") return false;

  const responseData = lastImagingStudyOpenContext.length
    ? {
        "context.type": "ImagingStudy",
        context: cloneContextArray(lastImagingStudyOpenContext),
      }
    : EMPTY_CAST_CONTEXT;

  client?.sendCastRequestResponse(
    correlationId,
    "STATUS",
    responseData,
    event["hub.topic"]
  );
  return true;
}

function handleImagingStudyMessage(message: Record<string, unknown>): boolean {
  const event = message?.event as { "hub.event"?: string } | undefined;
  const name = getHubEventLower(event);
  if (name === "imagingstudy-close") {
    lastImagingStudyOpenContext = [];
    console.info(`${LOG} ImagingStudy-close`);
    void Promise.resolve(onImagingStudyClose?.()).catch((err) =>
      console.error(`${LOG} ImagingStudy-close handler failed`, err)
    );
    return true;
  }
  if (name !== "imagingstudy-open") return false;

  const context = messageEventContext(message);
  lastImagingStudyOpenContext = cloneContextArray(context);
  console.info(`${LOG} ImagingStudy-open context items=`, context.length);
  void Promise.resolve(onImagingStudyOpen?.(lastImagingStudyOpenContext)).catch(
    (err) => console.error(`${LOG} ImagingStudy-open handler failed`, err)
  );
  return true;
}

type StatusResponseItem = {
  actor?: unknown;
  data?: {
    "context.type"?: unknown;
    context?: unknown;
  };
};

/**
 * Ask the worklist for current STATUS once the websocket is connected.
 * Always sends so the worklist can mark this Image Display connected.
 * If it reports an open ImagingStudy and we are not already loading one, open it.
 */
async function requestStatusFromWorklist(): Promise<void> {
  if (!client) {
    console.warn(`${LOG} STATUS skip: no client`);
    return;
  }
  if (statusRequestState !== "idle") return;

  const skipApplyStudy =
    iraUrlHasStudyParams() || lastImagingStudyOpenContext.length > 0;
  if (skipApplyStudy) {
    console.warn(
      `${LOG} STATUS presence only (already have study URL or context)`
    );
  }

  const subscriber =
    subscriberName || client.getSessionConfig()?.subscriberName?.trim() || "";
  if (!subscriber) {
    console.warn(`${LOG} STATUS skip: no subscriber name yet`);
    return;
  }

  statusRequestState = "pending";
  const statusDataType = "STATUS";

  try {
    const message: Record<string, unknown> = {
      "subscriber.name": subscriber,
      "subscriber.product.name": "SLICERLIVE-IRA",
      "subscriber.actor": ID_ACTOR,
      "target.actor": WORKLIST_ACTOR,
      event: {
        "hub.event": requestEventFor(statusDataType),
        ...(topic ? { "hub.topic": topic } : {}),
        context: { dataType: statusDataType },
      },
    };

    console.warn(`${LOG} STATUS request → ${WORKLIST_ACTOR}`, {
      subscriber,
      topic,
      presenceOnly: skipApplyStudy,
    });
    const result = await client.request(message);
    console.warn(`${LOG} STATUS response`, {
      httpOk: result.ok,
      status: result.status,
      data: result.data,
    });

    if (!result.ok) {
      statusRequestState = "idle";
      return;
    }

    // Presence ping done; do not re-open a study we already have from URL/open.
    if (skipApplyStudy || lastImagingStudyOpenContext.length || iraUrlHasStudyParams()) {
      statusRequestState = "done";
      return;
    }

    const envelope =
      result.data && typeof result.data === "object"
        ? (result.data as { responses?: StatusResponseItem[] })
        : {};
    const responses = Array.isArray(envelope.responses) ? envelope.responses : [];

    let chosen: StatusResponseItem["data"] | undefined;
    for (const item of responses) {
      const itemData = item?.data;
      if (
        !itemData ||
        typeof itemData !== "object" ||
        itemData["context.type"] !== "ImagingStudy"
      ) {
        continue;
      }
      const actor = String(item?.actor ?? "")
        .trim()
        .toUpperCase();
      if (actor === WORKLIST_ACTOR || chosen === undefined) {
        chosen = itemData;
        if (actor === WORKLIST_ACTOR) break;
      }
    }

    if (!chosen) {
      console.warn(
        `${LOG} STATUS: no ImagingStudy in ${responses.length} response(s)`
      );
      // Allow a later retry (e.g. worklist stored context after we asked).
      statusRequestState = "idle";
      return;
    }

    if (lastImagingStudyOpenContext.length || iraUrlHasStudyParams()) {
      statusRequestState = "done";
      return;
    }

    const context = cloneContextArray(chosen.context);
    if (!context.length) {
      console.warn(`${LOG} STATUS: ImagingStudy context empty`);
      statusRequestState = "idle";
      return;
    }

    lastImagingStudyOpenContext = context;
    statusRequestState = "done";
    console.warn(
      `${LOG} ImagingStudy from STATUS context items=`,
      context.length
    );
    void Promise.resolve(onImagingStudyOpen?.(lastImagingStudyOpenContext)).catch(
      (err) =>
        console.error(`${LOG} STATUS ImagingStudy-open handler failed`, err)
    );
  } catch (err) {
    statusRequestState = "idle";
    console.error(`${LOG} failed to request STATUS`, err);
  }
}

/** Schedule STATUS catch-up after the bind socket is up (hub registers on handshake). */
function scheduleStatusRequestFromWorklist(delayMs: number): void {
  globalThis.setTimeout(() => {
    void requestStatusFromWorklist();
  }, delayMs);
}

/** Apply radio visual classes for the current connection. */
export function applyIraCastRadioVisual(statusEl: HTMLElement): void {
  const visual = castRadioStatusVisual(connection);
  statusEl.classList.remove(
    "ira-cast-idle",
    "ira-cast-connecting",
    "ira-cast-connected",
    "ira-cast-disconnected",
    "ira-cast-error",
    "ira-cast-pulse"
  );
  statusEl.classList.add(`ira-cast-${visual.tone}`);
  statusEl.classList.toggle("ira-cast-pulse", visual.pulse);
  statusEl.title = connectionDetail;
  const slash = statusEl.querySelector(".cast-radio-slash") as HTMLElement | null;
  if (slash) slash.hidden = !visual.showSlash;
}

/**
 * Join Cast when worklist deep-link params are present.
 * No-op (idle) when neither topic nor id-token is set.
 */
export async function connectIraCast(): Promise<void> {
  const params = new URLSearchParams(globalThis.location.search);
  topic = params.get("topic")?.trim() || "";
  let urlToken = params.get("id-token")?.trim() || "";

  if (!topic && !urlToken) {
    setConnection("idle", "Not connected");
    return;
  }

  stripCastParamsFromUrl();
  setConnection("connecting", "Connecting to Cast hub…");

  subscriberName = generateSubscriberName("IRA");
  const hub = resolveHubConfig();
  client = CastClient.newInstance({
    hub,
    session: {
      subscriberName,
      productName: "SLICERLIVE-IRA",
      productVersion: "0.1",
      actors: [ID_ACTOR],
      topic: topic || undefined,
      events: [
        "imagingstudy-open",
        "imagingstudy-close",
        "status-request",
        "subscription-removed",
        "status-update",
      ],
      lease: 7200,
      defaultTargetActor: "*",
    },
    callbackUrl: `${globalThis.location.origin}/castCallback`,
    preserveSessionTopicFromToken: Boolean(topic),
    autoReconnect: true,
  });

  client.onMessage((message: Record<string, unknown>) => {
    if (handleStatusRequest(message)) return;
    handleImagingStudyMessage(message);
  });

  client.onConnectionStateChange((wsState: string) => {
    console.warn(`${LOG} ws state=`, wsState);
    if (wsState === "connecting") {
      setConnection("connecting", "Websocket connecting");
    } else if (wsState === "connected") {
      setConnection("connected", "Websocket connected");
      // Delay so the hub has registered this bind socket before /request.
      scheduleStatusRequestFromWorklist(150);
    } else if (wsState === "error") {
      statusRequestState = "idle";
      setConnection("error", "Websocket error");
    } else {
      statusRequestState = "idle";
      setConnection("disconnected", "Websocket disconnected");
    }
  });

  try {
    if (urlToken) {
      client.setToken(urlToken);
    } else {
      const { code } = await client.authenticate();
      if (!code || !(await client.getToken(code))) {
        setConnection("error", "Token failed");
        return;
      }
    }
    if (topic) client.setTopic(topic);

    // Persist early so ImagingStudy-open/close reloads can rebuild the deep-link
    // even if the message arrives before subscribe() resolves.
    const hubTokenEarly = String(
      client.getHubConfig?.()?.token || urlToken || ""
    ).trim();
    if (hubTokenEarly) urlToken = hubTokenEarly;
    persistCastCredentials(topic, urlToken);

    const status = await client.subscribe();
    const session = client.getSessionConfig();
    if (session.subscriberName) subscriberName = session.subscriberName;
    if (session.topic) topic = session.topic;

    const hubToken = String(client.getHubConfig?.()?.token || urlToken || "").trim();
    if (hubToken) urlToken = hubToken;
    persistCastCredentials(topic, urlToken);

    if (status === 202) {
      console.warn(
        `${LOG} subscribed topic=`,
        topic,
        "subscriber=",
        subscriberName
      );
      // Backup if onopen already fired before the handler was useful, or was missed.
      scheduleStatusRequestFromWorklist(400);
    } else {
      setConnection("disconnected", `Subscribe failed (${String(status)})`);
    }
  } catch (err) {
    console.error(`${LOG} connect failed`, err);
    setConnection(
      "error",
      err instanceof Error ? err.message : String(err)
    );
  }
}
