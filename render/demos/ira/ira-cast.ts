/**
 * Cast join for IRA — deep-link ?topic=&id-token= like OHIF CastService / worklist viewers.
 * Subscribes as Image Display (ID), answers status-request, emits status-request after
 * subscribe to catch up on an already-open worklist study, and forwards ImagingStudy-open.
 */
import CastClient, {
  castRadioStatusVisual,
  cloneContextArray,
  generateSubscriberName,
  getHubEventLower,
  isRequestEvent,
  isRunningInCloud,
  messageEventContext,
  parseStatusUpdateMessage,
  publishCastUrlSend,
  requestCastProductStatus,
  requestEventFor,
  resolveCastConferenceView,
  resolveCastFileMessage,
  resolveCastHubAdminUrl,
  batchContextFiles,
  type CastConferenceView,
  type CastProductStatusProbe,
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
let userName = "";
let connection: IraCastConnection = "idle";
let connectionDetail = "Not connected";
let conference: CastConferenceView | null = null;
/** When true, apply inbound ImagingStudy-open from conference host. */
let conferenceFollowHost = false;
let lastImagingStudyOpenContext: unknown[] = [];
/** Cached LiveScene document (blobBase + nodes) for STATUS + Scene View. */
let lastLiveScene: IraLiveSceneDoc | null = null;
/** idle → not attempted; pending → in flight; done → finished or intentionally skipped */
let statusRequestState: "idle" | "pending" | "done" = "idle";
let onStatus: StatusListener | null = null;
let onImagingStudyOpen: ImagingStudyOpenHandler | null = null;
let onImagingStudyClose: ImagingStudyCloseHandler | null = null;
let onLiveSceneChange: (() => void) | null = null;

export type IraLiveSceneDoc = {
  url?: string;
  mrson?: string;
  blobBase?: string;
  nodes?: Record<string, unknown>;
  [key: string]: unknown;
};

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
 * Update the address bar study params without navigating (keeps Cast WebSocket alive).
 * Used for ImagingStudy-open / -close and shareable deep links.
 */
export function syncIraStudyUrl(studyParams: Record<string, string> = {}): void {
  if (!globalThis.history?.replaceState) return;
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

  globalThis.history.replaceState(null, "", url.toString());
}

/**
 * @deprecated Prefer {@link syncIraStudyUrl} — kept as an alias (no page reload).
 */
export function assignIraLocation(studyParams: Record<string, string> = {}): void {
  syncIraStudyUrl(studyParams);
}

export function getIraCastSessionInfo(): {
  topic: string;
  subscriberName: string;
  userName: string;
  connection: IraCastConnection;
  connectionDetail: string;
  hubEndpoint: string;
  conference: CastConferenceView | null;
} {
  return {
    topic,
    subscriberName:
      subscriberName || client?.getSessionConfig?.()?.subscriberName || "",
    userName: userName || client?.getSessionConfig?.()?.userName || "",
    connection,
    connectionDetail,
    hubEndpoint: client?.getHubConfig?.()?.hub_endpoint || resolveHubConfig().hub_endpoint,
    conference,
  };
}

export function setIraConferenceView(view: CastConferenceView | null): void {
  conference = view;
  if (!view) {
    conferenceFollowHost = false;
  }
  onStatus?.(connection, connectionDetail);
}

export function setIraConferenceFollowHost(followHost: boolean): void {
  conferenceFollowHost = Boolean(followHost);
}

type ConferenceHubHandler = (message: Record<string, unknown>) => boolean;
let conferenceHubHandler: ConferenceHubHandler | null = null;

export function setIraConferenceHubHandler(
  fn: ConferenceHubHandler | null
): void {
  conferenceHubHandler = fn;
}

async function syncIraConferenceView(): Promise<void> {
  if (connection !== "connected") {
    setIraConferenceView(null);
    return;
  }
  try {
    const hubEndpoint =
      client?.getHubConfig?.()?.hub_endpoint || resolveHubConfig().hub_endpoint;
    const view = await resolveCastConferenceView(
      hubEndpoint,
      topic,
      subscriberName,
      userName
    );
    setIraConferenceView(view);
  } catch {
    /* keep prior */
  }
}

function handleConferenceHubEvent(message: Record<string, unknown>): boolean {
  if (conferenceHubHandler?.(message)) {
    return true;
  }
  const event = message?.event as
    | { "hub.event"?: string; context?: Record<string, unknown> }
    | undefined;
  const hubEvent = getHubEventLower(event);
  if (hubEvent === "conference-start") {
    void syncIraConferenceView();
    return true;
  }
  if (hubEvent === "conference-end") {
    const context =
      event?.context && typeof event.context === "object" ? event.context : {};
    const leaveTopic = String(context.leaveTopic ?? "").trim();
    if (!leaveTopic || leaveTopic === topic) {
      setIraConferenceView(null);
    } else {
      void syncIraConferenceView();
    }
    return true;
  }
  return false;
}

export function getIraHubAdminUrl(): string {
  const endpoint =
    client?.getHubConfig?.()?.hub_endpoint || resolveHubConfig().hub_endpoint;
  return resolveCastHubAdminUrl(endpoint) || endpoint;
}

export function setIraImagingStudyContext(context: unknown[]): void {
  lastImagingStudyOpenContext = Array.isArray(context) ? context : [];
}

export function getIraImagingStudyOpenContext(): unknown[] {
  return cloneContextArray(lastImagingStudyOpenContext);
}

/** Ignore hub echoes of ImagingStudy events we just published (avoid dispose/remount). */
let suppressInboundImagingStudy = 0;

function beginSuppressInboundImagingStudy(ms = 1500): void {
  suppressInboundImagingStudy += 1;
  globalThis.setTimeout(() => {
    suppressInboundImagingStudy = Math.max(0, suppressInboundImagingStudy - 1);
  }, ms);
}

function buildIraPublishPayload(hubEvent: string, context: unknown[]) {
  return {
    event: {
      "hub.topic": topic,
      "hub.event": hubEvent,
      context,
    },
    actor: ID_ACTOR,
    "target.actor": "*",
    "target.product.name": "*",
  };
}

/**
 * After a successful local (or other IRA-owned) load: publish ImagingStudy-close for the
 * previously open study (if any), then ImagingStudy-open for ``nextContext``.
 * Updates local open context. No-ops publish when not connected (still sets local context).
 */
export async function publishIraImagingStudyTransition(
  nextContext: unknown[]
): Promise<{ closed: boolean; opened: boolean }> {
  const next = Array.isArray(nextContext) ? cloneContextArray(nextContext) : [];
  const previous = cloneContextArray(lastImagingStudyOpenContext);

  if (!client || connection !== "connected" || !topic) {
    lastImagingStudyOpenContext = next;
    lastLiveScene = null;
    onLiveSceneChange?.();
    console.warn(`${LOG} ImagingStudy transition local-only (not connected)`);
    return { closed: false, opened: false };
  }

  beginSuppressInboundImagingStudy();
  let closed = false;
  let opened = false;
  try {
    if (previous.length) {
      try {
        const res = await client.publish(
          buildIraPublishPayload("ImagingStudy-close", previous)
        );
        closed = Boolean(res?.ok);
        if (!closed) {
          console.warn(
            `${LOG} ImagingStudy-close failed`,
            res ? `HTTP ${res.status}` : "No response"
          );
        } else {
          console.info(`${LOG} published ImagingStudy-close`);
        }
      } catch (err) {
        console.error(`${LOG} ImagingStudy-close exception`, err);
      }
    }

    try {
      const res = await client.publish(
        buildIraPublishPayload("ImagingStudy-open", next)
      );
      opened = Boolean(res?.ok);
      if (!opened) {
        console.warn(
          `${LOG} ImagingStudy-open failed`,
          res ? `HTTP ${res.status}` : "No response"
        );
      } else {
        console.info(`${LOG} published ImagingStudy-open`);
      }
    } catch (err) {
      console.error(`${LOG} ImagingStudy-open exception`, err);
    }
  } finally {
    lastImagingStudyOpenContext = next;
    lastLiveScene = null;
    onLiveSceneChange?.();
  }
  return { closed, opened };
}

export function getIraCastClient(): ReturnType<typeof CastClient.newInstance> | null {
  return client;
}

export type InferenceProbeResult = CastProductStatusProbe;

type StatusUpdateHandler = (message: string, level: string) => void;
let onStatusUpdate: StatusUpdateHandler | null = null;

export function setIraStatusUpdateHandler(fn: StatusUpdateHandler | null): void {
  onStatusUpdate = fn;
}

/** Handler for inbound Cast ``nifti-send`` (resolved event with file bytes). */
export type IraNiftiSendHandler = (
  event: Record<string, unknown>,
) => void | Promise<void>;

let onNiftiSend: IraNiftiSendHandler | null = null;

export function setIraNiftiSendHandler(fn: IraNiftiSendHandler | null): void {
  onNiftiSend = fn;
}

/** Handler for inbound Cast ``dicom-send`` (resolved event with file bytes). */
export type IraDicomSendHandler = (
  event: Record<string, unknown>,
) => void | Promise<void>;

let onDicomSend: IraDicomSendHandler | null = null;

export function setIraDicomSendHandler(fn: IraDicomSendHandler | null): void {
  onDicomSend = fn;
}

/** Probe a Cast resource-server product via STATUS request. */
export async function probeInferenceProduct(
  productName: string,
): Promise<InferenceProbeResult> {
  if (!client || connection !== "connected") {
    return { online: false, items: [], error: "Not connected" };
  }
  const subscriber =
    subscriberName || client.getSessionConfig()?.subscriberName?.trim() || "";
  if (!subscriber) return { online: false, items: [], error: "No subscriber name" };

  return requestCastProductStatus(client, {
    targetProductName: productName,
    subscriberName: subscriber,
    subscriberProductName: "SLICERLIVE-IRA",
    subscriberActor: ID_ACTOR,
    topic: topic || undefined,
  });
}

/** Publish a URL-file job to an inference product (dicom-send / nifti-send). */
export async function publishInferenceJob(opts: {
  product: string;
  hubEvent: string;
  files: Array<{ url: string; fileName: string }>;
  contextExtras?: Record<string, unknown>;
}): Promise<{ ok: boolean; detail: string }> {
  if (!client || connection !== "connected") {
    return { ok: false, detail: "Not connected" };
  }
  const subscriber =
    subscriberName || client.getSessionConfig()?.subscriberName?.trim() || "";
  if (!subscriber) return { ok: false, detail: "No subscriber name" };
  if (!topic) return { ok: false, detail: "No Cast topic" };
  if (!opts.files.length) return { ok: false, detail: "No volume files" };

  return publishCastUrlSend(client, {
    targetProductName: opts.product,
    hubEvent: opts.hubEvent,
    files: opts.files,
    subscriberName: subscriber,
    subscriberProductName: "SLICERLIVE-IRA",
    subscriberActor: ID_ACTOR,
    topic,
    contextExtras: opts.contextExtras,
  });
}

export function getIraLiveScene(): IraLiveSceneDoc | null {
  return lastLiveScene;
}

export function setIraLiveScene(scene: IraLiveSceneDoc | null): void {
  lastLiveScene = scene;
  onLiveSceneChange?.();
}

export function setIraLiveSceneChangeListener(fn: (() => void) | null): void {
  onLiveSceneChange = fn;
}

function buildStatusResponseData(): {
  "context.type": string;
  context: unknown[];
} {
  const context = cloneContextArray(lastImagingStudyOpenContext);
  if (lastLiveScene) {
    const withoutScene = context.filter((item) => {
      if (!item || typeof item !== "object") return true;
      return (item as { key?: string }).key !== "scene";
    });
    withoutScene.push({ key: "scene", resource: lastLiveScene });
    return { "context.type": "ImagingStudy", context: withoutScene };
  }
  if (context.length) {
    return { "context.type": "ImagingStudy", context };
  }
  return EMPTY_CAST_CONTEXT;
}

function setConnection(next: IraCastConnection, detail: string): void {
  const wasConnected = connection === "connected";
  connection = next;
  connectionDetail = detail;
  if (wasConnected && next !== "connected") {
    conference = null;
  }
  onStatus?.(next, detail);
  if (next === "connected") {
    void syncIraConferenceView();
  }
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

  const responseData = buildStatusResponseData();

  client?.sendCastRequestResponse(
    correlationId,
    "STATUS",
    responseData,
    event["hub.topic"]
  );
  return true;
}

function handleStatusUpdate(message: Record<string, unknown>): boolean {
  const update = parseStatusUpdateMessage(message);
  if (!update) return false;
  if (update.message) onStatusUpdate?.(update.message, update.level);
  return true;
}

function isNiftiSendMessage(message: Record<string, unknown>): boolean {
  const event = message?.event as { "hub.event"?: string } | undefined;
  return getHubEventLower(event) === "nifti-send";
}

function isDicomSendMessage(message: Record<string, unknown>): boolean {
  const event = message?.event as { "hub.event"?: string } | undefined;
  return getHubEventLower(event) === "dicom-send";
}

async function handleResolvedFileSend(
  message: Record<string, unknown>,
  kind: "nifti-send" | "dicom-send",
  handler: IraNiftiSendHandler | IraDicomSendHandler | null,
): Promise<void> {
  if (!client || !handler) return;
  try {
    const resolved = (await resolveCastFileMessage(client, message)) as Record<
      string,
      unknown
    >;
    const event = (resolved?.event ?? message.event) as
      | Record<string, unknown>
      | undefined;
    if (!event || typeof event !== "object") {
      console.warn(`${LOG} ${kind}: missing event after resolve`);
      return;
    }
    const files = batchContextFiles(event);
    const withData = files.filter((f) => {
      const data = (f as { data?: unknown }).data;
      return (
        data instanceof ArrayBuffer ||
        data instanceof Uint8Array ||
        typeof data === "string"
      );
    });
    if (!withData.length) {
      console.warn(`${LOG} ${kind}: no file payloads after resolve`);
      return;
    }
    console.info(
      `${LOG} ${kind}: ${withData.length} file(s) ready`,
      withData.map((f) => (f as { fileName?: string }).fileName || "?"),
    );
    await handler(event);
  } catch (err) {
    console.error(`${LOG} ${kind} handler failed`, err);
  }
}

async function handleNiftiSendMessage(
  message: Record<string, unknown>,
): Promise<void> {
  await handleResolvedFileSend(message, "nifti-send", onNiftiSend);
}

async function handleDicomSendMessage(
  message: Record<string, unknown>,
): Promise<void> {
  await handleResolvedFileSend(message, "dicom-send", onDicomSend);
}

function handleImagingStudyMessage(message: Record<string, unknown>): boolean {
  const event = message?.event as { "hub.event"?: string } | undefined;
  const name = getHubEventLower(event);
  if (name === "imagingstudy-close") {
    if (suppressInboundImagingStudy > 0) {
      console.info(`${LOG} ImagingStudy-close ignored (local publish echo)`);
      return true;
    }
    lastImagingStudyOpenContext = [];
    lastLiveScene = null;
    onLiveSceneChange?.();
    console.info(`${LOG} ImagingStudy-close`);
    void Promise.resolve(onImagingStudyClose?.()).catch((err) =>
      console.error(`${LOG} ImagingStudy-close handler failed`, err)
    );
    return true;
  }
  if (name !== "imagingstudy-open") return false;

  if (suppressInboundImagingStudy > 0) {
    const context = messageEventContext(message);
    lastImagingStudyOpenContext = cloneContextArray(context);
    console.info(`${LOG} ImagingStudy-open ignored (local publish echo)`);
    return true;
  }

  if (conference && !conferenceFollowHost) {
    console.info(`${LOG} ImagingStudy-open ignored (not following host)`);
    return true;
  }

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
 * Ask a worklist for STATUS.
 * Default (connect): presence ping on local topic; apply ImagingStudy only if we have none.
 * Join-and-follow: pass host topic + forceApply to adopt the host's open study.
 */
async function requestStatusFromWorklist(opts?: {
  topic?: string;
  forceApply?: boolean;
}): Promise<void> {
  if (!client) {
    console.warn(`${LOG} STATUS skip: no client`);
    return;
  }
  const forceApply = Boolean(opts?.forceApply);
  const requestTopic = String(opts?.topic || topic || "").trim();
  if (!forceApply && statusRequestState !== "idle") return;

  const skipApplyStudy =
    !forceApply &&
    (iraUrlHasStudyParams() || lastImagingStudyOpenContext.length > 0);
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

  if (!forceApply) statusRequestState = "pending";
  const statusDataType = "STATUS";

  try {
    const message: Record<string, unknown> = {
      "subscriber.name": subscriber,
      "subscriber.product.name": "SLICERLIVE-IRA",
      "subscriber.actor": ID_ACTOR,
      "target.actor": WORKLIST_ACTOR,
      event: {
        "hub.event": requestEventFor(statusDataType),
        ...(requestTopic ? { "hub.topic": requestTopic } : {}),
        context: { dataType: statusDataType },
      },
    };

    console.warn(`${LOG} STATUS request → ${WORKLIST_ACTOR}`, {
      subscriber,
      topic: requestTopic,
      presenceOnly: skipApplyStudy,
      forceApply,
    });
    const result = await client.request(message);
    console.warn(`${LOG} STATUS response`, {
      httpOk: result.ok,
      status: result.status,
      data: result.data,
    });

    if (!result.ok) {
      if (!forceApply) statusRequestState = "idle";
      return;
    }

    // Presence ping done; do not re-open a study we already have from URL/open.
    if (
      !forceApply &&
      (skipApplyStudy ||
        lastImagingStudyOpenContext.length ||
        iraUrlHasStudyParams())
    ) {
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
      if (!forceApply) statusRequestState = "idle";
      return;
    }

    if (
      !forceApply &&
      (lastImagingStudyOpenContext.length || iraUrlHasStudyParams())
    ) {
      statusRequestState = "done";
      return;
    }

    const context = cloneContextArray(chosen.context);
    if (!context.length) {
      console.warn(`${LOG} STATUS: ImagingStudy context empty`);
      if (!forceApply) statusRequestState = "idle";
      return;
    }

    lastImagingStudyOpenContext = context;
    if (!forceApply) statusRequestState = "done";
    console.warn(
      `${LOG} ImagingStudy from STATUS context items=`,
      context.length,
      forceApply ? "(follow host)" : ""
    );
    void Promise.resolve(onImagingStudyOpen?.(lastImagingStudyOpenContext)).catch(
      (err) =>
        console.error(`${LOG} STATUS ImagingStudy-open handler failed`, err)
    );
  } catch (err) {
    if (!forceApply) statusRequestState = "idle";
    console.error(`${LOG} failed to request STATUS`, err);
  }
}

/** Join-and-follow: pull open study from the host worklist topic. */
export async function requestIraStatusFromHostWorklist(
  hostTopic: string
): Promise<void> {
  const topicOverride = String(hostTopic || "").trim();
  if (!topicOverride) {
    console.warn(`${LOG} host STATUS skip: no host topic`);
    return;
  }
  await requestStatusFromWorklist({ topic: topicOverride, forceApply: true });
}

/** Schedule STATUS catch-up after the bind socket is up (hub registers on handshake). */
function scheduleStatusRequestFromWorklist(delayMs: number): void {
  globalThis.setTimeout(() => {
    void requestStatusFromWorklist();
  }, delayMs);
}

/** Apply radio visual classes for the current connection. */
export function applyIraCastRadioVisual(statusEl: HTMLElement): void {
  const visual = castRadioStatusVisual(connection, {
    conferenceActive: Boolean(conference),
  });
  statusEl.classList.remove(
    "ira-cast-idle",
    "ira-cast-connecting",
    "ira-cast-connected",
    "ira-cast-disconnected",
    "ira-cast-error",
    "ira-cast-pulse",
    "ira-cast-conference-pulse"
  );
  statusEl.classList.add(`ira-cast-${visual.tone}`);
  statusEl.classList.toggle("ira-cast-pulse", visual.pulse);
  statusEl.classList.toggle("ira-cast-conference-pulse", visual.conferencePulse);
  statusEl.title = conference?.title
    ? `${connectionDetail} · ${conference.title}`
    : connectionDetail;
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
        "conference-start",
        "conference-end",
        "dicom-send",
        "nifti-send",
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
    if (handleConferenceHubEvent(message)) return;
    if (handleStatusUpdate(message)) return;
    if (isDicomSendMessage(message)) {
      void handleDicomSendMessage(message);
      return;
    }
    if (isNiftiSendMessage(message)) {
      void handleNiftiSendMessage(message);
      return;
    }
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
    userName = String(session.userName || "").trim() || topic;

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
