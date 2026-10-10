/**
 * Cast join for IRA — deep-link ?topic=&id-token= like OHIF CastService / worklist viewers.
 * Subscribes as Image Display (ID), answers status-request, emits status-request after
 * subscribe to catch up on an already-open worklist study, and forwards ImagingStudy-open.
 */
import CastClient, {
  castRadioStatusVisual,
  castReportingPopupFeatures,
  castReportingPopupPlacement,
  CAST_REPORTING_WINDOW_NAME,
  CAST_WORKLIST_WINDOW_NAME,
  cloneContextArray,
  generateSubscriberName,
  getHubEventLower,
  isRequestEvent,
  isRunningInCloud,
  mapInferenceServersFromStatusResult,
  matchesCastProduct,
  messageActor,
  messageEventContext,
  messageProductName,
  messageSubscriberName,
  parseStatusUpdateMessage,
  placeCastPopupWindow,
  publishCastUrlSend,
  requestCastProductStatus,
  requestEventFor,
  resolveCastConferenceView,
  resolveCastFileMessage,
  resolveCastHubAdminUrl,
  batchContextFiles,
  showCastInfoToast,
  type CastConferenceView,
  type CastProductStatusProbe,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import {
  SKIP_STUDY_RELOAD_TOAST,
  sameOpenImagingStudyContext,
} from "../../../../cast-context-identity.ts";
import {
  contextItemsFromOpenContext,
  materializeLiveSceneBlobUrls,
  sceneDocFromOpenContext,
} from "./ira-scene.ts";
import { filterSceneUpdateContextForVanishing } from "./ira-vanishing.ts";
import {
  EMPTY_SEG_CATALOG,
  type SegCatalog,
} from "./ira-seg-catalog.ts";

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
export function resolveHubConfig(
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

const ID_ACTOR = "ID";
const WORKLIST_ACTOR = "WORKLIST_CLIENT";
const REPORTING_ACTOR = "REPORTING_CLIENT";
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
/** When true, apply inbound ImagingStudy-open / display scene-updates from host. */
let conferenceFollowHost = false;
/**
 * Explicit invite decision: null = unset (default follow when roster says following),
 * true/false = Join and follow / Join but do not follow.
 */
let conferenceFollowChoice: boolean | null = null;
let lastImagingStudyOpenContext: unknown[] = [];
/** Cached LiveScene document (blobBase + nodes) for STATUS + Scene View. */
let lastLiveScene: IraLiveSceneDoc | null = null;
/** Always included on STATUS as context key ``segmentation``. */
let lastSegCatalog: SegCatalog = EMPTY_SEG_CATALOG;
/** idle → not attempted; pending → in flight; done → finished or intentionally skipped */
let statusRequestState: "idle" | "pending" | "done" = "idle";
let onStatus: StatusListener | null = null;
let onImagingStudyOpen: ImagingStudyOpenHandler | null = null;
let onImagingStudyClose: ImagingStudyCloseHandler | null = null;
let onLiveSceneChange: (() => void) | null = null;

/** Reporting peers (RPT / CAST-RPT) — same presence idea as worklist connectedApps. */
const reportingConnectedApps = new Map<
  string,
  { productName: string; actor: string }
>();

/** Worklist peers (WKLST / CAST-WKLST / WORKLIST_CLIENT). */
const worklistConnectedApps = new Map<
  string,
  { productName: string; actor: string }
>();

const TITLE_WORKLIST = "Open SlicerWorklist";
const TITLE_WORKLIST_ONLINE = "SlicerWorklist is connected on this topic";
const TITLE_WORKLIST_OFFLINE =
  "Open SlicerWorklist (no worklist responded on this topic)";
const TITLE_REPORTING = "Open DICOM SR Report Creator";
const TITLE_REPORTING_ONLINE = "DICOM SR is connected on this topic";
const TITLE_REPORTING_OFFLINE =
  "Open DICOM SR Report Creator (no reporting responded on this topic)";
const TITLE_REMOTE_AI = "Evidence Creators — inference servers";
const TITLE_REMOTE_AI_ONLINE =
  "Evidence Creators — inference server(s) online on this topic";

/** True when startup STATUS (or later probe) found at least one online AI product. */
let inferenceServersOnline = false;

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
  conferenceFollowHost: boolean;
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
    conferenceFollowHost,
  };
}

export function setIraConferenceView(view: CastConferenceView | null): void {
  const wasHost = conference?.selfRole === "leading";
  conference = view;
  if (!view) {
    conferenceFollowHost = false;
    conferenceFollowChoice = null;
  } else if (view.selfRole === "leading") {
    // Host must not be marked "following".
    conferenceFollowHost = false;
    conferenceFollowChoice = null;
  } else if (view.selfRole === "following") {
    // Default follow when listed as attendee unless user opted out / in via invite.
    if (conferenceFollowChoice === null) {
      conferenceFollowChoice = true;
      conferenceFollowHost = true;
    } else {
      conferenceFollowHost = conferenceFollowChoice;
    }
  }
  const nowHost = conference?.selfRole === "leading";
  if (nowHost && !wasHost) {
    console.info(`${LOG} conference host active — scene-update publish armed`);
    onConferenceHost?.();
  }
  onStatus?.(connection, connectionDetail);
}

export function setIraConferenceFollowHost(followHost: boolean): void {
  conferenceFollowChoice = Boolean(followHost);
  // Host leads; do not treat leading session as "following" for inbound gates.
  if (conference?.selfRole === "leading") {
    conferenceFollowHost = false;
  } else {
    conferenceFollowHost = conferenceFollowChoice;
  }
  onStatus?.(connection, connectionDetail);
}

/** True when this IRA session is the Cast conference host (leading). */
export function isIraConferenceHost(): boolean {
  return conference?.selfRole === "leading";
}

/** Fired once when this session becomes conference host (leading). */
type ConferenceHostHandler = () => void;
let onConferenceHost: ConferenceHostHandler | null = null;

export function setIraConferenceHostHandler(
  fn: ConferenceHostHandler | null,
): void {
  onConferenceHost = fn;
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
  const event = message?.event as
    | { "hub.event"?: string; context?: Record<string, unknown> }
    | undefined;
  const hubEvent = getHubEventLower(event);
  const inviteHandled = Boolean(conferenceHubHandler?.(message));
  if (inviteHandled) {
    // Invite refresh is async; still sync so worklist-started conferences arm
    // host scene-update publish via setIraConferenceView → onConferenceHost.
    if (hubEvent === "conference-start") {
      void syncIraConferenceView();
    } else if (hubEvent === "conference-end") {
      const context =
        event?.context && typeof event.context === "object" ? event.context : {};
      const leaveTopic = String(context.leaveTopic ?? "").trim();
      if (!leaveTopic || leaveTopic === topic) {
        setIraConferenceView(null);
      } else {
        void syncIraConferenceView();
      }
    }
    return true;
  }
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

/**
 * Base URL for the Cast DICOM SR reporting SPA.
 * Hub/cloud SPA mounts use same-origin `/reporting-client/`; local dual-server uses :8150.
 */
export function resolveIraReportingClientUrl(
  location: Location = globalThis.location
): string {
  if (shouldUseSameOriginHub(location)) {
    return `${location.origin}/reporting-client/`;
  }
  return "http://localhost:8150/";
}

/**
 * Base URL for the Cast worklist SPA.
 * Hub/cloud SPA mounts use same-origin `/worklist-client/`; local dual-server uses :8140.
 */
export function resolveIraWorklistUrl(
  location: Location = globalThis.location
): string {
  if (shouldUseSameOriginHub(location)) {
    return `${location.origin}/worklist-client/`;
  }
  return "http://localhost:8140/";
}

/** Worklist browser window from STATUS `clientWindow` (startup / presence probe). */
type WorklistClientWindow = {
  left: number;
  top: number;
  width: number;
  height: number;
};

type StatusResponseItem = {
  actor?: unknown;
  data?: {
    "context.type"?: unknown;
    context?: unknown;
    clientWindow?: unknown;
  };
};

const REPORTING_WINDOW_NAME = CAST_REPORTING_WINDOW_NAME;
const WORKLIST_WINDOW_NAME = CAST_WORKLIST_WINDOW_NAME;

let worklistClientWindow: WorklistClientWindow | null = null;
let reportingWindow: Window | null = null;
let worklistWindow: Window | null = null;

function parseClientWindow(raw: unknown): WorklistClientWindow | null {
  if (!raw || typeof raw !== "object") return null;
  const obj = raw as Record<string, unknown>;
  const left = Number(obj.left);
  const top = Number(obj.top);
  const width = Number(obj.width);
  const height = Number(obj.height);
  if (
    ![left, top, width, height].every((n) => Number.isFinite(n)) ||
    width <= 0 ||
    height <= 0
  ) {
    return null;
  }
  return {
    left: Math.floor(left),
    top: Math.floor(top),
    width: Math.floor(width),
    height: Math.floor(height),
  };
}

function adoptWorklistClientWindowFromStatusResponses(
  responses: StatusResponseItem[]
): void {
  for (const item of responses) {
    const actor = String(item?.actor ?? "")
      .trim()
      .toUpperCase();
    if (actor && actor !== WORKLIST_ACTOR) continue;
    const next = parseClientWindow(
      item?.data && typeof item.data === "object"
        ? (item.data as { clientWindow?: unknown }).clientWindow
        : null
    );
    if (!next) continue;
    worklistClientWindow = next;
    console.warn(`${LOG} worklist clientWindow from STATUS`, next);
    return;
  }
}

/** Same size/placement as shared helper — relative to this IRA window. */
function reportingPopupPlacement() {
  return castReportingPopupPlacement();
}

/**
 * Focus DICOM SR only if this page already holds a live Window handle.
 * Do not probe with window.open('', name) on the open path — that can steal the
 * user gesture and block the real open.
 */
function focusExistingReportingWindow(): boolean {
  if (!reportingWindow || reportingWindow.closed) {
    reportingWindow = null;
    return false;
  }
  try {
    reportingWindow.focus();
  } catch {
    /* ignore */
  }
  return true;
}

/**
 * Bring DICOM SR to the front after segment growth.
 * Prefer our Window handle; if reporting is Cast-connected but opened elsewhere
 * (e.g. worklist), reclaim by shared name and focus (no navigation).
 */
function focusReportingWindowAfterSegGrowth(): void {
  if (focusExistingReportingWindow()) return;
  if (reportingConnectedApps.size === 0) return;
  let existing: Window | null = null;
  try {
    existing = globalThis.open("", REPORTING_WINDOW_NAME);
  } catch {
    return;
  }
  if (!existing || existing.closed) return;
  try {
    const href = String(existing.location?.href || "");
    if (!href || href === "about:blank") {
      try {
        existing.close();
      } catch {
        /* ignore */
      }
      return;
    }
  } catch {
    // Cross-origin reporting app — expected when opened from worklist.
  }
  try {
    existing.focus();
  } catch {
    /* ignore */
  }
  reportingWindow = existing;
}

/** Open DICOM SR reporting with the same topic / id-token as this Image Display. */
export function openIraReportingClient(): Window | null {
  if (focusExistingReportingWindow()) {
    return reportingWindow;
  }

  const url = new URL(resolveIraReportingClientUrl());
  const persisted = readPersistedCredentials();
  const nextTopic = topic || persisted.topic;
  const nextToken =
    String(client?.getConnectionState?.()?.token || "").trim() ||
    persisted.token ||
    String(client?.getHubConfig?.()?.token || "").trim();
  if (nextTopic) url.searchParams.set("topic", nextTopic);
  if (nextToken) url.searchParams.set("id-token", nextToken);
  const href = url.toString();
  const place = reportingPopupPlacement();
  const features = castReportingPopupFeatures(place);
  // One open() only — named window reuses an existing castViewer-reporting if any.
  const win = globalThis.open(href, REPORTING_WINDOW_NAME, features);
  if (!win) {
    // Popup blocked — still navigate a new tab without window features.
    return globalThis.open(href, REPORTING_WINDOW_NAME);
  }
  reportingWindow = win;
  placeCastPopupWindow(win, place);
  return win;
}

/**
 * Focus an already-open worklist tab via a held Window handle, or opener only when
 * that opener claimed the Worklist window name.
 */
function focusExistingWorklistWindow(): boolean {
  if (worklistWindow && !worklistWindow.closed) {
    try {
      worklistWindow.focus();
    } catch {
      /* ignore */
    }
    return true;
  }
  worklistWindow = null;

  const opener = window.opener;
  if (opener && !opener.closed) {
    let openerName = "";
    try {
      openerName = String(opener.name || "");
    } catch {
      openerName = "";
    }
    if (openerName === WORKLIST_WINDOW_NAME) {
      try {
        opener.focus();
      } catch {
        /* ignore */
      }
      worklistWindow = opener;
      return true;
    }
  }
  return false;
}

/** Open SlicerWorklist with the same topic / id-token as this Image Display. */
export function openIraWorklistClient(): Window | null {
  if (connection !== "connected" && !topic) {
    console.warn(`${LOG} open worklist skipped: not connected`);
    return null;
  }
  if (focusExistingWorklistWindow()) {
    return worklistWindow;
  }

  const url = new URL(resolveIraWorklistUrl());
  const persisted = readPersistedCredentials();
  const nextTopic = topic || persisted.topic;
  const nextToken =
    String(client?.getConnectionState?.()?.token || "").trim() ||
    persisted.token ||
    String(client?.getHubConfig?.()?.token || "").trim();
  if (nextTopic) url.searchParams.set("topic", nextTopic);
  if (nextToken) url.searchParams.set("id-token", nextToken);
  const href = url.toString();
  // Named open focuses an existing Worklist tab (after it claims the name) or opens one.
  // Prefer a normal tab (no popup features) so focus matches worklist → IRA.
  const win = globalThis.open(href, WORKLIST_WINDOW_NAME);
  if (win) worklistWindow = win;
  return win;
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

/** Handler for inbound Cast ``scene-update`` (mrson ops / display patches). */
export type IraSceneUpdateHandler = (
  context: unknown[],
) => void | Promise<void>;

let onSceneUpdate: IraSceneUpdateHandler | null = null;

export function setIraSceneUpdateHandler(fn: IraSceneUpdateHandler | null): void {
  onSceneUpdate = fn;
}

/** Re-apply segment display names when the UI locale changes (3D cards + chrome). */
export type IraSegLocaleRefreshHandler = () => void;

let onSegLocaleRefresh: IraSegLocaleRefreshHandler | null = null;

export function setIraSegLocaleRefreshHandler(
  fn: IraSegLocaleRefreshHandler | null,
): void {
  onSegLocaleRefresh = fn;
}

/** No-op when no study is mounted. */
export function refreshIraSegLabelsForLocale(): void {
  onSegLocaleRefresh?.();
}

/**
 * Publish Cast scene-update display ops (mrson patches) to actor ID peers.
 * Used when this IRA is conference host so followers mirror VR/seg display state.
 */
export async function publishIraSceneUpdate(
  ops: Array<Record<string, unknown>>,
): Promise<boolean> {
  if (!client || connection !== "connected" || !topic) {
    console.warn(`${LOG} scene-update: not connected`);
    return false;
  }
  if (!Array.isArray(ops) || !ops.length) {
    console.info(`${LOG} scene-update skip: no ops`);
    return false;
  }
  try {
    const res = await client.publish({
      event: {
        "hub.topic": topic,
        "hub.event": "scene-update",
        context: [{ key: "scene", resource: { ops } }],
      },
      actor: ID_ACTOR,
      // Fan-out like ImagingStudy so all topic peers (IRA ID actors) receive it.
      "target.actor": "*",
      "target.product.name": "*",
    });
    const ok = Boolean(res && res.ok);
    if (!ok) {
      console.warn(
        `${LOG} scene-update failed`,
        res ? `HTTP ${res.status}` : "No response",
      );
    } else {
      console.info(`${LOG} published scene-update ops=`, ops.length);
    }
    return ok;
  } catch (err) {
    console.error(`${LOG} scene-update exception`, err);
    return false;
  }
}

/** Payload for Cast ``PNGFULLSIZE`` responses (base64 PNG, no data-URL prefix). */
export type IraPngCapturePayload = {
  contentType: string;
  data: string;
  width: number;
  height: number;
};

export type IraPngCaptureHandler = () =>
  | IraPngCapturePayload
  | null
  | Promise<IraPngCapturePayload | null>;

let onPngCapture: IraPngCaptureHandler | null = null;

/** @deprecated Prefer {@link setIraPngFullSizeCaptureHandler} */
export function setIraPngThumbnailCaptureHandler(
  fn: IraPngCaptureHandler | null,
): void {
  onPngCapture = fn;
}

export function setIraPngFullSizeCaptureHandler(
  fn: IraPngCaptureHandler | null,
): void {
  onPngCapture = fn;
}

function handleSceneUpdateMessage(message: Record<string, unknown>): boolean {
  const event = message?.event as { "hub.event"?: string } | undefined;
  if (getHubEventLower(event) !== "scene-update") return false;
  const context = messageEventContext(message);
  const items = Array.isArray(context) ? context : [];
  // Collaborative vanishing ink always delivers; display/study ops still require follow-host.
  // Conference hosts are not "following" but must still apply reporting (and peer) display ops.
  const deliver =
    conference && !conferenceFollowHost && !isIraConferenceHost()
      ? filterSceneUpdateContextForVanishing(items)
      : items;
  if (!deliver.length) {
    if (conference && !conferenceFollowHost && !isIraConferenceHost()) {
      console.info(`${LOG} scene-update ignored (not following host; no vanishing ops)`);
    }
    return true;
  }
  void Promise.resolve(onSceneUpdate?.(deliver)).catch((err) =>
    console.error(`${LOG} scene-update handler failed`, err),
  );
  return true;
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

export function getIraSegCatalog(): SegCatalog {
  return lastSegCatalog;
}

let lastPublishedSegCount = 0;

/**
 * Publish StructureReport-update with the current SegCatalog (metadata only).
 * No-ops when not connected. Used so reporting clients update without STATUS poll.
 * When the segment count increases, focus the DICOM SR window this page opened
 * (browsers allow opener→popup focus; the popup cannot focus itself from WS).
 */
export async function publishIraStructureReportUpdate(): Promise<boolean> {
  if (!client || connection !== "connected" || !topic) {
    return false;
  }
  const context = [{ key: "segmentation", resource: lastSegCatalog }];
  try {
    const res = await client.publish(
      buildIraPublishPayload("StructureReport-update", context)
    );
    const ok = Boolean(res?.ok);
    if (!ok) {
      console.warn(
        `${LOG} StructureReport-update failed`,
        res ? `HTTP ${res.status}` : "No response"
      );
    } else {
      const n = lastSegCatalog.segmentations.reduce(
        (acc, s) => acc + (s.segments?.length || 0),
        0
      );
      console.info(`${LOG} published StructureReport-update segments=`, n);
      if (n > lastPublishedSegCount) {
        focusReportingWindowAfterSegGrowth();
      }
      lastPublishedSegCount = n;
    }
    return ok;
  } catch (err) {
    console.error(`${LOG} StructureReport-update exception`, err);
    return false;
  }
}

/** Update the STATUS segmentation catalog (always emitted; use empty when cleared). */
export function setIraSegCatalog(catalog: SegCatalog | null | undefined): void {
  lastSegCatalog =
    catalog && Array.isArray(catalog.segmentations)
      ? { segmentations: catalog.segmentations }
      : EMPTY_SEG_CATALOG;
  void publishIraStructureReportUpdate();
}

function withSegmentationContext(context: unknown[]): unknown[] {
  const withoutSeg = context.filter((item) => {
    if (!item || typeof item !== "object") return true;
    return (item as { key?: string }).key !== "segmentation";
  });
  withoutSeg.push({ key: "segmentation", resource: lastSegCatalog });
  return withoutSeg;
}

function buildStatusResponseData(): {
  "context.type": string;
  context: unknown[];
} {
  let context = cloneContextArray(lastImagingStudyOpenContext);
  if (lastLiveScene) {
    context = context.filter((item) => {
      if (!item || typeof item !== "object") return true;
      return (item as { key?: string }).key !== "scene";
    });
    context.push({ key: "scene", resource: lastLiveScene });
  }
  context = withSegmentationContext(context);
  if (lastImagingStudyOpenContext.length || lastLiveScene) {
    return { "context.type": "ImagingStudy", context };
  }
  // No open study — still always include segmentation (may be empty).
  return { "context.type": "", context };
}

function setConnection(next: IraCastConnection, detail: string): void {
  const wasConnected = connection === "connected";
  connection = next;
  connectionDetail = detail;
  if (wasConnected && next !== "connected") {
    conference = null;
    conferenceFollowHost = false;
    conferenceFollowChoice = null;
    clearReportingConnectedApps();
    clearWorklistConnectedApps();
    clearInferenceServersOnline();
    worklistClientWindow = null;
  }
  onStatus?.(next, detail);
  if (next === "connected") {
    void syncIraConferenceView();
  }
}

/** Same connected-class idea as worklist `updateConnectedAppButtons` for DICOM SR. */
function updateIraReportingButton(): void {
  const btn = document.getElementById("iraReportingBtn");
  if (!btn) return;
  const online = reportingConnectedApps.size > 0;
  btn.classList.toggle("ira-btn-connected", online);
  btn.title = online ? TITLE_REPORTING_ONLINE : TITLE_REPORTING_OFFLINE;
}

function updateIraWorklistButton(): void {
  const btn = document.getElementById("iraWorklistBtn");
  if (!btn) return;
  const online = worklistConnectedApps.size > 0;
  btn.classList.toggle("ira-btn-connected", online);
  btn.title = online ? TITLE_WORKLIST_ONLINE : TITLE_WORKLIST_OFFLINE;
}

function updateIraRemoteAiButton(): void {
  const btn = document.getElementById("iraMedicalInferenceBtn");
  if (!btn) return;
  btn.classList.toggle("ira-btn-connected", inferenceServersOnline);
  btn.title = inferenceServersOnline ? TITLE_REMOTE_AI_ONLINE : TITLE_REMOTE_AI;
}

function setInferenceServersOnline(online: boolean): void {
  inferenceServersOnline = online;
  updateIraRemoteAiButton();
}

function clearInferenceServersOnline(): void {
  if (!inferenceServersOnline) {
    updateIraRemoteAiButton();
    return;
  }
  inferenceServersOnline = false;
  updateIraRemoteAiButton();
}

/** Apply Remote AI header highlight from a collated STATUS request result. */
function applyInferenceOnlineFromStatusResult(result: {
  ok?: boolean;
  status?: number;
  data?: unknown;
}): void {
  const mapped = mapInferenceServersFromStatusResult(result);
  const online = mapped.some((r) => r.probe?.online);
  setInferenceServersOnline(online);
  if (online) {
    const titles = mapped
      .filter((r) => r.probe?.online)
      .map((r) => r.server.title);
    console.info(`${LOG} STATUS: AI online —`, titles.join(", "));
  }
}

function clearReportingConnectedApps(): void {
  if (reportingConnectedApps.size === 0) {
    updateIraReportingButton();
    return;
  }
  reportingConnectedApps.clear();
  updateIraReportingButton();
}

function clearWorklistConnectedApps(): void {
  if (worklistConnectedApps.size === 0) {
    updateIraWorklistButton();
    return;
  }
  worklistConnectedApps.clear();
  updateIraWorklistButton();
}

function isReportingPresencePeer(productName: string, actor: string): boolean {
  const actorKey = actor.trim().toUpperCase();
  if (actorKey === REPORTING_ACTOR) return true;
  const product = productName.trim().toUpperCase();
  if (
    product === "RPT" ||
    product.startsWith("RPT-") ||
    product === "CAST-RPT" ||
    product.startsWith("CAST-RPT")
  ) {
    return true;
  }
  return matchesCastProduct(productName, "reporting");
}

function isWorklistPresencePeer(productName: string, actor: string): boolean {
  const actorKey = actor.trim().toUpperCase();
  if (actorKey === WORKLIST_ACTOR) return true;
  const product = productName.trim().toUpperCase();
  return (
    product === "WKLST" ||
    product.startsWith("WKLST-") ||
    product === "CAST-WKLST" ||
    product.startsWith("CAST-WKLST")
  );
}

/** Port of worklist `markConnectedApp` / `upsertConnectedApp` for RPT peers. */
function markReportingConnectedApp(message: Record<string, unknown>): void {
  const subscriber = messageSubscriberName(message);
  if (!subscriber) return;
  const productName = messageProductName(message);
  const actor = messageActor(message);
  if (!isReportingPresencePeer(productName, actor)) return;

  const productKey = productName.trim().toUpperCase();
  if (productKey) {
    for (const [name, existing] of reportingConnectedApps) {
      if (
        name !== subscriber &&
        existing.productName.trim().toUpperCase() === productKey
      ) {
        reportingConnectedApps.delete(name);
      }
    }
  }

  reportingConnectedApps.set(subscriber, {
    productName: productName.trim(),
    actor: actor.trim(),
  });
  updateIraReportingButton();
}

function markWorklistConnectedApp(message: Record<string, unknown>): void {
  const subscriber = messageSubscriberName(message);
  if (!subscriber) return;
  const productName = messageProductName(message);
  const actor = messageActor(message);
  if (!isWorklistPresencePeer(productName, actor)) return;

  const productKey = productName.trim().toUpperCase();
  if (productKey) {
    for (const [name, existing] of worklistConnectedApps) {
      if (
        name !== subscriber &&
        existing.productName.trim().toUpperCase() === productKey
      ) {
        worklistConnectedApps.delete(name);
      }
    }
  }

  worklistConnectedApps.set(subscriber, {
    productName: productName.trim(),
    actor: actor.trim(),
  });
  updateIraWorklistButton();
}

function markWorklistConnectedFromStatusResponses(
  responses: StatusResponseItem[]
): void {
  let found = false;
  for (const item of responses) {
    const actor = String(item?.actor ?? "").trim();
    if (!isWorklistPresencePeer("", actor)) continue;
    const key = actor || WORKLIST_ACTOR;
    worklistConnectedApps.set(key, {
      productName: "WKLST",
      actor: actor || WORKLIST_ACTOR,
    });
    found = true;
  }
  if (found) updateIraWorklistButton();
}

/** Port of worklist `removeConnectedApp` for reporting / worklist peers. */
function handleSubscriptionRemoved(message: Record<string, unknown>): boolean {
  const event = message?.event as { "hub.event"?: string } | undefined;
  if (!event || getHubEventLower(event) !== "subscription-removed") {
    return false;
  }
  const subscriber = messageSubscriberName(message);
  if (!subscriber) return false;
  let changed = false;
  if (reportingConnectedApps.has(subscriber)) {
    reportingConnectedApps.delete(subscriber);
    updateIraReportingButton();
    changed = true;
  }
  if (worklistConnectedApps.has(subscriber)) {
    worklistConnectedApps.delete(subscriber);
    updateIraWorklistButton();
    changed = true;
  }
  return changed;
}

function stripCastParamsFromUrl(): void {
  if (!globalThis.history?.replaceState) return;
  const url = new URL(globalThis.location.href);
  if (
    !url.searchParams.has("topic") &&
    !url.searchParams.has("id-token") &&
    !url.searchParams.has("followHost")
  ) {
    return;
  }
  url.searchParams.delete("topic");
  url.searchParams.delete("id-token");
  url.searchParams.delete("followHost");
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

  // Same as worklist: inbound STATUS from reporting lights DICOM SR.
  markReportingConnectedApp(message);
  markWorklistConnectedApp(message);

  const responseData = buildStatusResponseData();

  client?.sendCastRequestResponse(
    correlationId,
    "STATUS",
    responseData,
    event["hub.topic"]
  );
  return true;
}

function isPngFullSizeRequest(message: Record<string, unknown>): boolean {
  const event = message?.event as
    | {
        "hub.event"?: string;
        context?: Record<string, unknown>;
      }
    | undefined;
  if (!event || !isRequestEvent(getHubEventLower(event))) return false;
  const context =
    event.context && typeof event.context === "object" ? event.context : {};
  return String(context.dataType || "").toUpperCase() === "PNGFULLSIZE";
}

async function handlePngFullSizeRequest(
  message: Record<string, unknown>,
): Promise<boolean> {
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
  if (String(context.dataType || "").toUpperCase() !== "PNGFULLSIZE") {
    return false;
  }

  markReportingConnectedApp(message);
  markWorklistConnectedApp(message);

  let payload: IraPngCapturePayload | null = null;
  try {
    payload = (await onPngCapture?.()) ?? null;
  } catch (err) {
    console.error(`${LOG} PNGFULLSIZE capture failed`, err);
  }
  if (!payload?.data) {
    console.warn(`${LOG} PNGFULLSIZE: no capture available`);
    // Still answer so the hub request does not hang until timeout.
    client?.sendCastRequestResponse(
      correlationId,
      "PNGFULLSIZE",
      { contentType: "image/png", data: "", width: 0, height: 0, error: "no capture" },
      event["hub.topic"],
    );
    return true;
  }

  client?.sendCastRequestResponse(
    correlationId,
    "PNGFULLSIZE",
    {
      contentType: payload.contentType || "image/png",
      data: payload.data,
      width: payload.width,
      height: payload.height,
    },
    event["hub.topic"],
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
    lastSegCatalog = EMPTY_SEG_CATALOG;
    onLiveSceneChange?.();
    console.info(`${LOG} ImagingStudy-close`);
    void Promise.resolve(onImagingStudyClose?.()).catch((err) =>
      console.error(`${LOG} ImagingStudy-close handler failed`, err)
    );
    return true;
  }
  if (name !== "imagingstudy-open") return false;

  if (suppressInboundImagingStudy > 0) {
    const context = imagingStudyContextItems(message);
    lastImagingStudyOpenContext = cloneContextArray(context);
    console.info(`${LOG} ImagingStudy-open ignored (local publish echo)`);
    return true;
  }

  if (conference && !conferenceFollowHost && !isIraConferenceHost()) {
    console.info(`${LOG} ImagingStudy-open ignored (not following host)`);
    return true;
  }

  void resolveAndApplyImagingStudyOpen(message);
  return true;
}

function imagingStudyContextItems(message: Record<string, unknown>): unknown[] {
  const event = message?.event as { context?: unknown } | undefined;
  const ctx = event?.context;
  if (Array.isArray(ctx)) return ctx;
  if (ctx && typeof ctx === "object") {
    return contextItemsFromOpenContext(ctx);
  }
  return messageEventContext(message);
}

/** Raw DICOM modality code from Coding / CodeableConcept (keeps CR/DX). */
function codingModalityRaw(value: unknown): string {
  if (!value) return "";
  if (typeof value === "string") return value.trim().toUpperCase();
  if (Array.isArray(value)) {
    for (const entry of value) {
      const hit = codingModalityRaw(entry);
      if (hit) return hit;
    }
    return "";
  }
  if (typeof value === "object") {
    const obj = value as { code?: unknown; coding?: unknown; text?: unknown };
    if (obj.coding != null) return codingModalityRaw(obj.coding);
    if (obj.code != null) return String(obj.code).trim().toUpperCase();
    if (obj.text != null) return String(obj.text).trim().toUpperCase();
  }
  return "";
}

/** Best-effort modality from ImagingStudy-open context (study / series / file label). */
function modalityFromImagingStudyContext(context: unknown[]): string {
  if (!Array.isArray(context)) return "";
  for (const item of context) {
    if (!item || typeof item !== "object") continue;
    const key = String((item as { key?: unknown }).key || "");
    const resource = (item as { resource?: unknown }).resource;
    if (!resource || typeof resource !== "object") continue;
    const res = resource as {
      modality?: unknown;
      series?: unknown;
      files?: unknown;
      m?: unknown;
    };
    const fromStudy = codingModalityRaw(res.modality);
    if (fromStudy) return fromStudy;
    if (Array.isArray(res.series)) {
      for (const series of res.series) {
        if (!series || typeof series !== "object") continue;
        const fromSeries = codingModalityRaw(
          (series as { modality?: unknown }).modality
        );
        if (fromSeries) return fromSeries;
      }
    }
    // IDC / worklist sometimes put modality on volume file labels.
    if (key === "files" || Array.isArray(res.files)) {
      const files = Array.isArray(res.files) ? res.files : [];
      for (const file of files) {
        if (!file || typeof file !== "object") continue;
        const row = file as { role?: unknown; label?: unknown };
        const role = String(row.role || "")
          .trim()
          .toLowerCase();
        if (role && role !== "volume") continue;
        const label = String(row.label || "")
          .trim()
          .toUpperCase();
        if (label === "CR" || label === "DX") return label;
      }
    }
    // SegRoulette / IDC entry shape may expose ``m`` on the resource.
    const fromM = String(res.m || "")
      .trim()
      .toUpperCase();
    if (fromM) return fromM;
  }
  return "";
}

function isProjectionRadiographyModality(modality: string): boolean {
  const m = String(modality || "")
    .trim()
    .toUpperCase();
  return m === "CR" || m === "DX";
}

/**
 * CR/DX stay closed (or close if a study was loaded). Toast explains skip.
 * @returns true when the open was rejected
 */
function rejectCrDxImagingStudyOpen(
  context: unknown[],
  reason: string
): boolean {
  const modality = modalityFromImagingStudyContext(context);
  if (!isProjectionRadiographyModality(modality)) return false;

  const hadOpen =
    lastImagingStudyOpenContext.length > 0 || iraUrlHasStudyParams();
  console.info(
    `${LOG} ImagingStudy-open rejected (${reason}): modality ${modality} — not opening in SlicerLive`
  );
  showCastInfoToast(`Modality ${modality} — not opening`);

  // Never adopt CR/DX as the open study.
  lastImagingStudyOpenContext = [];
  if (hadOpen) {
    lastLiveScene = null;
    lastSegCatalog = EMPTY_SEG_CATALOG;
    onLiveSceneChange?.();
    void Promise.resolve(onImagingStudyClose?.()).catch((err) =>
      console.error(`${LOG} ImagingStudy-close after CR/DX reject failed`, err)
    );
  }
  return true;
}

async function resolveAndApplyImagingStudyOpen(
  message: Record<string, unknown>
): Promise<void> {
  if (!client) return;
  try {
    const resolved = (await resolveCastFileMessage(client, message)) as Record<
      string,
      unknown
    >;
    const event = (resolved?.event ?? message.event) as
      | { context?: unknown }
      | undefined;
    const ctx = event?.context;
    let items = imagingStudyContextItems(resolved);
    let sceneDoc = sceneDocFromOpenContext(
      Array.isArray(ctx) ? ctx : (ctx as Record<string, unknown>) || items
    );
    if (sceneDoc) {
      const files = batchContextFiles(event as Record<string, unknown>);
      sceneDoc = materializeLiveSceneBlobUrls(
        sceneDoc,
        files as Array<{
          fileName?: string;
          role?: string;
          data?: ArrayBuffer | Uint8Array | string;
        }>
      );
      // Ensure handler sees embedded scene with rewritten blob URLs.
      items = items.filter((item) => {
        if (!item || typeof item !== "object") return true;
        return (item as { key?: unknown }).key !== "scene";
      });
      items.push({ key: "scene", resource: sceneDoc });
    }
    if (rejectCrDxImagingStudyOpen(items, "websocket")) return;
    lastImagingStudyOpenContext = cloneContextArray(items);
    console.info(
      `${LOG} ImagingStudy-open context items=`,
      items.length,
      sceneDoc ? "(LiveScene)" : ""
    );
    await Promise.resolve(onImagingStudyOpen?.(lastImagingStudyOpenContext));
  } catch (err) {
    console.error(`${LOG} ImagingStudy-open resolve/apply failed`, err);
  }
}

/**
 * Ask a worklist for STATUS.
 * Default (connect): presence ping on local topic; apply ImagingStudy only if we have none.
 * Join-and-follow: pass host topic + forceApply to adopt the host's open study.
 * Always adopts worklist ``clientWindow`` for DICOM SR popup placement
 * (shared `castReportingPopupPlacement` with that opener bounds).
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
      "target.actor": "*",
      event: {
        "hub.event": requestEventFor(statusDataType),
        ...(requestTopic ? { "hub.topic": requestTopic } : {}),
        context: { dataType: statusDataType },
      },
    };

    console.warn(`${LOG} STATUS request → *`, {
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

    applyInferenceOnlineFromStatusResult(result);

    const envelope =
      result.data && typeof result.data === "object"
        ? (result.data as { responses?: StatusResponseItem[] })
        : {};
    const responses = Array.isArray(envelope.responses) ? envelope.responses : [];
    adoptWorklistClientWindowFromStatusResponses(responses);
    markWorklistConnectedFromStatusResponses(responses);

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

    if (sameOpenImagingStudyContext(context, lastImagingStudyOpenContext)) {
      console.warn(
        `${LOG} STATUS: same study already open — skip ImagingStudy apply`,
        forceApply ? "(follow host)" : ""
      );
      if (!forceApply) statusRequestState = "done";
      showCastInfoToast(SKIP_STUDY_RELOAD_TOAST);
      return;
    }

    if (rejectCrDxImagingStudyOpen(context, "STATUS")) {
      if (!forceApply) statusRequestState = "done";
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

/** Apply connection tone classes + topic label for the Cast status control. */
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
    "ira-cast-pulse"
  );
  statusEl.classList.add(`ira-cast-${visual.tone}`);
  statusEl.classList.toggle("ira-cast-pulse", visual.pulse);
  const label = topic.trim() || "—";
  statusEl.textContent = label;
  const tip = conference?.title
    ? `${connectionDetail} · ${conference.title}`
    : connectionDetail;
  statusEl.title = tip;
  const btn = statusEl.closest("button");
  if (btn) {
    btn.title = tip ? `Session info — ${tip}` : "Session info";
    btn.setAttribute("aria-label", `Session info (${label})`);
  }
}

/**
 * Join Cast when worklist deep-link params are present.
 * No-op (idle) when neither topic nor id-token is set.
 */
export async function connectIraCast(): Promise<void> {
  const params = new URLSearchParams(globalThis.location.search);
  topic = params.get("topic")?.trim() || "";
  let urlToken = params.get("id-token")?.trim() || "";
  const followHostParam = params.get("followHost")?.trim().toLowerCase() || "";
  if (followHostParam === "1" || followHostParam === "true") {
    setIraConferenceFollowHost(true);
  } else if (followHostParam === "0" || followHostParam === "false") {
    setIraConferenceFollowHost(false);
  }

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
        "pngfullsize-request",
        "subscription-removed",
        "status-update",
        "conference-start",
        "conference-end",
        "dicom-send",
        "nifti-send",
        "scene-update",
      ],
      lease: 7200,
      defaultTargetActor: "*",
    },
    callbackUrl: `${globalThis.location.origin}/castCallback`,
    preserveSessionTopicFromToken: Boolean(topic),
    autoReconnect: true,
  });

  client.onMessage((message: Record<string, unknown>) => {
    if (handleSubscriptionRemoved(message)) return;
    if (handleStatusRequest(message)) return;
    if (isPngFullSizeRequest(message)) {
      void handlePngFullSizeRequest(message);
      return;
    }
    if (handleConferenceHubEvent(message)) return;
    if (handleStatusUpdate(message)) return;
    if (handleSceneUpdateMessage(message)) return;
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
