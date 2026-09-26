/**
 * IRA top-bar Cast topic (session info) + Reporting + Start/Leave conference + Remote AI + ?.
 */
import {
  applyCastConferenceHeaderActions,
  confirmCastConferenceTakeover,
  createCastConferenceDialog,
  createCastConferenceInviteController,
  deleteCastConference,
  openCastConferenceParticipantsPopup,
  renderCastConferencePresence,
  shouldShowConferenceInvite,
  showCastInfoToast,
  transferCastConferenceLead,
  type CastConferenceView,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import {
  applyHeaderChrome,
  conferenceStringsFor,
  resolveStoredLocale,
} from "../../../../i18n/index.ts";
import {
  applyIraCastRadioVisual,
  getIraCastSessionInfo,
  getIraLiveScene,
  openIraReportingClient,
  openIraWorklistClient,
  requestIraStatusFromHostWorklist,
  setIraCastStatusListener,
  setIraConferenceFollowHost,
  setIraConferenceHubHandler,
  setIraConferenceView,
} from "./ira-cast.ts";
import { openChromeHelp } from "../sl-chrome.ts";
import {
  closeMedicalInferenceDialog,
  installMedicalInferenceUi,
  openMedicalInferenceDialog,
} from "./ira-inference.ts";
import {
  indexDirectory,
  indexFiles,
  type SeriesEntry,
} from "../../../logic/readers/dicom-local.ts";
import { unzipFileToFiles } from "../../../logic/readers/unzip-files.ts";

const ID_ACTOR = "ID";

function iraConferenceStrings() {
  return conferenceStringsFor(resolveStoredLocale());
}

export type IraLocalStudyPick =
  | { kind: "folder"; name: string; handle: FileSystemDirectoryHandle }
  | { kind: "zip"; name: string; file: File };

export type IraHeaderOpts = {
  /** Called when the user chooses a series from a local folder/ZIP. */
  onMountLocalSeries?: (
    entry: SeriesEntry,
    opts: { anonymized: boolean }
  ) => Promise<void>;
};

const THEME_KEY = "pw46.ira.theme";
const THEME_SLICERLIVE = "slicerlive";
const THEME_OHIF = "ohif";

function getStoredTheme(): string {
  try {
    const v = localStorage.getItem(THEME_KEY)?.trim();
    if (v === THEME_OHIF || v === THEME_SLICERLIVE) return v;
  } catch {
    /* ignore */
  }
  return THEME_SLICERLIVE;
}

function setStoredTheme(theme: string): void {
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* ignore */
  }
}

function applyTheme(theme: string): void {
  const next = theme === THEME_OHIF ? THEME_OHIF : THEME_SLICERLIVE;
  document.documentElement.dataset.theme = next;
  setStoredTheme(next);
}

function fillSessionDl(dl: HTMLElement): void {
  const info = getIraCastSessionInfo();
  const rows: Array<[string, string]> = [
    ["Hub", info.hubEndpoint || "—"],
    ["Subscriber", info.subscriberName || "—"],
    ["User", info.userName || "—"],
    ["Status", info.connectionDetail],
  ];
  dl.replaceChildren();
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    dl.append(dt, dd);
  }
}

function fillSceneViewPre(pre: HTMLElement): void {
  const scene = getIraLiveScene();
  if (!scene) {
    pre.textContent = "No LiveScene loaded";
    return;
  }
  try {
    pre.textContent = JSON.stringify(scene, null, 2);
  } catch {
    pre.textContent = String(scene);
  }
}

function openSceneView(): void {
  const sceneViewModal = document.getElementById("iraSceneViewModal");
  const sceneViewPre = document.getElementById("iraSceneViewPre");
  if (!getIraLiveScene()) return;
  if (sceneViewPre) fillSceneViewPre(sceneViewPre);
  if (sceneViewModal) sceneViewModal.hidden = false;
}

type LocalStudyPick = IraLocalStudyPick;

let localStudyPick: LocalStudyPick | null = null;
let onMountLocalSeries:
  | ((
      entry: SeriesEntry,
      opts: { anonymized: boolean }
    ) => Promise<void>)
  | undefined;

function setLoadLocalStatus(msg: string, show = true): void {
  const el = document.getElementById("iraLoadLocalStatus");
  if (!el) return;
  el.textContent = msg;
  el.hidden = !show || !msg;
}

function clearLoadLocalSeries(): void {
  const host = document.getElementById("iraLoadLocalSeries");
  if (!host) return;
  host.replaceChildren();
  host.hidden = true;
}

function setLocalStudySelection(pick: LocalStudyPick | null): void {
  localStudyPick = pick;
  const label = document.getElementById("iraLoadLocalSelection");
  const confirm = document.getElementById(
    "iraLoadLocalConfirmBtn"
  ) as HTMLButtonElement | null;
  if (label) {
    if (!pick) label.textContent = "No folder or ZIP selected.";
    else if (pick.kind === "folder") label.textContent = `Folder: ${pick.name}`;
    else label.textContent = `ZIP: ${pick.name}`;
  }
  if (confirm) confirm.disabled = !pick;
  clearLoadLocalSeries();
  setLoadLocalStatus("", false);
}

function openLoadLocalStudyDialog(): void {
  const modal = document.getElementById("iraLoadLocalModal");
  const anon = document.getElementById(
    "iraLoadLocalAnonymize"
  ) as HTMLInputElement | null;
  if (anon) anon.checked = true;
  setLocalStudySelection(null);
  if (modal) {
    modal.removeAttribute("hidden");
    modal.hidden = false;
  }
}

function closeLoadLocalStudyDialog(): void {
  const modal = document.getElementById("iraLoadLocalModal");
  if (modal) {
    modal.setAttribute("hidden", "");
    modal.hidden = true;
  }
  setLocalStudySelection(null);
}

function seriesButtonLabel(entry: SeriesEntry, anonymized: boolean): {
  title: string;
  meta: string;
} {
  const modality = String(entry.modality || "?").trim() || "?";
  const desc = String(entry.description || "").trim();
  const patient = anonymized
    ? "Anonymous"
    : String(entry.patientName || "").trim() || "Unknown patient";
  const title = desc
    ? `${modality} · ${desc}`
    : `${modality} · ${entry.seriesInstanceUID.slice(-12)}`;
  const meta = `${patient} · ${entry.count} image${entry.count === 1 ? "" : "s"}`;
  return { title, meta };
}

async function indexLocalPick(
  pick: LocalStudyPick,
  onProgress: (msg: string) => void,
): Promise<SeriesEntry[]> {
  if (pick.kind === "folder") {
    return indexDirectory(pick.handle, (p) => {
      onProgress(
        `Scanning: ${p.dicom} DICOM / ${p.scanned} files${p.note ? ` — ${p.note}` : ""}`,
      );
    });
  }
  onProgress("Unpacking ZIP…");
  const files = await unzipFileToFiles(pick.file);
  return indexFiles(files, (p) => {
    onProgress(
      `Scanning: ${p.dicom} DICOM / ${p.scanned} files${p.note ? ` — ${p.note}` : ""}`,
    );
  });
}

async function mountChosenSeries(
  entry: SeriesEntry,
  anonymized: boolean,
): Promise<void> {
  if (!onMountLocalSeries) {
    throw new Error("Local study loader is not ready yet");
  }
  setLoadLocalStatus("Reconstructing series…");
  await onMountLocalSeries(entry, { anonymized });
  closeLoadLocalStudyDialog();
}

function renderSeriesPicker(
  entries: SeriesEntry[],
  anonymized: boolean,
): void {
  const host = document.getElementById("iraLoadLocalSeries");
  if (!host) return;
  host.replaceChildren();
  host.hidden = false;
  for (const entry of entries) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ira-load-local-series-btn";
    const { title, meta } = seriesButtonLabel(entry, anonymized);
    const titleEl = document.createElement("span");
    titleEl.textContent = title;
    const metaEl = document.createElement("span");
    metaEl.className = "ira-load-local-series-meta";
    metaEl.textContent = meta;
    btn.append(titleEl, metaEl);
    btn.title = `${title} — ${meta}`;
    btn.addEventListener("click", () => {
      host.querySelectorAll("button").forEach((b) => {
        (b as HTMLButtonElement).disabled = true;
      });
      void mountChosenSeries(entry, anonymized).catch((err) => {
        console.error("[ira] local series mount failed", err);
        setLoadLocalStatus(
          `Error: ${(err as Error)?.message ?? String(err)}`,
        );
        host.querySelectorAll("button").forEach((b) => {
          (b as HTMLButtonElement).disabled = false;
        });
      });
    });
    host.append(btn);
  }
  setLoadLocalStatus(`Select a series (${entries.length} found).`);
}

function installLoadLocalStudyUi(): void {
  const zipInput = document.getElementById(
    "iraLoadLocalZipInput"
  ) as HTMLInputElement | null;

  document
    .getElementById("iraLoadLocalStudyBtn")
    ?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      openLoadLocalStudyDialog();
    });

  document
    .getElementById("iraLoadLocalCancelBtn")
    ?.addEventListener("click", () => closeLoadLocalStudyDialog());

  document
    .getElementById("iraLoadLocalFolderBtn")
    ?.addEventListener("click", async () => {
      const picker = (
        globalThis as unknown as {
          showDirectoryPicker?: (o?: unknown) => Promise<FileSystemDirectoryHandle>;
        }
      ).showDirectoryPicker;
      if (!picker) {
        const label = document.getElementById("iraLoadLocalSelection");
        if (label) {
          label.textContent =
            "Folder picker unavailable in this browser — choose a ZIP instead.";
        }
        return;
      }
      try {
        const handle = await picker({ id: "ira-local-dicom" });
        setLocalStudySelection({
          kind: "folder",
          name: handle.name || "Selected folder",
          handle,
        });
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          console.error("[ira] folder pick failed", err);
        }
      }
    });

  document.getElementById("iraLoadLocalZipBtn")?.addEventListener("click", async () => {
    const picker = (
      globalThis as unknown as {
        showOpenFilePicker?: (o?: unknown) => Promise<FileSystemFileHandle[]>;
      }
    ).showOpenFilePicker;
    if (picker) {
      try {
        const handles = await picker({
          multiple: false,
          types: [
            {
              description: "ZIP archive",
              accept: { "application/zip": [".zip"] },
            },
          ],
        });
        const file = await handles[0]?.getFile();
        if (file) {
          setLocalStudySelection({ kind: "zip", name: file.name, file });
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          console.error("[ira] zip pick failed", err);
        }
      }
      return;
    }
    zipInput?.click();
  });

  zipInput?.addEventListener("change", () => {
    const file = zipInput.files?.[0];
    if (file) setLocalStudySelection({ kind: "zip", name: file.name, file });
    zipInput.value = "";
  });

  document
    .getElementById("iraLoadLocalConfirmBtn")
    ?.addEventListener("click", () => {
      const pick = localStudyPick;
      if (!pick) return;
      const anon = document.getElementById(
        "iraLoadLocalAnonymize"
      ) as HTMLInputElement | null;
      const anonymized = Boolean(anon?.checked);
      const confirm = document.getElementById(
        "iraLoadLocalConfirmBtn"
      ) as HTMLButtonElement | null;
      const folderBtn = document.getElementById(
        "iraLoadLocalFolderBtn"
      ) as HTMLButtonElement | null;
      const zipBtn = document.getElementById(
        "iraLoadLocalZipBtn"
      ) as HTMLButtonElement | null;
      if (confirm) confirm.disabled = true;
      if (folderBtn) folderBtn.disabled = true;
      if (zipBtn) zipBtn.disabled = true;
      clearLoadLocalSeries();
      setLoadLocalStatus("Scanning…");
      void indexLocalPick(pick, setLoadLocalStatus)
        .then(async (entries) => {
          if (!entries.length) {
            setLoadLocalStatus("No DICOM image series found.");
            return;
          }
          if (entries.length === 1) {
            await mountChosenSeries(entries[0], anonymized);
            return;
          }
          renderSeriesPicker(entries, anonymized);
        })
        .catch((err) => {
          console.error("[ira] local study index failed", err);
          setLoadLocalStatus(
            `Error: ${(err as Error)?.message ?? String(err)}`,
          );
        })
        .finally(() => {
          if (confirm) confirm.disabled = !localStudyPick;
          if (folderBtn) folderBtn.disabled = false;
          if (zipBtn) zipBtn.disabled = false;
        });
    });
}

async function requestIraConferenceTakeover(): Promise<void> {
  const info = getIraCastSessionInfo();
  const view = info.conference;
  if (!view || info.connection !== "connected" || !info.topic) return;
  if (view.selfRole === "leading" || info.conferenceFollowHost) return;
  const ok = await confirmCastConferenceTakeover({
    strings: iraConferenceStrings(),
  });
  if (!ok) return;
  try {
    await transferCastConferenceLead(
      info.hubEndpoint,
      view.hostTopic,
      info.topic,
      info.userName || undefined,
    );
  } catch (err) {
    console.error("[ira] take over failed", err);
    showCastInfoToast(
      err instanceof Error ? err.message : "Failed to take over",
    );
  }
}

/** Conference role badge on the Cast status control (no header chips). */
function renderConferenceChips(view: CastConferenceView | null): void {
  const statusControl = document.getElementById("iraCastStatusBtn");
  const statusLabelEl = document.getElementById("iraCastStatus");
  if (!(statusControl instanceof HTMLElement)) {
    return;
  }
  const info = getIraCastSessionInfo();
  renderCastConferencePresence({
    view,
    statusControl,
    statusLabelEl,
    classPrefix: "ira-conference",
    selfStatusMode: "short",
    followHost: Boolean(info.conferenceFollowHost),
  });
}

function refreshConferenceHeaderActions(view: CastConferenceView | null): void {
  const info = getIraCastSessionInfo();
  applyCastConferenceHeaderActions({
    view,
    followHost: Boolean(info.conferenceFollowHost),
    connected: info.connection === "connected",
    strings: iraConferenceStrings(),
    startBtn: document.getElementById(
      "iraStartConferenceBtn",
    ) as HTMLButtonElement | null,
    endBtn: document.getElementById(
      "iraLeaveConferenceBtn",
    ) as HTMLButtonElement | null,
    stopFollowBtn: document.getElementById(
      "iraStopFollowingBtn",
    ) as HTMLButtonElement | null,
    resumeFollowBtn: document.getElementById(
      "iraResumeFollowingBtn",
    ) as HTMLButtonElement | null,
    takeOverBtn: document.getElementById(
      "iraTakeOverBtn",
    ) as HTMLButtonElement | null,
  });
}

function refreshRadio(): void {
  const statusEl = document.getElementById("iraCastStatus");
  if (statusEl) applyIraCastRadioVisual(statusEl);
  const view = getIraCastSessionInfo().conference;
  renderConferenceChips(view);
  refreshConferenceHeaderActions(view);
}

/** Install header actions and Cast topic control; call once early in main(). */
export function installIraHeader(opts: IraHeaderOpts = {}): void {
  onMountLocalSeries = opts.onMountLocalSeries;
  const locale = resolveStoredLocale();
  document.documentElement.lang = locale;
  applyHeaderChrome(document, locale);
  applyTheme(getStoredTheme());

  refreshRadio();
  setIraCastStatusListener(() => refreshRadio());

  const castBtn = document.getElementById("iraCastStatusBtn");
  const helpBtn = document.getElementById("iraSettingsBtn");
  const sessionModal = document.getElementById("iraSessionModal");
  const sceneViewModal = document.getElementById("iraSceneViewModal");
  const aboutModal = document.getElementById("iraAboutModal");
  const loadLocalModal = document.getElementById("iraLoadLocalModal");
  const sessionDl = document.getElementById("iraSessionDl");

  installMedicalInferenceUi();
  installLoadLocalStudyUi();

  if (!castBtn) return;

  let openConferenceDialog = (_anchor?: HTMLElement | null): void => {
    console.warn("[ira] conference dialog not available");
  };
  try {
    const conferenceDialog = createCastConferenceDialog({
      classPrefix: "cast-conference",
      strings: iraConferenceStrings(),
      getSession: () => {
        const info = getIraCastSessionInfo();
        return {
          hubEndpoint: info.hubEndpoint,
          topic: info.topic,
          subscriberName: info.subscriberName,
          userName: info.userName,
          connected: info.connection === "connected",
        };
      },
      onConferenceChange: (view) => {
        setIraConferenceView(view);
      },
      onFollowChange: (followHost) => {
        setIraConferenceFollowHost(followHost);
      },
    });
    openConferenceDialog = (anchor) => conferenceDialog.open(anchor ?? null);
  } catch (err) {
    console.error("[ira] conference dialog init failed", err);
  }

  try {
    const inviteController = createCastConferenceInviteController({
      classPrefix: "cast-conference-invite",
      strings: iraConferenceStrings(),
      getSession: () => {
        const info = getIraCastSessionInfo();
        return {
          hubEndpoint: info.hubEndpoint,
          topic: info.topic,
          subscriberName: info.subscriberName,
          userName: info.userName,
          connected: info.connection === "connected",
        };
      },
      shouldShow: () =>
        shouldShowConferenceInvite({
          selfActors: [ID_ACTOR],
        }),
      onConferenceChange: (view) => {
        setIraConferenceView(view);
      },
      onFollowChange: (followHost, invite) => {
        setIraConferenceFollowHost(followHost);
        const hostTopic = String(invite?.hostTopic || "").trim();
        if (followHost && hostTopic) {
          void requestIraStatusFromHostWorklist(hostTopic);
        }
      },
      onLeadChange: (info) => {
        const label =
          String(info?.leaderLabel || info?.leaderUserName || "").trim() ||
          "Someone";
        showCastInfoToast(`${label} took over`);
      },
    });
    setIraConferenceHubHandler((message) =>
      inviteController.handleHubMessage(message)
    );
  } catch (err) {
    console.error("[ira] conference invite init failed", err);
  }

  castBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    const info = getIraCastSessionInfo();
    const view = info.conference;
    if (view?.places?.length) {
      openCastConferenceParticipantsPopup({
        anchor: castBtn,
        view,
        followHost: Boolean(info.conferenceFollowHost),
        classPrefix: "ira-conference",
      });
      return;
    }
    if (sessionDl) fillSessionDl(sessionDl);
    if (sessionModal) sessionModal.hidden = false;
  });
  helpBtn?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    const hasScene = Boolean(getIraLiveScene());
    openChromeHelp(document.body, undefined, [
      {
        label: "Scene",
        disabled: !hasScene,
        title: hasScene ? "View LiveScene JSON" : "No LiveScene loaded",
        onClick: () => openSceneView(),
      },
    ]);
  });

  document
    .getElementById("iraStartConferenceBtn")
    ?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (getIraCastSessionInfo().conference) return;
      const btn =
        ev.currentTarget instanceof HTMLElement ? ev.currentTarget : null;
      openConferenceDialog(btn);
    });
  document.getElementById("iraMedicalInferenceBtn")?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    openMedicalInferenceDialog();
  });
  document.getElementById("iraWorklistBtn")?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    openIraWorklistClient();
  });
  document.getElementById("iraReportingBtn")?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    openIraReportingClient();
  });
  document
    .getElementById("iraLeaveConferenceBtn")
    ?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const info = getIraCastSessionInfo();
      const view = info.conference;
      if (!view || info.connection !== "connected") return;
      const leaveBtn = document.getElementById(
        "iraLeaveConferenceBtn"
      ) as HTMLButtonElement | null;
      const hostAction = Boolean(view.isConferenceOwner);
      if (leaveBtn) {
        leaveBtn.disabled = true;
        leaveBtn.textContent = hostAction ? "Ending…" : "Leaving…";
      }
      void deleteCastConference(
        info.hubEndpoint,
        view.hostTopic,
        hostAction ? undefined : info.topic
      )
        .then(() => {
          setIraConferenceView(null);
        })
        .catch((err) => {
          console.error("[ira] leave conference failed", err);
        })
        .finally(() => {
          refreshRadio();
        });
    });
  document
    .getElementById("iraStopFollowingBtn")
    ?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      setIraConferenceFollowHost(false);
      refreshRadio();
    });
  document
    .getElementById("iraResumeFollowingBtn")
    ?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const info = getIraCastSessionInfo();
      if (!info.conference || info.conferenceFollowHost) return;
      if (info.conference.selfRole === "leading") return;
      setIraConferenceFollowHost(true);
      refreshRadio();
    });
  document.getElementById("iraTakeOverBtn")?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    void requestIraConferenceTakeover();
  });
  document.querySelectorAll("[data-ira-close-modal]").forEach((el) => {
    el.addEventListener("click", () => {
      if (sessionModal) sessionModal.hidden = true;
      if (sceneViewModal) sceneViewModal.hidden = true;
      if (aboutModal) aboutModal.hidden = true;
      if (loadLocalModal) closeLoadLocalStudyDialog();
      closeMedicalInferenceDialog();
    });
  });
}
