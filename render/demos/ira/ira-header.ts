/**
 * IRA top-bar Cast radio menu (Session / Scene View / Hub / Conferencing) + ? opens controls help.
 */
import {
  castRadioStatusMarkup,
  createCastConferenceDialog,
  createCastConferenceInviteController,
  deleteCastConference,
  shouldShowConferenceInvite,
  type CastConferenceView,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import {
  applyIraCastRadioVisual,
  getIraCastSessionInfo,
  getIraHubAdminUrl,
  getIraLiveScene,
  setIraCastStatusListener,
  setIraConferenceFollowHost,
  setIraConferenceHubHandler,
  setIraConferenceView,
  setIraLiveSceneChangeListener,
} from "./ira-cast.ts";
import { openChromeHelp } from "../sl-chrome.ts";
import {
  closeMedicalInferenceDialog,
  installMedicalInferenceUi,
  openMedicalInferenceDialog,
} from "./ira-inference.ts";

const ID_ACTOR = "ID";

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
    ["Topic", info.topic || "(none)"],
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

function refreshSceneViewButton(): void {
  const btn = document.getElementById(
    "iraSceneViewBtn"
  ) as HTMLButtonElement | null;
  if (!btn) return;
  const hasScene = Boolean(getIraLiveScene());
  btn.disabled = !hasScene;
  btn.title = hasScene ? "Scene View" : "No LiveScene loaded";
}

/** Short status for the tight IRA toolbar (full string stays on title). */
function iraChipStatus(place: {
  isSelf?: boolean;
  isLeading?: boolean;
  isFollowing?: boolean;
  statusLabel?: string;
}): string {
  if (place.isSelf && place.isLeading) return "Host";
  if (place.isSelf) return "You";
  if (place.isFollowing) return "Following";
  return String(place.statusLabel || "").trim();
}

/** Conference participant chips (same idea as worklist roster). */
function renderConferenceChips(view: CastConferenceView | null): void {
  const host = document.getElementById("iraConferenceChips");
  if (!host) return;
  host.replaceChildren();
  if (!view?.places?.length) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  for (const place of view.places) {
    const chip = document.createElement("span");
    chip.className = "ira-conference-chip";
    if (place.isSelf) chip.classList.add("ira-conference-chip-self");
    if (place.isLeading) chip.classList.add("ira-conference-chip-leading");
    if (place.isFollowing) chip.classList.add("ira-conference-chip-following");

    const name = document.createElement("span");
    name.className = "ira-conference-chip-name";
    name.textContent = place.label;
    const tip = [place.label, place.statusLabel, place.placeId]
      .map((s) => String(s || "").trim())
      .filter(Boolean)
      .join(" — ");
    name.title = tip;
    chip.title = tip;
    chip.append(name);

    const short = iraChipStatus(place);
    if (short) {
      const status = document.createElement("span");
      status.className = "ira-conference-chip-status";
      status.textContent = short;
      chip.append(status);
    }
    host.append(chip);
  }
}

function refreshLeaveConferenceButton(view: CastConferenceView | null): void {
  const btn = document.getElementById(
    "iraLeaveConferenceBtn"
  ) as HTMLButtonElement | null;
  if (!btn) return;
  const info = getIraCastSessionInfo();
  const active = Boolean(view?.places?.length);
  const connected = info.connection === "connected";
  if (!active) {
    btn.hidden = true;
    btn.disabled = true;
    btn.textContent = "Leave conference";
    btn.title = "Leave conference";
    btn.setAttribute("aria-label", "Leave conference");
    return;
  }
  const hostAction = view?.selfRole === "leading";
  const label = hostAction ? "End conference" : "Leave conference";
  btn.hidden = false;
  btn.disabled = !connected;
  btn.textContent = label;
  btn.title = label;
  btn.setAttribute("aria-label", label);
}

function refreshRadio(): void {
  const statusEl = document.getElementById("iraCastStatus");
  if (statusEl) applyIraCastRadioVisual(statusEl);
  const view = getIraCastSessionInfo().conference;
  renderConferenceChips(view);
  refreshLeaveConferenceButton(view);
}

/** Install menus and radio mark; call once early in main(). */
export function installIraHeader(): void {
  applyTheme(getStoredTheme());

  const statusEl = document.getElementById("iraCastStatus");
  if (statusEl) {
    statusEl.innerHTML = castRadioStatusMarkup({ slashId: "iraCastRadioSlash" });
  }
  refreshRadio();
  setIraCastStatusListener(() => refreshRadio());
  refreshSceneViewButton();
  setIraLiveSceneChangeListener(() => refreshSceneViewButton());

  const castBtn = document.getElementById("iraCastStatusBtn");
  const castMenu = document.getElementById("iraCastStatusMenu");
  const helpBtn = document.getElementById("iraSettingsBtn");
  const sessionModal = document.getElementById("iraSessionModal");
  const sceneViewModal = document.getElementById("iraSceneViewModal");
  const aboutModal = document.getElementById("iraAboutModal");
  const sessionDl = document.getElementById("iraSessionDl");
  const sceneViewPre = document.getElementById("iraSceneViewPre");
  if (!castBtn || !castMenu) return;

  installMedicalInferenceUi();

  const closeMenus = () => {
    castMenu.hidden = true;
    castBtn.setAttribute("aria-expanded", "false");
  };

  let openConferenceDialog = (): void => {
    console.warn("[ira] conference dialog not available");
  };
  try {
    const conferenceDialog = createCastConferenceDialog({
      classPrefix: "cast-conference",
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
    });
    openConferenceDialog = () => conferenceDialog.open();
  } catch (err) {
    console.error("[ira] conference dialog init failed", err);
  }

  try {
    const inviteController = createCastConferenceInviteController({
      classPrefix: "cast-conference-invite",
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
      onFollowChange: (followHost) => {
        setIraConferenceFollowHost(followHost);
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
    const open = castMenu.hidden;
    closeMenus();
    castMenu.hidden = !open;
    castBtn.setAttribute("aria-expanded", open ? "true" : "false");
  });
  helpBtn?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    closeMenus();
    openChromeHelp();
  });
  document.addEventListener("click", () => closeMenus());
  castMenu.addEventListener("click", (ev) => ev.stopPropagation());

  document.getElementById("iraConferenceBtn")?.addEventListener("click", () => {
    closeMenus();
    openConferenceDialog();
  });
  document.getElementById("iraMedicalInferenceBtn")?.addEventListener("click", (ev) => {
    ev.stopPropagation();
    closeMenus();
    openMedicalInferenceDialog();
  });
  document
    .getElementById("iraLeaveConferenceBtn")
    ?.addEventListener("click", (ev) => {
      ev.stopPropagation();
      closeMenus();
      const info = getIraCastSessionInfo();
      const view = info.conference;
      if (!view || info.connection !== "connected") return;
      const leaveBtn = document.getElementById(
        "iraLeaveConferenceBtn"
      ) as HTMLButtonElement | null;
      const hostAction = view.selfRole === "leading";
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
  document.getElementById("iraSessionInfoBtn")?.addEventListener("click", () => {
    if (sessionDl) fillSessionDl(sessionDl);
    if (sessionModal) sessionModal.hidden = false;
    closeMenus();
  });
  document.getElementById("iraSceneViewBtn")?.addEventListener("click", () => {
    const btn = document.getElementById("iraSceneViewBtn") as HTMLButtonElement | null;
    if (btn?.disabled) return;
    if (sceneViewPre) fillSceneViewPre(sceneViewPre);
    if (sceneViewModal) sceneViewModal.hidden = false;
    closeMenus();
  });
  document.getElementById("iraHubBtn")?.addEventListener("click", () => {
    window.open(getIraHubAdminUrl(), "_blank", "noopener,noreferrer");
    closeMenus();
  });
  document.querySelectorAll("[data-ira-close-modal]").forEach((el) => {
    el.addEventListener("click", () => {
      if (sessionModal) sessionModal.hidden = true;
      if (sceneViewModal) sceneViewModal.hidden = true;
      if (aboutModal) aboutModal.hidden = true;
      closeMedicalInferenceDialog();
    });
  });
}
