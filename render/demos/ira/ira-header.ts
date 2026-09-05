/**
 * IRA top-bar Cast radio menu (Session / Hub) + ? opens controls help.
 */
import {
  castRadioStatusMarkup,
} from "../../../../../cast-interface/cast-js-client/dist/cast-client.js";
import {
  applyIraCastRadioVisual,
  getIraCastSessionInfo,
  getIraHubAdminUrl,
  setIraCastStatusListener,
} from "./ira-cast.ts";
import { openChromeHelp } from "../sl-chrome.ts";

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

function refreshRadio(): void {
  const statusEl = document.getElementById("iraCastStatus");
  if (statusEl) applyIraCastRadioVisual(statusEl);
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

  const castBtn = document.getElementById("iraCastStatusBtn");
  const castMenu = document.getElementById("iraCastStatusMenu");
  const helpBtn = document.getElementById("iraSettingsBtn");
  const sessionModal = document.getElementById("iraSessionModal");
  const aboutModal = document.getElementById("iraAboutModal");
  const sessionDl = document.getElementById("iraSessionDl");
  if (!castBtn || !castMenu) return;

  const closeMenus = () => {
    castMenu.hidden = true;
    castBtn.setAttribute("aria-expanded", "false");
  };

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

  document.getElementById("iraSessionInfoBtn")?.addEventListener("click", () => {
    if (sessionDl) fillSessionDl(sessionDl);
    if (sessionModal) sessionModal.hidden = false;
    closeMenus();
  });
  document.getElementById("iraHubBtn")?.addEventListener("click", () => {
    window.open(getIraHubAdminUrl(), "_blank", "noopener,noreferrer");
    closeMenus();
  });
  document.querySelectorAll("[data-ira-close-modal]").forEach((el) => {
    el.addEventListener("click", () => {
      if (sessionModal) sessionModal.hidden = true;
      if (aboutModal) aboutModal.hidden = true;
    });
  });
}
