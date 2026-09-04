/**
 * IRA top-bar Cast radio + settings menus (theme / Controls / About / Session / Hub).
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
  document
    .querySelectorAll<HTMLButtonElement>("[data-theme-option]")
    .forEach((btn) => {
      btn.setAttribute(
        "aria-checked",
        btn.dataset.themeOption === next ? "true" : "false"
      );
    });
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

  const castBtn = document.getElementById("iraCastStatusBtn") as HTMLButtonElement;
  const castMenu = document.getElementById("iraCastStatusMenu") as HTMLElement;
  const settingsBtn = document.getElementById("iraSettingsBtn") as HTMLButtonElement;
  const settingsMenu = document.getElementById("iraSettingsMenu") as HTMLElement;
  const sessionModal = document.getElementById("iraSessionModal") as HTMLElement;
  const aboutModal = document.getElementById("iraAboutModal") as HTMLElement;
  const sessionDl = document.getElementById("iraSessionDl") as HTMLElement;

  const closeMenus = () => {
    castMenu.hidden = true;
    settingsMenu.hidden = true;
    castBtn.setAttribute("aria-expanded", "false");
    settingsBtn.setAttribute("aria-expanded", "false");
  };

  castBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    const open = castMenu.hidden;
    closeMenus();
    castMenu.hidden = !open;
    castBtn.setAttribute("aria-expanded", open ? "true" : "false");
  });
  settingsBtn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    const open = settingsMenu.hidden;
    closeMenus();
    settingsMenu.hidden = !open;
    settingsBtn.setAttribute("aria-expanded", open ? "true" : "false");
  });
  document.addEventListener("click", () => closeMenus());
  castMenu.addEventListener("click", (ev) => ev.stopPropagation());
  settingsMenu.addEventListener("click", (ev) => ev.stopPropagation());

  document.getElementById("iraSessionInfoBtn")?.addEventListener("click", () => {
    fillSessionDl(sessionDl);
    sessionModal.hidden = false;
    closeMenus();
  });
  document.getElementById("iraHubBtn")?.addEventListener("click", () => {
    window.open(getIraHubAdminUrl(), "_blank", "noopener,noreferrer");
    closeMenus();
  });
  document.querySelectorAll<HTMLButtonElement>("[data-theme-option]").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyTheme(btn.dataset.themeOption || THEME_SLICERLIVE);
      closeMenus();
    });
  });
  document.getElementById("iraControlsBtn")?.addEventListener("click", () => {
    closeMenus();
    openChromeHelp();
  });
  document.getElementById("iraAboutBtn")?.addEventListener("click", () => {
    aboutModal.hidden = false;
    closeMenus();
  });
  document.querySelectorAll("[data-ira-close-modal]").forEach((el) => {
    el.addEventListener("click", () => {
      sessionModal.hidden = true;
      aboutModal.hidden = true;
    });
  });
}
