(() => {
  "use strict";

  const root = document.documentElement;
  const storageKey = "debug-agent-theme";
  const choices = ["light", "dark", "system"];
  const system = window.matchMedia("(prefers-color-scheme: dark)");
  let choice = "dark";

  try {
    const saved = localStorage.getItem(storageKey);
    if (choices.includes(saved)) choice = saved;
  } catch (_) { /* Theme selection also works without browser storage. */ }

  function render() {
    const resolved = choice === "system" ? (system.matches ? "dark" : "light") : choice;
    root.dataset.theme = resolved;
    root.dataset.themePreference = choice;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = resolved === "light" ? "#f5f3ef" : "#151a21";
    for (const button of document.querySelectorAll("button[data-theme-choice]")) {
      button.setAttribute("aria-pressed", String(button.dataset.themeChoice === choice));
      button.title = button.getAttribute("aria-label");
    }
  }

  // Run before the stylesheet is painted, including when a saved choice exists.
  render();
  document.addEventListener("DOMContentLoaded", render, {once: true});
  document.addEventListener("site:language", render);
  document.addEventListener("click", event => {
    if (!(event.target instanceof Element)) return;
    const button = event.target.closest("button[data-theme-choice]");
    if (!button || !choices.includes(button.dataset.themeChoice)) return;
    choice = button.dataset.themeChoice;
    render();
    try { localStorage.setItem(storageKey, choice); } catch (_) { /* Storage is optional. */ }
  });
  system.addEventListener("change", () => { if (choice === "system") render(); });
})();
