// ShopRank page: example tabs and the keyword-search / ShopRank switch. Without JavaScript every example
// and both lists stay visible, so nothing depends on this file.
(() => {
  document.documentElement.classList.add("js");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  const tabs = [...document.querySelectorAll(".tab")];
  const panels = tabs.map((t) => document.getElementById(t.getAttribute("aria-controls")));

  const strip = document.querySelector(".tabs");

  function select(i, focus) {
    tabs.forEach((t, j) => {
      const on = i === j;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      panels[j].hidden = !on;
    });
    if (focus) tabs[i].focus();
    // keep the chosen tab visible inside the strip without scrolling the page
    const t = tabs[i];
    if (t.offsetLeft < strip.scrollLeft || t.offsetLeft + t.offsetWidth > strip.scrollLeft + strip.clientWidth) {
      strip.scrollTo({ left: t.offsetLeft - 16, behavior: reduce.matches ? "auto" : "smooth" });
    }
  }

  tabs.forEach((tab, i) => {
    tab.addEventListener("click", (e) => { e.preventDefault(); select(i, false); });
    tab.addEventListener("keydown", (e) => {
      const keys = { ArrowRight: 1, ArrowLeft: -1, Home: -i, End: tabs.length - 1 - i };
      if (!(e.key in keys)) return;
      e.preventDefault();
      select((i + keys[e.key] + tabs.length) % tabs.length, true);
    });
  });
  const fromHash = panels.findIndex((p) => "#" + p.id === location.hash);
  select(fromHash >= 0 ? fromHash : 0, false);

  // In-page links to an example (proof line, "Where it fails") open that tab and scroll to it.
  document.addEventListener("click", (e) => {
    const a = e.target.closest('a[href^="#ex-"]:not(.tab)');
    if (!a) return;
    const i = panels.findIndex((p) => "#" + p.id === a.getAttribute("href"));
    if (i < 0) return;
    e.preventDefault();
    select(i, false);
    panels[i].scrollIntoView({ behavior: reduce.matches ? "auto" : "smooth", block: "start" });
  });

  // Switch: rows animate from their old rank to their new rank (FLIP), so the re-ranking is visible.
  document.querySelectorAll(".example").forEach((panel) => {
    const buttons = panel.querySelectorAll(".switch button");
    buttons.forEach((btn) => btn.addEventListener("click", () => {
      const mode = btn.dataset.show;
      if (panel.dataset.mode === mode) return;
      const before = new Map();
      panel.querySelectorAll(`.col-${panel.dataset.mode} .row`).forEach((r) => before.set(r.dataset.pid, r.getBoundingClientRect().top));
      panel.dataset.mode = mode;
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      if (reduce.matches) return;
      panel.querySelectorAll(`.col-${mode} .row`).forEach((row) => {
        const old = before.get(row.dataset.pid);
        if (old === undefined) { row.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 300 }); return; }
        const dy = old - row.getBoundingClientRect().top;
        if (!dy) return;
        row.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }],
          { duration: 600, easing: "cubic-bezier(0.16, 1, 0.3, 1)" });
      });
    }));
  });
})();
