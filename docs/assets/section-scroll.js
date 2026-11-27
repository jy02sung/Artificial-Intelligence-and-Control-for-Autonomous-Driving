let scrollFrame;
function cancelSectionScroll() {
  cancelAnimationFrame(scrollFrame);
}

function sectionScrollProgress(t) {
  const ramp = 0.25;
  const cruiseSpeed = 1 / (1 - ramp);
  if (t < ramp) return cruiseSpeed * t * t / (2 * ramp);
  if (t <= 1 - ramp) return cruiseSpeed * (t - ramp / 2);
  const remaining = 1 - t;
  return 1 - cruiseSpeed * remaining * remaining / (2 * ramp);
}

function scrollToSection(target, alignment = "start") {
  cancelSectionScroll();
  const root = document.documentElement;
  const start = window.scrollY;
  const rect = target.getBoundingClientRect();
  const inset = parseFloat(getComputedStyle(root).scrollPaddingTop) || 0;
  const offset = alignment === "center" ? (window.innerHeight - rect.height) / 2 : inset;
  const end = Math.max(0, Math.min(root.scrollHeight - window.innerHeight, start + rect.top - offset));
  const distance = end - start;

  if (!distance || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    window.scrollTo(0, end);
    target.focus({preventScroll: true});
    return;
  }

  const duration = Math.min(560, 360 + Math.abs(distance) * 0.07);
  const began = performance.now();
  function slide(now) {
    const t = Math.min(1, Math.max(0, (now - began) / duration));
    window.scrollTo(0, start + distance * sectionScrollProgress(t));
    if (t < 1) scrollFrame = requestAnimationFrame(slide);
    else target.focus({preventScroll: true});
  }
  scrollFrame = requestAnimationFrame(slide);
}

document.addEventListener("click", event => {
  const link = event.target.closest("a[href]");
  if (!link || !link.hash || event.defaultPrevented || event.button !== 0 ||
      event.metaKey || event.ctrlKey || event.shiftKey || event.altKey ||
      (link.target && link.target !== "_self") || link.hasAttribute("download")) return;
  if (link.origin !== location.origin || link.pathname !== location.pathname ||
      link.search !== location.search) return;
  const target = document.getElementById(link.hash.slice(1));
  if (!target) return;
  event.preventDefault();
  if (location.hash !== link.hash) history.pushState(null, "", link.hash);
  scrollToSection(target);
});

for (const type of ["wheel", "touchstart", "pointerdown", "keydown", "popstate", "hashchange"]) {
  window.addEventListener(type, cancelSectionScroll, {passive: true});
}
