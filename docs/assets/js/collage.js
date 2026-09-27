// Builds the looping hero collage grid from collage_manifest.json and
// pauses offscreen clips (IntersectionObserver) so a 10x10 video grid
// doesn't burn CPU decoding video nobody can see.
document.addEventListener("DOMContentLoaded", async () => {
  const grid = document.querySelector(".collage-grid");
  if (!grid) return;

  let manifest;
  try {
    const res = await fetch("assets/data/collage_manifest.json");
    manifest = await res.json();
  } catch (err) {
    console.error("Could not load collage manifest", err);
    return;
  }

  const frag = document.createDocumentFragment();
  for (const clip of manifest) {
    const video = document.createElement("video");
    video.src = clip.path;
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.setAttribute("aria-hidden", "true");
    frag.appendChild(video);
  }
  grid.appendChild(frag);

  const videos = grid.querySelectorAll("video");
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const v = entry.target;
        if (entry.isIntersecting) {
          v.play().catch(() => {});
        } else {
          v.pause();
        }
      }
    },
    { threshold: 0.05 }
  );
  videos.forEach((v) => observer.observe(v));
});
