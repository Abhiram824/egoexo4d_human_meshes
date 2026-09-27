// Builds the activity -> place tree and wires it to the single <video>
// player, driven entirely by viewer_manifest.json (no page reload).
document.addEventListener("DOMContentLoaded", async () => {
  const tree = document.querySelector(".activity-tree");
  const stage = document.querySelector(".viewer-stage video");
  const caption = document.querySelector(".viewer-caption");
  if (!tree || !stage) return;

  let manifest;
  try {
    const res = await fetch("assets/data/viewer_manifest.json");
    manifest = await res.json();
  } catch (err) {
    console.error("Could not load viewer manifest", err);
    return;
  }

  const labels = manifest.activity_labels || {};
  const activities = manifest.activities || {};

  function selectClip(button, entry, activity) {
    tree.querySelectorAll(".place-list button.active").forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    stage.src = entry.path;
    stage.play().catch(() => {});
    if (caption) {
      caption.textContent = `${labels[activity] || activity} · take ${entry.take}`;
    }
  }

  let firstButton = null;
  let firstEntry = null;
  let firstActivity = null;

  for (const activity of Object.keys(activities)) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = labels[activity] || activity;
    details.appendChild(summary);

    const list = document.createElement("ul");
    list.className = "place-list";

    const institutions = activities[activity];
    let sampleNum = 1;
    for (const institution of Object.keys(institutions)) {
      const entry = institutions[institution];
      const sampleLabel = `Sample ${sampleNum++}`;
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = sampleLabel;
      button.addEventListener("click", () => selectClip(button, entry, activity));
      li.appendChild(button);
      list.appendChild(li);

      if (!firstButton) {
        firstButton = button;
        firstEntry = entry;
        firstActivity = activity;
      }
    }
    details.appendChild(list);
    tree.appendChild(details);
  }

  if (firstButton) {
    tree.querySelector("details").open = true;
    selectClip(firstButton, firstEntry, firstActivity);
  }
});
