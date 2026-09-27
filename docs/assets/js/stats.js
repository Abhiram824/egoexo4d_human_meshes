// Populates the dataset-overview stat cards on index.html from
// dataset_stats.json (computed from the released take set).
document.addEventListener("DOMContentLoaded", async () => {
  const root = document.querySelector("[data-stats-root]");
  if (!root) return;

  let stats;
  try {
    const res = await fetch("assets/data/dataset_stats.json");
    stats = await res.json();
  } catch (err) {
    console.error("Could not load dataset stats", err);
    return;
  }

  // document-wide, not scoped to the stats section: the hero sentence quotes
  // the same figures and must not be able to drift from the cards
  const set = (selector, value) => {
    document.querySelectorAll(selector).forEach((el) => {
      el.textContent = value;
    });
  };

  set("[data-stat=total-takes]", stats.total_takes.toLocaleString());
  set("[data-stat=total-hours]", `${stats.total_video_hours.toLocaleString()}h`);
  set("[data-stat=per-cam-hours]", `${stats.per_camera_hours.toLocaleString()}h`);
  set("[data-stat=activities]", Object.keys(stats.activities).length);

  // bare-number variants for use inside prose, where the unit is written out
  set("[data-stat=total-hours-num]", Math.round(stats.total_video_hours).toLocaleString());
  set("[data-stat=activities-num]", Object.keys(stats.activities).length);
});
