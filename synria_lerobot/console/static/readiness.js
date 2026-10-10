"use strict";
/** Display server-owned diagnostic snapshots; never open sources on page load or refresh. */
let readinessBusy = false, readinessReport = null;

/** Compare form settings by content so edits and page reloads invalidate unrelated results. */
function matchingReadinessSettings(left, right) {
  const keys = Object.keys(left).sort();
  return JSON.stringify(keys) === JSON.stringify(Object.keys(right).sort()) &&
    keys.every(key => JSON.stringify(left[key]) === JSON.stringify(right[key]));
}

/** Render accessible progress, individual failure reasons and explicit snapshot limitations. */
function renderReadiness(report) {
  readinessReport = report;
  const changed = report.checked_utc && !matchingReadinessSettings(report.form_settings, formValues());
  const confirmations = [...document.querySelectorAll("[data-readiness-confirm]")];
  const unconfirmed = !state.demo && confirmations.some(node => !node.checked);
  const invalid = (changed || unconfirmed) && report.status === "ready";
  const level = invalid ? "expired" : report.status;
  const percent = invalid ? 0 : report.percent;
  $("readiness-label").textContent = invalid ? "Settings or confirmations changed · run again" : report.label;
  $("readiness-count").textContent = `${report.passed_count} / ${report.check_count} checks passed`;
  $("readiness-meter").dataset.level = level;
  $("readiness-meter").setAttribute("aria-valuenow", String(percent));
  $("readiness-meter").setAttribute("aria-valuetext", $("readiness-label").textContent);
  $("readiness-fill").style.width = `${percent}%`;
  $("readiness-scope").textContent = `${report.scope} ${report.checked_utc ? `Checked ${report.checked_utc}. Expires after ${report.expires_after_s} seconds.` : ""}${changed ? " These results describe previous form settings; run again after changes." : ""}`;
  $("readiness-results").replaceChildren();
  for (const [name, check] of Object.entries(report.checks)) {
    const node = document.createElement("li");
    node.className = check.passed ? "passed" : "failed";
    const label = check.message === "Not run" ? "NOT CHECKED" : (check.passed ? "PASS" : "NEEDS ATTENTION");
    node.textContent = `${label} · ${name.replaceAll("_", " ")} — ${check.message}`;
    if (check.details) node.append(text("pre", JSON.stringify(check.details, null, 2)));
    $("readiness-results").append(node);
  }
  $("readiness-confirmations").hidden = Boolean(state.demo);
  $("check-readiness").disabled = readinessBusy || report.status === "checking" || Boolean(state.active_session);
}

/** Explicit operator action starts a bounded diagnostic, never a recording or driver. */
async function runReadiness() {
  if (!$("new-form").reportValidity()) return;
  const confirmations = Object.fromEntries([...document.querySelectorAll("[data-readiness-confirm]")]
    .map(node => [node.dataset.readinessConfirm, node.checked]));
  if (!state.demo && Object.values(confirmations).some(value => !value)) {
    throw Error("Complete the physical safety confirmations before opening read-only sources.");
  }
  readinessBusy = true;
  $("check-readiness").disabled = true;
  try {
    renderReadiness(await api("readiness", {settings:formValues(), confirmations}));
  } finally {
    readinessBusy = false;
    await pollReadiness();
  }
}

/** Poll existing results only; no HTTP GET performs live connection probes. */
async function pollReadiness() { renderReadiness(await api("readiness")); }

/** Refresh stable names without opening devices or silently choosing replacement cameras. */
async function refreshCameraMapping() {
  const inventory = await api("cameras");
  $("camera-note").textContent = inventory.message || `${inventory.cameras.length} stable video entries found; verify two distinct camera identities.`;
  for (const id of ["wrist-choice", "front-choice"]) {
    const select = $(id), previous = select.value;
    select.replaceChildren();
    const empty = text("option", "Choose camera ID"); empty.value = ""; select.append(empty);
    for (const path of inventory.cameras) {
      const option = text("option", path); option.value = path; select.append(option);
    }
    select.value = inventory.cameras.includes(previous) ? previous : "";
  }
  if (readinessReport) renderReadiness(readinessReport);
}

$("check-readiness").onclick = () => safe(runReadiness);
$("refresh-cameras").onclick = () => safe(refreshCameraMapping);
$("new-form").addEventListener("change", () => { if (readinessReport) renderReadiness(readinessReport); });
$("readiness-confirmations").addEventListener("change", () => { if (readinessReport) renderReadiness(readinessReport); });
safe(pollReadiness);
setInterval(() => safe(pollReadiness), 1000);
