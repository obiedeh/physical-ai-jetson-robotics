"use strict";
/** Display server-owned diagnostic snapshots; never open sources on page load or refresh. */
let readinessBusy = false, readinessReport = null;

/** Give each software dependency a distinct state and a short actionable failure reason. */
function softwarePaths(dependencies, pending = false) {
  const names = {lerobot:"Dataset library", cv2:"Camera library", rclpy:"ROS Python library", sensor_msgs:"ROS state messages", ffmpeg:"Video encoder"};
  $("software-paths").replaceChildren();
  for (const [name, available] of Object.entries(dependencies)) {
    const label = pending ? "Launching" : (available ? "Ready" : "Not ready");
    const reason = pending ? "Checking availability" : (available ? "Available" : `${name} is unavailable in this console environment`);
    const row = text("li", `${label} · ${names[name] || name} — ${reason}`);
    row.dataset.level = pending ? "launching" : (available ? "ready" : "not_ready");
    $("software-paths").append(row);
  }
}

/** Show library availability separately from physical connection and recording authorization. */
async function checkSoftware() {
  $("software-panel").dataset.level = "launching";
  $("software-label").textContent = "Launching";
  $("software-message").textContent = "Checking console software; no hardware services are being launched.";
  $("check-software").disabled = true;
  softwarePaths(Object.fromEntries((state.demo ? ["lerobot", "cv2", "ffmpeg"] :
    ["lerobot", "cv2", "rclpy", "sensor_msgs", "ffmpeg"]).map(name => [name, false])), true);
  try {
    const result = await api("software");
    if (typeof window !== "undefined") window.SynriaSoftware = result;
    $("software-panel").dataset.level = result.status;
    $("software-label").textContent = result.label;
    $("software-message").textContent = result.message;
    softwarePaths(result.dependencies);
    return result.status === "ready";
  } catch (reason) {
    if (typeof window !== "undefined") window.SynriaSoftware = {status:"not_ready", message:`Software check failed: ${reason.message || reason}`};
    $("software-panel").dataset.level = "not_ready";
    $("software-label").textContent = "Not ready";
    $("software-message").textContent = `Software check failed: ${reason.message || reason}`;
    $("software-paths").replaceChildren();
    return false;
  } finally {
    $("check-software").disabled = false;
    if(typeof window !== "undefined")window.dispatchEvent(new Event("console-readiness"));
  }
}

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
  const confirmations = typeof window !== "undefined" && window.SynriaConfirmations ? window.SynriaConfirmations() : [...document.querySelectorAll("[data-readiness-confirm]")];
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
  const pathNames = state.simulation
    ? {follower_sample:"Sim arm feed", action_sample:"Leader arm (real follower feed)", wrist_sample:"Sim wrist camera", front_sample:"Sim front camera", source_startup:"Read-only source startup", isaac_record_mode:"Isaac record mode"}
    : {follower_sample:"Arm feed", action_sample:"Leader arm", wrist_sample:"Wrist camera", front_sample:"Overhead camera", source_startup:"Read-only source startup"};
  for (const [name, check] of Object.entries(report.checks)) {
    const node = document.createElement("li");
    const level = check.status || (check.message === "Not run" ? "not_checked" : (check.passed ? "ready" : "not_ready"));
    node.className = check.passed ? "passed" : (level === "not_checked" ? "not-checked" : "failed");
    const label = {not_checked:"Not checked", launching:"Launching", ready:"Ready", not_ready:"Not ready", configured:"Configured", not_applicable:"Not applicable"}[level];
    node.dataset.level = level;
    node.textContent = `${label} · ${pathNames[name] || name.replaceAll("_", " ")} — ${check.message}`;
    if (check.details) {
      const details = document.createElement("details");
      details.append(text("summary", "Technical details"), text("pre", JSON.stringify(check.details, null, 2)));
      node.append(details);
    }
    $("readiness-results").append(node);
  }
  $("readiness-confirmations").hidden = Boolean(state.demo);
  if ($("sim-panel")) $("sim-panel").hidden = !state.simulation;
  const isaacDown = Boolean(state.simulation) && !(window.SynriaIsaac && window.SynriaIsaac.running());
  $("check-readiness").disabled = readinessBusy || report.status === "checking" || Boolean(state.active_session) || isaacDown;
  $("check-readiness").title = isaacDown ? "Start Isaac record mode first; the checks run by themselves once it is running." : "";
  if(typeof window !== "undefined")window.dispatchEvent(new Event("console-readiness"));
}

/** The reason a check cannot start right now, in operator words, or null. */
function checkBlocker(confirmations) {
  if (typeof token !== "undefined" && !token && state.token_required !== false) return "This tab has no launch key. Open the link printed in the console terminal.";
  if (!state.demo && Object.values(confirmations).some(value => !value)) {
    return state.simulation ? "Tick the box to confirm you have checked the arm" : "Complete the physical safety confirmations before opening read-only sources.";
  }
  if (state.simulation && !(window.SynriaIsaac && window.SynriaIsaac.running())) return "Start Isaac record mode first; the checks run by themselves once it is running.";
  return null;
}

/** Words for the step the server is on, from the live snapshot. */
function checkStepWords(report) {
  const names = {source_startup:"opening read-only sources", follower_sample:state.simulation ? "sim arm feed" : "arm feed", action_sample:"leader arm",
    wrist_sample:state.simulation ? "sim wrist camera" : "wrist camera", front_sample:state.simulation ? "sim front camera" : "overhead camera", source_alignment:"timing alignment"};
  const launching = Object.entries(report?.checks || {}).find(([, check]) => check.status === "launching");
  return launching ? (names[launching[0]] || launching[0].replaceAll("_", " ")) : "setup";
}

/** Show progress or the blocker next to the Check now button; never leaves a click silent. */
function showCheckProgress(level, words) {
  const node = $("check-progress"); if (!node) return;
  node.dataset.level = level; node.textContent = words;
}

/** Explicit operator action starts a bounded diagnostic, never a recording or driver. */
async function runReadiness() {
  if (!$("new-form").reportValidity()) { showCheckProgress("problem", "Fill in the highlighted fields first."); return; }
  const confirmations = Object.fromEntries((typeof window !== "undefined" && window.SynriaConfirmations ? window.SynriaConfirmations() : [...document.querySelectorAll("[data-readiness-confirm]")])
    .map(node => [node.dataset.readinessConfirm, node.checked]));
  const blocker = checkBlocker(confirmations);
  if (blocker) { showCheckProgress("problem", blocker); throw Error(blocker); }
  // A deliberate new attempt supersedes the earlier setup error; background polling never does.
  if (typeof window !== "undefined") error("");
  readinessBusy = true;
  $("check-readiness").disabled = true;
  const startedAt = Date.now();
  showCheckProgress("checking", "Checking… 0 s");
  const live = typeof setInterval !== "function" ? null : setInterval(async () => {
    const seconds = Math.round((Date.now() - startedAt) / 1000);
    try { const snapshot = await api("readiness"); renderReadiness(snapshot); showCheckProgress("checking", `Checking ${checkStepWords(snapshot)}… ${seconds} s`); }
    catch (reason) { showCheckProgress("checking", `Checking… ${seconds} s`); }
  }, 500);
  try {
    if (!await checkSoftware()) { showCheckProgress("problem", "Console software is missing something; see the software panel."); return; }
    const result = await api("readiness", {settings:formValues(), confirmations});
    renderReadiness(result);
    const seconds = Math.round((Date.now() - startedAt) / 1000);
    showCheckProgress(result.status === "ready" || result.status === "attention" ? "ok" : "problem",
      `${result.label}, ${result.passed_count} of ${result.check_count} checks passed in ${seconds} s.`);
  } catch (reason) {
    showCheckProgress("problem", `Check failed: ${reason.message || reason}`); throw reason;
  } finally {
    if (live) clearInterval(live);
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
$("check-software").onclick = () => safe(checkSoftware);
$("refresh-cameras").onclick = () => safe(refreshCameraMapping);
$("new-form").addEventListener("change", () => { if (readinessReport) renderReadiness(readinessReport); });
$("readiness-confirmations").addEventListener("change", () => { if (readinessReport) renderReadiness(readinessReport); });
safe(pollReadiness);
safe(checkSoftware);
setInterval(() => safe(pollReadiness), 1000);
