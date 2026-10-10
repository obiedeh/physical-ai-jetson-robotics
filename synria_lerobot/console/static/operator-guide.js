"use strict";
/** Guide operators through existing recorder paths without adding device or command capabilities. */
const SynriaGuide = (() => {
  const help = {
    source: {title:"Prepare the follower state source", summary:"The console listens; it does not start the driver or control the arm.", steps:[
      "Secure the arm, clear the workspace and verify the approved power and controlled-stop procedure. Never release torque on an unsupported arm.",
      "Keep the existing leader–follower hardware-sync teleoperation unchanged. Reuse its verified read-only state source; never start a second owner of the follower USB connection.",
      "The source must publish the selected follower topic with arm writes disabled and startup torque unchanged. Selecting ros2_control in this form does not launch it or verify those properties.",
      "For ros2_control, use only the operator-verified read-only configuration. No verified startup command is recorded here. Do not use the write-enabled D2 policy launch as a recording shortcut.",
      "Compare this console’s displayed ROS domain and middleware with the source terminal. The topic and domain must agree. A timeout alone cannot distinguish an absent publisher from discovery or configuration problems.",
      "After the approved source is running, return to Check connections and explicitly run the read-only check. If the setup is not verified, stop here and complete the safety record; do not guess flags or force readiness."], doc:"runbook"},
    timeout: {title:"No follower state received", summary:"A subscription was created, but no message arrived before the startup wait ended.", steps:[
      "Confirm the existing approved state source is actually running. This console never starts it, and a USB entry is not proof of a publisher.",
      "Check the configured follower topic. Compare ROS domain and middleware with the environment shown under Prepare.",
      "Use the state-source procedure in Operator help. Do not start another process on an already owned port or change working teleoperation.",
      "Retry the connection check after resolving the cause. A longer first-sample wait only accommodates discovery; it cannot create missing data or relax steady-state freshness.",
      "Cameras have not failed merely because they remain Not checked: the shared preflight deliberately checks state and command publishers before opening them."], doc:"runbook"},
    software: {title:"Prepare recording software", summary:"A required library or encoder is not visible to this console process.", steps:[
      "Save or explicitly discard pending frames before closing the console. Do not terminate an active capture to repair the environment.",
      "Use the RTX console launcher in the existing Python 3.12 recording environment. It activates that environment and loads ROS library paths in physical mode, without starting robot services.",
      "If a dependency remains unavailable, follow the runbook’s dedicated-environment installation procedure. Do not install into the working teleoperation environment.",
      "Reopen using the new launch URL, then Recheck software. Library availability alone does not verify binary compatibility or physical connections."], doc:"runbook"},
    camera: {title:"Restore the camera input", summary:"The selected camera did not provide an acceptable frame.", steps:[
      "Identify two distinct physical cameras and refresh stable IDs. Multiple video-index entries for one USB identity are not two cameras.",
      "Close other camera users, including preview applications that remain running after their window closes. Check for a lingering preview with pgrep -a cheese; close it through its normal controls.",
      "Confirm the selected video interface supports capture. The list is name-only; it cannot establish that index0 or index1 is usable on this device.",
      "Inspect covers, lighting and framing. Correct the cause rather than relaxing freshness, image-size or visual gates, then rerun the explicit connection check."], doc:"runbook"},
    mapping: {title:"Choose two distinct camera identities", summary:"Wrist and fixed-front must refer to different physical cameras.", steps:[
      "Refresh camera IDs in Configure, then identify the C10 wrist camera and fixed front webcam.",
      "Do not assign index0 and index1 of the same USB identity to different roles.",
      "Listing a device does not open or verify it. The connection check and later live framing review are still required."], doc:"runbook"},
    rate: {title:"Requested rate exceeds available state", summary:"Recording cannot sample faster than the measured incoming state stream.", steps:[
      "Read the measured rate in technical evidence. The requested, incoming and achieved sample rates are different facts.",
      "Choose a supported explicit FPS in Configure; 15 and 30 are vendor candidates, not guarantees.",
      "For an existing dataset, changing FPS requires a separate dataset/session. Do not bypass the rate gate or increase a driver’s rate speculatively."], doc:"protocol"},
    freshness: {title:"Source timing is not acceptable", summary:"A source is stale, its ROS stamp disagrees with arrival time, or the views are misaligned.", steps:[
      "Confirm source continuity, host/ROS clock configuration and camera load through the approved procedure.",
      "Do not increase freshness or skew thresholds to hide the failure. A first-message startup wait does not relax steady-state rules.",
      "Keep failed captures and their gate results. Retry the read-only check only after addressing the cause."], doc:"protocol"},
    publishers: {title:"Command path is not quiet", summary:"A publisher or graph-query failure prevents read-only recording.", steps:[
      "Inspect the exact guarded topic and error in technical evidence. A publisher blocks recording even if it is not currently sending.",
      "Follow the operator-approved procedure to resolve the source configuration. Do not kill publishers, disable the guard or reconfigure working teleoperation just to pass.",
      "Recording needs no policy bridge, arming message or command publisher. Retry only after the approved read-only arrangement is established."], doc:"runbook"},
    metadata: {title:"Complete recording identity and power", summary:"These facts are required for recording, but missing text does not prevent the no-episode diagnostic.", steps:[
      "Enter the actual follower manufacturer serial from verified equipment records, packaging or vendor confirmation. Do not copy the USB adapter ID or invent an identity.",
      "Enter the current observed starting power state; do not restore an assumption from a prior session.",
      "If identity cannot be established, leave it unknown and stop before recording. Connection checks still require all physical safety confirmations."], doc:"safety"},
    timing: {title:"Register the real task window", summary:"This skill has no operator-timed qualifying episode window yet.", steps:[
      "Time the actual task under the approved physical procedure; do not infer duration from a software test.",
      "Prospectively record observations, chosen minimum/maximum, rationale, operator and date in DECISIONS.md. Update that task in config/synria_tasks.json through reviewed repository changes.",
      "The console cannot set or override this window. A disposable 20-second smoke is a separate, nonqualifying path and still requires identity, safety and shared preflight."], doc:"decisions"},
    limits: {title:"Verify the physical limits", summary:"Qualifying evidence requires operator-verified joint and gripper limits.", steps:[
      "Review config/synria_limits.yaml against the actual arm, installed gripper and accepted safety evidence.",
      "Record verified_by and verified_on only after genuine verification. Do not fill them merely to make the indicator green.",
      "Unverified limits remain visible and qualifying counts stay zero. This console neither edits limits nor authorizes motion."], doc:"limits"},
    disk: {title:"Restore safe recording space", summary:"The workspace has insufficient available space or a filesystem operation failed.", steps:[
      "Do not discard pending frames just to retry. Keep the process alive while resolving a save failure.",
      "Check the external workspace’s capacity and permissions. Preserve datasets, sidecars, recording locks and journals.",
      "After freeing space through reviewed file management, retry saving the same pending episode or rerun preflight. Do not lower the disk floor to conceal the problem."], doc:"runbook"},
    recovery: {title:"Preserve the dataset for recovery", summary:"The writer detected a lock, journal or recovery inconsistency.", steps:[
      "Stop new collection and keep the exact error. Do not remove the lock or journal, rename sidecars or overwrite the dataset.",
      "Confirm whether another recorder owns the dataset. Never open a second writer or force a deletion.",
      "Use operator-reviewed transaction recovery. Review remains read-only where available; writes and deletion stay blocked until the cause is resolved."], doc:"runbook"},
    save: {title:"Keep pending frames and retry safely", summary:"A stopped capture is not durable until its save completes.", steps:[
      "Keep the console process running. Closing a browser tab does not stop capture, but terminating the process loses pending frames.",
      "Inspect the save error and resolve storage or source issues. Use Retry save to retain the same frames and label.",
      "If recovery is blocked, preserve locks and journals and stop writes. Discard is only for an unusable capture and requires a reason; retain genuine task failures with Failure.",
      "After a save, review both views, state/action traces, gates and native final still before scaling collection."], doc:"runbook"},
    contract: {title:"Keep dataset settings consistent", summary:"The existing dataset does not match the proposed session contract or capture provenance.", steps:[
      "Resume with the exact original task, image/camera settings, FPS, action source, lookahead, gripper and velocity contract.",
      "If the physical setup or contract intentionally changed, create a separate dataset/session. Do not edit saved sidecars to force a match.",
      "Preserve the original dataset and evidence. Reindex rebuilds the catalog; it cannot repair or override a contract mismatch."], doc:"protocol"},
    network: {title:"Reconnect without losing a capture", summary:"The browser cannot confirm the server’s current state.", steps:[
      "Do not assume the recorder stopped. Capture lives in the server and may continue until its hard cap.",
      "Reconnect to the same console and inspect restored state before repeating any recording action. Do not launch another instance or kill the process while frames may be pending.",
      "If the process was restarted intentionally, open its newest launch URL. Save/discard decisions remain explicit; no request is retried automatically."], doc:"runbook"},
    auth: {title:"Reopen this console’s launch URL", summary:"The page does not have the current per-launch authorization token.", steps:[
      "Open the URL printed by the running console, including its token fragment. A restart changes that token.",
      "Use the exact local host and port; do not paste the token into a public report.",
      "Reopening a view does not stop or restart capture. Inspect the restored session before taking action."], doc:"runbook"},
    general: {title:"Resolve the reported check", summary:"The operation was refused; no successful result should be assumed.", steps:[
      "Read the exact error and expand technical evidence. Correct the indicated field or prerequisite before retrying.",
      "Do not bypass safety checks, edit dataset evidence or restart a process with pending frames.",
      "Use the full operator procedure below for the required order. If the error concerns unverified hardware behavior, stop physical setup until it is documented."], doc:"runbook"},
  };

  /** Classify known refusal messages for assistance only; never use text matching to authorize actions. */
  function issueFor(name = "", message = "") {
    const value = `${name} ${message}`.toLowerCase();
    const rules = [
      ["auth", /token|origin|host is not/], ["network", /failed to fetch|network|load failed|connection refused/],
      ["recovery", /recover|journal|lock|another console|another writer/],
      ["publishers", /publisher|command_topic|command topic|graph/],
      ["timeout", /no state sample|first sample|state_rate.*no |state source.*timed out/],
      ["mapping", /camera_mapping|two distinct|same usb|camera identities/],
      ["freshness", /stale|freshness|header|skew|alignment/],
      ["camera", /camera|wrist|front|black|frozen|duplicat.*frame/],
      ["rate", /state_rate|fps.*exceed|rate.*exceed/],
      ["metadata", /recording_metadata|follower_serial|power_state_start/],
      ["timing", /qualifying_configuration|episode window|operator timing/],
      ["limits", /limits_verification|unverified.*limit|limits unverified/],
      ["software", /dependencies|rclpy|sensor_msgs|ffmpeg|no module|import|library/],
      ["disk", /disk|space|permission denied|read-only file/],
      ["contract", /contract|provenance|mismatch/], ["save", /sav|pending frame/],
      ["source", /state_source|follower|source_startup/],
    ];
    const key = rules.find(([, pattern]) => pattern.test(value))?.[0] || "general";
    return {key, ...help[key]};
  }

  /** Project real check results into four input cards without treating construction as data receipt. */
  function connectionPaths(report, current = true) {
    const checks = report?.checks || {};
    const definitions = [
      ["follower", "Follower state", ["follower_sample", "state_rate", "state_source"]],
      ["action", "Action source", ["action_sample"]],
      ["wrist", "Wrist camera", ["wrist_sample", "wrist_camera"]],
      ["front", "Overhead / front", ["front_sample", "front_camera"]],
    ];
    return definitions.map(([id, title, keys]) => {
      const sample = checks[keys[0]];
      const failureKey = keys.find(key => checks[key]?.passed === false &&
        !["not_checked", "launching"].includes(checks[key]?.status) && checks[key]?.message !== "Not run");
      const key = failureKey || keys.find(key => checks[key]?.status === "launching") || keys[0];
      const check = checks[key] || sample;
      let status = "not_checked", message = "Waiting for an explicit connection check.";
      if (!current) message = "Previous result only. Settings changed or the snapshot expired; run again.";
      else if (failureKey) { status = "not_ready"; message = check.message; }
      else if (check?.status === "launching") { status = "launching"; message = check.message; }
      else if (sample?.passed === true) { status = "ready"; message = sample.message; }
      else if (check?.message && check.message !== "Not run") message = check.message;
      else if (id !== "follower" && checks.state_rate?.passed === false)
        message = "Waiting for follower state. This input has not been tested.";
      return {id, title, status, message, help:issueFor(key, message)};
    });
  }

  /** Identify missing connection form inputs without filling unknown identity or overriding backend validation. */
  function missingInputs(settings, demo) {
    const names = ["name", "task_id", "gripper_type", "fps", "operator", "scene"];
    if (!demo) names.push("state_source", "wrist_camera", "front_camera");
    return names.filter(name => settings[name] === undefined || String(settings[name]).trim() === "" ||
      (name === "fps" && (!Number.isInteger(Number(settings[name])) || Number(settings[name]) <= 0)));
  }

  /** Choose the next visible action, keeping software, connectivity and qualifying evidence separate. */
  function nextStep({settings, demo, report, softwareReady, current, active, confirmed}) {
    if (active) return {step:"record", title:"Continue the open session", message:"Resolve pending frames before closing sources or changing setup."};
    if (!softwareReady) return {step:"prepare", title:"Check the recording environment", message:"Resolve missing libraries before opening physical sources.", help:"software"};
    const missing = missingInputs(settings, demo);
    if (missing.length) return {step:"configure", title:"Complete your connection setup", message:`Still needed: ${missing.map(name => name.replaceAll("_", " ")).join(", ")}.`, field:missing[0]};
    if (!confirmed) return {step:"check", title:"Confirm the physical safety checklist", message:"These checks need operator confirmation; software cannot verify torque, e-stop behavior or the sync cable."};
    if (!current || !report?.checked_utc) return {step:"check", title:"Run a fresh connection check", message:"A changed or expired snapshot is not current evidence. This check records no episode."};
    const failed = connectionPaths(report).find(path => path.status === "not_ready");
    if (failed) return {step:"check", title:failed.help.title, message:failed.message, help:failed.help.key};
    if (report.status === "checking") return {step:"check", title:"Connection checks in progress", message:"Wait for source cleanup. No robot service is being launched."};
    const technical = Object.entries(report.checks || {}).find(([name, check]) => check.passed === false &&
      check.status !== "not_checked" && check.message !== "Not run" &&
      !["recording_metadata", "qualifying_configuration", "limits_verification"].includes(name));
    if (technical) {const problem = issueFor(...[technical[0], technical[1].message]); return {step:"check", title:problem.title, message:technical[1].message, help:problem.key};}
    return {step:"collect", title:"Review recording requirements", message:"Connectivity is separate from identity, operator timing and verified limits. Actual session preflight runs again."};
  }

  let metadata = null, selectedStep = "prepare", initialized = false, lastError = "", lastUpdate = "";
  const steps = {prepare:"setup-prepare", configure:"setup-configure", check:"readiness-panel", collect:"setup-collect"};

  /** Navigate the guided view without creating sessions, opening sources or issuing commands. */
  function go(step) {
    if (step === "record") {screen("record"); return;}
    screen("new"); selectedStep = step in steps ? step : "prepare";
    for (const [name, id] of Object.entries(steps)) $(id).hidden = name !== selectedStep;
    $("mode-help").hidden = selectedStep !== "prepare";
    $("setup-intro").hidden = selectedStep !== "configure";
    $("setup-facts").closest("article").hidden = selectedStep !== "configure";
    document.querySelectorAll("[data-setup-step]").forEach(node => {
      if (node.dataset.setupStep === selectedStep) node.setAttribute("aria-current", "step");
      else node.removeAttribute("aria-current");
    });
    update();
  }

  /** Read only the allowlisted committed document selected by a help topic. */
  async function documentHelp(id) {
    const result = await api(`help/${id}`);
    $("help-title").textContent = result.path;
    $("help-content").replaceChildren(text("p", "Read-only repository reference from this checkout. The console does not edit this file."), text("pre", result.text));
    if (!$("operator-help").open) $("operator-help").showModal();
  }

  /** Explain a refusal locally with ordered recovery steps, never an executable launch button. */
  function openHelp(key = "source", detail = "") {
    const item = help[key] || help.general;
    $("help-title").textContent = item.title;
    const content = $("help-content"); content.replaceChildren(text("p", item.summary));
    if (detail) content.append(text("pre", detail));
    const list = document.createElement("ol");
    for (const step of item.steps) list.append(text("li", step));
    content.append(list, button("Open repository reference", () => documentHelp(item.doc)));
    if (key === "source") {
      const candidate = document.createElement("details");
      candidate.append(text("summary", "Standalone alternative · unverified candidate, not a launch action"),
        text("p", "Only after confirming the installed driver supports both parameters, torque behavior is safe, and no process owns the port. Use the operator-validated driver terminal with its workspace already sourced. The default startup releases torque and is prohibited."),
        text("pre", 'export FOLLOWER_PORT=/dev/serial/by-id/REPLACE_WITH_FOLLOWER_USB_ID\nros2 run alicia_d_driver alicia_d_driver_node --ros-args \\\n  -p port:="$FOLLOWER_PORT" \\\n  -p joint_commands_enabled:=false \\\n  -p torque_off_on_start:=false'));
      content.append(candidate);
    }
    if (!$("operator-help").open) $("operator-help").showModal();
  }

  /** Keep actionable errors visible across successful polling; dismissal changes only the view. */
  function renderError(value) {
    const message = String(value?.message || value || "");
    if (message === lastError) return;
    lastError = message;
    if (!message) {$("notice").replaceChildren(); return;}
    const informational = /^(Console closed\.|Reindex corrections:)/.test(message);
    $("notice").dataset.level = informational ? "info" : "error";
    if(informational){$("notice").replaceChildren(text("strong", message.startsWith("Console") ? "Console closed" : "Catalog refreshed"), text("p",message),button("Dismiss message",()=>renderError("")));return;}
    const issue = issueFor("", message);
    $("notice").replaceChildren(text("strong", issue.title), text("p", message),
      button("What to do", () => openHelp(issue.key, message)),
      button("Dismiss message", () => renderError("")));
  }

  /** Organize existing named fields into readable groups without changing the submitted payload. */
  function groupFields() {
    const grid = $("new-form").querySelector(".form-grid");
    const groups = [
      ["Session and task", ["name", "task_id", "operator", "gripper_type", "scene"]],
      ["State and action inputs", ["state_source", "follower_topic", "action_source", "leader_topic"]],
      ["Recording settings", ["fps", "image_width", "image_height", "action_lookahead_steps", "state_startup_timeout_s", "guard_command_topic", "state_has_velocity"]],
      ["Required recording evidence", ["follower_serial", "leader_serial", "power_state_start", "target_episodes", "notes"]],
    ];
    for (const [index, [title, fields]] of groups.entries()) {
      const group = document.createElement("fieldset"); group.className = "setup-fields";
      group.dataset.order = String(index);
      group.append(text("legend", title));
      const inner = document.createElement("div"); inner.className = "form-grid"; group.append(inner);
      for (const name of fields) inner.append($("new-form").elements.namedItem(name).closest("label"));
      grid.append(group);
    }
    grid.classList.add("grouped-fields");
  }

  /** Render concise path cards and separate qualification blockers from technical diagnostics. */
  function update() {
    if (!initialized) return;
    const report = typeof readinessReport === "undefined" ? null : readinessReport;
    const settings = formValues();
    const current = Boolean(report?.checked_utc && report.status !== "expired" &&
      matchingReadinessSettings(report.form_settings, settings));
    const confirmed = state.demo || [...document.querySelectorAll("[data-readiness-confirm]")].every(node => node.checked);
    const softwareReady = $("software-panel").dataset.level === "ready";
    const signature = JSON.stringify([settings, report, current, confirmed, softwareReady, state.active_session?.id, state.demo, metadata, selectedStep]);
    if(signature === lastUpdate)return;
    lastUpdate = signature;
    const next = nextStep({settings, demo:state.demo, report, softwareReady, current:current && confirmed, active:state.active_session, confirmed});
    $("next-action-title").textContent = next.title;
    $("next-action-message").textContent = next.message;
    $("next-action-button").textContent = next.help ? "Show recovery steps" : (next.step === "record" ? "Open recording" : "Continue →");
    $("next-action-button").onclick = () => {
      if (next.help) {go(next.step); openHelp(next.help, next.message);}
      else {go(next.step); if (next.field) $("new-form").elements.namedItem(next.field)?.closest("label")?.querySelector("input:not([type=hidden]),select,textarea")?.focus();}
    };
    $("connection-paths").replaceChildren();
    for (const path of connectionPaths(report, current && confirmed)) {
      const card = document.createElement("article"); card.dataset.level = path.status;
      card.append(text("span", {ready:"Ready", not_ready:"Not ready", launching:"Launching", not_checked:"Not checked"}[path.status]), text("h3", path.title), text("p", path.message));
      card.append(button("Help and next steps", () => openHelp(path.help.key, path.message)));
      $("connection-paths").append(card);
    }
    const requirements = [];
    if (!state.demo) {
      if (!String(settings.follower_serial || "").trim() || !String(settings.power_state_start || "").trim()) requirements.push("metadata");
      const task = tasks.find(item => item.task_id === settings.task_id);
      if (!task || task.episode_window.min_episode_s === null) requirements.push("timing");
      if (!metadata?.limits_verified) requirements.push("limits");
    }
    $("recording-blockers").replaceChildren(text("h3", "Before qualifying recording"));
    for (const key of requirements) $("recording-blockers").append(button(help[key].title, () => openHelp(key)));
    if (!requirements.length) $("recording-blockers").append(text("p", state.demo ? "Synthetic demo never qualifies as physical evidence." : "Declared requirements supplied. Actual session preflight and operator safety review still apply."));
    $("collection-requirements").textContent = requirements.length ? requirements.map(key => help[key].summary).join(" ") : "Review the latest connection results, then create your session. The writer and source guards run again on open.";
    const requiredMissing = missingInputs(settings, state.demo).length > 0;
    $("create").disabled = requiredMissing || (!state.demo && requirements.length > 0);
    $("smoke").disabled = requiredMissing || (!state.demo && requirements.includes("metadata")) || Boolean(state.active_session);
    $("create").title = $("create").disabled ? "Complete the listed configuration, identity, timing and verified-limits requirements." : "Create a named session; opening it performs shared source preflight.";
    $("smoke").title = $("smoke").disabled ? "Complete connection fields, manufacturer identity and observed power first." : "Starts one temporary capture after shared preflight; never counted.";
  }

  /** Initialize guided navigation after the existing recorder surface and presets are ready. */
  async function initialize() {
    if (initialized) return;
    initialized = true; groupFields();
    document.querySelectorAll("[data-setup-step]").forEach(node => node.onclick = () => go(node.dataset.setupStep));
    document.querySelectorAll("[data-go-step]").forEach(node => node.onclick = () => go(node.dataset.goStep));
    document.querySelectorAll("[data-guide-help]").forEach(node => node.onclick = () => openHelp(node.dataset.guideHelp));
    document.querySelectorAll("[data-guide-doc]").forEach(node => node.onclick = () => safe(() => documentHelp(node.dataset.guideDoc)));
    $("open-help").onclick = () => openHelp("source");
    $("close-help").onclick = () => $("operator-help").close();
    $("new-form").addEventListener("invalid", () => go("configure"), true);
    $("new-form").addEventListener("input", update);
    $("new-form").addEventListener("change", update);
    $("readiness-confirmations").addEventListener("change", update);
    const smokeAction = $("smoke").onclick;
    $("smoke").onclick = event => {if (confirm("Start one disposable 20-second capture after preflight? Verify physical safety and framing first. It never counts and is deleted when its sources close.")) smokeAction(event);};
    if (state.active_session) screen("record");
    else if (!sessions.length) go("prepare");
    else {selectedStep = "prepare"; for (const [name, id] of Object.entries(steps)) $(id).hidden = name !== selectedStep;}
    try {
      metadata = await api("operator-guide");
      const env = metadata.environment;
      $("environment-facts").textContent = `Host: ${env.host} · ROS: ${env.ros_distribution} · domain: ${env.ros_domain} · middleware: ${env.middleware} · Python: ${env.python}`;
    } catch (reason) {$("environment-facts").textContent = `Environment details unavailable: ${reason.message}. Use the committed runbook and current launch terminal.`;}
    update();
  }

  return {issueFor, connectionPaths, missingInputs, nextStep, openHelp, documentHelp, renderError, update, initialize, go};
})();
globalThis.SynriaGuide = SynriaGuide;
if (typeof document !== "undefined") {
  window.addEventListener("console-ready", () => safe(SynriaGuide.initialize), {once:true});
  window.addEventListener("console-state", SynriaGuide.update);
  window.addEventListener("console-readiness", SynriaGuide.update);
  if (window.synriaConsoleReady) safe(SynriaGuide.initialize);
}
