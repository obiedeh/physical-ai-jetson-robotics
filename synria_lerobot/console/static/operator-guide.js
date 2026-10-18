"use strict";
/** Guide operators through existing recorder paths without adding device or command capabilities. */
const SynriaGuide = (() => {
  const help = {
    source: {title:"Prepare the arm feed", summary:"The console listens; it does not start the driver or control the arm.", steps:[
      "Secure the arm, clear the bench and verify the approved power and controlled-stop procedure. Never release torque on an unsupported arm.",
      "Keep the existing leader–follower hardware-sync teleoperation unchanged. Reuse its verified read-only arm feed; never start a second owner of the follower USB connection.",
      "The source must publish the selected follower topic with arm writes disabled and startup torque unchanged. Choosing ros2_control under Change setup does not launch it or verify those properties.",
      "For ros2_control, use only the operator-verified read-only configuration. No verified startup command is recorded here. Do not use the write-enabled D2 policy launch as a recording shortcut.",
      "Compare this console’s displayed ROS domain and middleware with the source terminal. The topic and domain must agree. A timeout alone cannot distinguish an absent publisher from discovery or configuration problems.",
      "After the approved source is running, return to Check setup and run the setup check again. If the setup is not verified, stop here and complete the safety record; do not guess flags or force readiness."], doc:"runbook"},
    timeout: {title:"No arm feed received", summary:"The console subscribed, but no reading arrived before the startup wait ended.", steps:[
      "Confirm the arm feed you started is actually running. This console never starts it, and a USB entry is not proof of a publisher.",
      "Check the configured follower topic. Compare ROS domain and middleware with the environment shown on the Help page.",
      "Use the state-source procedure in Operator help. Do not start another process on an already owned port or change working teleoperation.",
      "Retry the connection check after resolving the cause. A longer first-sample wait only accommodates discovery; it cannot create missing data or relax steady-state freshness.",
      "Cameras have not failed merely because they remain Not checked: the setup check deliberately checks the arm feed and command publishers before opening them."], doc:"runbook"},
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
      "Refresh camera IDs under Change setup, then identify the C10 wrist camera and fixed front webcam.",
      "Do not assign index0 and index1 of the same USB identity to different roles.",
      "Listing a device does not open or verify it. The connection check and later live framing review are still required."], doc:"runbook"},
    rate: {title:"Requested rate exceeds available state", summary:"Recording cannot sample faster than the measured incoming state stream.", steps:[
      "Read the measured rate in technical evidence. The requested, incoming and achieved sample rates are different facts.",
      "Choose a supported explicit FPS under Change setup; 15 and 30 are vendor candidates, not guarantees.",
      "For an existing dataset, changing FPS requires a separate dataset/session. Do not bypass the rate gate or increase a driver’s rate speculatively."], doc:"protocol"},
    freshness: {title:"Cameras and arm feed are out of step", summary:"The newest camera frames and the newest arm reading were taken too far apart, or a source is stale.", steps:[
      "Press Check again. One out-of-step read can be a startup effect; a repeat is a real problem.",
      "If it repeats, look for a camera that is slow or busy (another app, a dim room forcing a long exposure) or an arm feed that has stopped updating, and fix that cause.",
      "Do not raise the alignment or freshness limits to hide the failure. A longer startup wait does not relax the steady-state rule.",
      "Technical detail: alignment is judged on fresh samples from the arm feed, both cameras and the USB leader when used; while any of them is missing the result reads Not checked rather than failed."], doc:"protocol"},
    publishers: {title:"Command path is not quiet", summary:"A publisher or graph-query failure prevents read-only recording.", steps:[
      "Inspect the exact guarded topic and error in technical evidence. A publisher blocks recording even if it is not currently sending.",
      "Follow the operator-approved procedure to resolve the source configuration. Do not kill publishers, disable the guard or reconfigure working teleoperation just to pass.",
      "Recording needs no policy bridge, arming message or command publisher. Retry only after the approved read-only arrangement is established."], doc:"runbook"},
    metadata: {title:"Complete recording identity and power", summary:"These facts are required for recording; the setup check, which records nothing, can still run without them.", steps:[
      "Under Change setup → Required recording evidence, choose the follower manufacturer serial from the dropdown if it is already known, or choose manual entry and copy it exactly from the follower arm label or verified equipment record. The ROS state feed and USB adapter name are not the arm serial.",
      "In the same section, enter Starting power state using the approved status display or current operator observation (for example, the observed powered and torque state). Do not toggle power or torque to fill the field; if uncertain, leave it unknown and stop before recording.",
      "If identity cannot be established, leave it unknown and stop before recording. Connection checks still require all physical safety confirmations."], doc:"safety"},
    timing: {title:"Register the real task window", summary:"This task has no timed take window yet, so its takes cannot count toward the dataset.", steps:[
      "Time the actual task under the approved physical procedure; do not infer duration from a software test.",
      "Prospectively record observations, chosen minimum/maximum, rationale, operator and date in DECISIONS.md. Update that task in config/synria_tasks.json through reviewed repository changes.",
      "The console cannot set or override this window. A practice run is separate, never counts, and still needs identity, safety and the setup check."], doc:"decisions"},
    limits: {title:"Verify the physical limits", summary:"Candidate ranges are present, but the file is explicitly unverified, so takes cannot count toward the dataset.", steps:[
      "Open config/synria_limits.yaml below to inspect the six candidate arm-joint ranges and the gripper stroke ranges used by the recorder's quality gate.",
      "The Alicia-D checkout documents a Joint5 soft cap for its Isaac ghost-arm path and self-collision margins/spheres for that simulation. Those are not, by themselves, verified physical-arm limits or a collision check used by this recorder; do not copy simulation collision geometry into the scalar joint-range file.",
      "Compare the candidate values with approved evidence for this exact physical arm and installed gripper. Record the source and operator verification through a reviewed change; do not set verified_by or verified_on just to clear the status.",
      "Until an operator-approved verification is recorded, no take counts toward the dataset. This console neither edits limits nor authorizes motion."], doc:"limits"},
    pose: {title:"Move the arm off its rest pose", summary:"The arm is parked beyond a recording bound, usually the folded rest position; the console only listens and cannot move it.", steps:[
      "Move the arm to an upright position, away from the folded rest pose, and press Check again.",
      "Start takes from a slightly raised rest. Any frame back at the park fails the joint limit quality check when the take is saved.",
      "Do not edit the limits file to make the check pass. The bounds are verified on the bench with the limits verification runbook, and only that changes them."], doc:"limits"},
    disk: {title:"Restore safe recording space", summary:"The workspace has insufficient available space or a filesystem operation failed.", steps:[
      "Do not discard pending frames just to retry. Keep the process alive while resolving a save failure.",
      "Check the external workspace’s capacity and permissions. Preserve datasets, sidecars, recording locks and journals.",
      "After freeing space through reviewed file management, retry saving the same take or run the setup check again. Do not lower the disk floor to conceal the problem."], doc:"runbook"},
    recovery: {title:"Preserve the dataset for recovery", summary:"The writer detected a lock, journal or recovery inconsistency.", steps:[
      "Stop new collection and keep the exact error. Do not remove the lock or journal, rename sidecars or overwrite the dataset.",
      "Confirm whether another recorder owns the dataset. Never open a second writer or force a deletion.",
      "Use operator-reviewed transaction recovery. Review remains read-only where available; writes and deletion stay blocked until the cause is resolved."], doc:"runbook"},
    save: {title:"Keep the frames and retry safely", summary:"A stopped take is not safe on disk until its save completes.", steps:[
      "Keep the console process running. Closing a browser tab does not stop capture, but terminating the process loses pending frames.",
      "Inspect the save error and resolve storage or camera issues. Use Retry save to keep the same frames and label.",
      "If recovery is blocked, preserve locks and journals and stop writes. Throw away is only for an unusable capture and needs a reason; Aborted is for a take you never attempted; keep genuine task failures as failures.",
      "After a save, review both views, the traces, the quality checks and the overhead photo before scaling up."], doc:"runbook"},
    contract: {title:"Keep dataset settings consistent", summary:"The existing dataset does not match the proposed session contract or capture provenance.", steps:[
      "Resume with the exact original task, image/camera settings, FPS, action source, lookahead, gripper and velocity contract.",
      "If the physical setup or contract intentionally changed, create a separate dataset/session. Do not edit saved sidecars to force a match.",
      "Preserve the original dataset and evidence. Reindex rebuilds the catalog; it cannot repair or override a contract mismatch."], doc:"protocol"},
    network: {title:"Reconnect without losing a take", summary:"The browser cannot confirm the server’s current state.", steps:[
      "Do not assume the recorder stopped. The take lives in the server and may continue until its time limit.",
      "Reconnect to the same console and look at the restored state before repeating any recording action. Do not launch another instance or kill the process while frames may be pending.",
      "If the process was restarted intentionally, open its newest launch URL. Save/discard decisions remain explicit; no request is retried automatically."], doc:"runbook"},
    auth: {title:"Reopen this console’s launch URL", summary:"The page does not have the current per-launch authorization token.", steps:[
      "Open the URL printed by the running console, including its token fragment. A restart changes that token.",
      "Use the exact local host and port; do not paste the token into a public report.",
      "Reopening a view does not stop or restart capture. Inspect the restored session before taking action."], doc:"runbook"},
    training: {title:"Record for the best training results", summary:"The model learns only what the takes show it: vary the placement, start off the fold, show recoveries, settle before lifting.", steps:[
      "Vary the placement on every take with the Randomize 1x, 2x and 3x buttons, rotated across takes, so the dataset covers the reachable band. Fixed-spot takes stopped teaching anything after about 25; varied takes were the only ones that produced picks from new spots.",
      "Start every take a little off the folded rest pose; at the fold Joint2 sits within a degree of its recording bound.",
      "In about one take in five, start the grasp slightly off, back off and re-grasp, so the model sees a recovery and not only perfect approaches.",
      "Close the gripper, let it settle for a beat, then lift; finish the way the scene scores the task: for pick_cup the cup held clear of the table, for cup_pick_and_place the cup released upright inside the mark with the gripper backed away.",
      "Keep the pace and the camera framing unchanged within a session, label honestly (Aborted for a take you did not attempt), and aim for 50 or more takes in one session."], doc:"protocol"},
    general: {title:"Resolve the reported check", summary:"The operation was refused; no successful result should be assumed.", steps:[
      "Read the exact error and expand technical evidence. Correct the indicated field or prerequisite before retrying.",
      "Do not bypass safety checks, edit dataset evidence or restart a process with an unsaved take.",
      "Use the full operator procedure below for the required order. If the error concerns unverified hardware behavior, stop physical setup until it is documented."], doc:"runbook"},
  };

  /** Classify known refusal messages for assistance only; never use text matching to authorize actions. */
  function issueFor(name = "", message = "") {
    const value = `${name} ${message}`.toLowerCase();
    const rules = [
      ["auth", /token|origin|host is not/], ["network", /failed to fetch|network|load failed|connection refused/],
      ["pose", /beyond a recording limit|rest position|rest pose/],
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
      ["follower", "Arm feed", ["follower_sample", "state_rate", "state_source"]],
      ["action", "Leader arm", ["action_sample"]],
      ["wrist", "Wrist camera", ["wrist_sample", "wrist_camera"]],
      ["front", "Overhead camera", ["front_sample", "front_camera"]],
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

  /** A failed check that belongs to no input path, such as disk space or dataset recovery. */
  function otherFailure(report, current = true) {
    if (!current) return null;
    const pathKeys = new Set(["follower_sample", "state_rate", "state_source", "action_sample", "wrist_sample", "wrist_camera", "front_sample", "front_camera"]);
    const skip = new Set(["recording_metadata", "qualifying_configuration", "limits_verification", "operator_safety", "diagnostic"]);
    const found = Object.entries(report?.checks || {}).find(([name, check]) => check.passed === false &&
      check.status !== "not_checked" && check.status !== "launching" && check.message !== "Not run" && !pathKeys.has(name) && !skip.has(name));
    if (!found) return null;
    return {name:found[0], message:found[1].message, help:issueFor(found[0], found[1].message)};
  }

  /** Qualifying requirements still missing for these settings: metadata, timing, limits. */
  function requirements(settings) {
    const needed = [];
    if (state.demo) return needed;
    if (!String(settings.follower_serial || "").trim() || !String(settings.power_state_start || "").trim()) needed.push("metadata");
    const task = tasks.find(item => item.task_id === settings.task_id);
    if (!task || task.episode_window.min_episode_s === null) needed.push("timing");
    if (!metadata?.limits_verified) needed.push("limits");
    return needed;
  }

  let metadata = null, initialized = false, lastError = "", lastUpdate = "";

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

  /** Fill the Help page with one card per procedure; cards open the same dialog as in-context help. */
  function renderHelpPage() {
    const container = $("help-topics");
    if (!container) return;
    container.replaceChildren();
    for (const [key, item] of Object.entries(help)) {
      const card = document.createElement("article"); card.className = "card";
      card.append(text("h3", item.title), text("p", item.summary), button("Show steps", () => openHelp(key)));
      container.append(card);
    }
  }

  /** Keep actionable errors visible across successful polling; dismissal changes only the view. */
  let noticeRoute = null;
  function renderError(value, options = {}) {
    const message = String(value?.message || value || "");
    if (message === lastError && !options.route) return;
    lastError = message;
    noticeRoute = options.route || null;
    if (!message) {$("notice").replaceChildren(); delete $("notice").dataset.level; return;}
    const informational = Boolean(options.route) || /^(Console closed\.|Reindex corrections:)/.test(message);
    $("notice").dataset.level = informational ? "info" : "error";
    if(informational){const title = options.title || (message.startsWith("Console") ? "Console closed" : message.startsWith("Reindex") ? "Catalog refreshed" : "Done");$("notice").replaceChildren(text("strong", title), text("p",message),button("Dismiss message",()=>renderError("")));return;}
    const issue = issueFor("", message);
    $("notice").replaceChildren(text("strong", issue.title), text("p", message),
      button("What to do", () => openHelp(issue.key, message)),
      button("Dismiss message", () => renderError("")));
  }

  /** Organize existing named fields into readable groups without changing the submitted payload. */
  function groupFields() {
    const grid = $("new-form").querySelector(".form-grid");
    const groups = [
      ["Operator and gripper", ["operator", "gripper_type", "scene"]],
      ["State and action inputs", ["state_source", "follower_topic", "action_source", "leader_topic"]],
      ["Recording settings", ["fps", "image_width", "image_height", "action_lookahead_steps", "state_startup_timeout_s", "guard_command_topic", "state_has_velocity"]],
      ["Required recording evidence", ["follower_serial", "leader_serial", "power_state_start", "notes"]],
    ];
    for (const [index, [title, fields]] of groups.entries()) {
      const group = document.createElement("fieldset"); group.className = "setup-fields";
      group.dataset.order = String(index);
      group.append(text("legend", title));
      const inner = document.createElement("div"); inner.className = "form-grid"; group.append(inner);
      for (const name of fields) { const field = $("new-form").elements.namedItem(name); const node = field instanceof RadioNodeList ? field[0] : field; const box = node ? (node.closest("fieldset.field") || node.closest("label")) : null; if (box) inner.append(box); }
      grid.append(group);
    }
    grid.classList.add("grouped-fields");
  }

  /** Refresh the wizard when readiness or form state changes; tiles and blockers live there. */
  function update() {
    if (!initialized) return;
    if (globalThis.SynriaWizard) SynriaWizard.refresh();
  }

  /** Initialize guided navigation after the existing recorder surface and presets are ready. */
  async function initialize() {
    if (initialized) return;
    initialized = true; groupFields();
    document.querySelectorAll("[data-guide-help]").forEach(node => node.onclick = () => openHelp(node.dataset.guideHelp));
    document.querySelectorAll("[data-guide-doc]").forEach(node => node.onclick = () => safe(() => documentHelp(node.dataset.guideDoc)));
    $("close-help").onclick = () => $("operator-help").close();
    renderHelpPage();
    $("new-form").addEventListener("input", update);
    $("new-form").addEventListener("change", update);
    $("readiness-confirmations").addEventListener("change", update);
    const smokeAction = $("smoke").onclick;
    $("smoke").onclick = event => {if (confirm("Start a practice run? It opens the cameras and arm feed, records one take and throws it away afterwards. Check safety and framing first.")) smokeAction(event);};
    SynriaWizard.initialize();
    try {
      metadata = await api("operator-guide");
      const env = metadata.environment;
      $("environment-facts").textContent = `Host: ${env.host} · ROS: ${env.ros_distribution} · domain: ${env.ros_domain} · middleware: ${env.middleware} · Python: ${env.python}`;
    } catch (reason) {$("environment-facts").textContent = `Environment details unavailable: ${reason.message}. Use the committed runbook and current launch terminal.`;}
    update();
  }

  /** A neutral confirmation shown on one route and cleared on the next navigation. */
  function clearNoticeOnLeave() {
    if ($("notice").dataset.level === "info" && noticeRoute && window.SynriaRouter.current()?.path !== noticeRoute) renderError("");
  }

  return {issueFor, connectionPaths, otherFailure, missingInputs, requirements, openHelp, clearNoticeOnLeave, documentHelp, renderError, renderHelpPage, update, initialize};
})();
globalThis.SynriaGuide = SynriaGuide;
if (typeof document !== "undefined") {
  window.addEventListener("console-ready", () => safe(SynriaGuide.initialize), {once:true});
  window.addEventListener("console-state", SynriaGuide.update);
  window.addEventListener("console-readiness", SynriaGuide.update);
  window.addEventListener("console-route", SynriaGuide.clearNoticeOnLeave);
  if (window.synriaConsoleReady) safe(SynriaGuide.initialize);
}
