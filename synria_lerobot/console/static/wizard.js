"use strict";
/** The new-session wizard: three linear steps, one fixed footer, nothing automatic but the check.
 *
 * Choose task, Check setup, Start. The primary button is disabled until the step is complete
 * and the footer lists what is left, in order. Back never loses input: the draft lives in local
 * storage and survives refresh. The only automatic action is the read-only setup check, which
 * runs on entering Check setup once the bench checks are confirmed (at once in the synthetic
 * demo). Expiry of a check never changes the step; Start reruns it in place instead.
 */
const SynriaWizard = (() => {
  const STEPS = ["task", "check"];
  const TITLES = {task:"Choose task", check:"Check setup"};
  const DRAFT_KEY = "synria-wizard-draft";
  const CHECK_MAX_AGE_S = 60;
  const FIELD_WORDS = {name:"name the session", gripper_type:"choose the gripper size", scene:"describe the scene",
    state_source:"choose the arm feed", wrist_camera:"choose the wrist camera", front_camera:"choose the overhead camera",
    operator:"enter your name", fps:"choose the rate", follower_serial:"choose the follower serial",
    power_state_start:"confirm the power state", task_id:"choose a task"};
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const STANDARD_POWER = "Arm powered and following the leader";
  let step = "task", checking = false, starting = false, armed = false, lastReason = "", autoName = true, lastTiles = "";

  /** Storage wrappers: a blocked store must never break the flow, only forget the draft. */
  // Per-tab storage: a refresh restores this tab's draft; another tab starts clean.
  function readDraft() { try { return JSON.parse(sessionStorage.getItem(DRAFT_KEY) || "null") || {}; } catch (reason) { return {}; } }
  function writeDraft(draft) { try { sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draft)); } catch (reason) { /* storage blocked: the draft only lives in the page */ } }
  function clearDraft() { try { sessionStorage.removeItem(DRAFT_KEY); } catch (reason) { /* nothing to clear */ } }

  /** Save every named field except safety confirmations, which are never restored. */
  function saveDraft() {
    const values = {};
    for (const field of $("new-form").elements) {
      if (!field.name || field.name === "power_choice") continue;
      if (field.type === "hidden" && !field.hasAttribute("data-presets") && field.name !== "power_state_start") continue;
      if (field.type === "radio") { if (field.checked) values[field.name] = field.value; continue; }
      const value = field.type === "checkbox" ? field.checked : field.value;
      if (value === "") continue; // blanks are not choices; suggestions may still fill them
      values[field.name] = value;
    }
    if (values.follower_serial === "__manual__") values.follower_serial_custom = $("follower-serial-custom").value;
    values.scene_note = $("scene-note").value;
    writeDraft({values, step, autoName});
  }

  /** Put saved values back without touching confirmations or inventing identity. */
  function restoreDraft() {
    const draft = readDraft();
    for (const [name, value] of Object.entries(draft.values || {})) {
      const field = $("new-form").elements.namedItem(name);
      if (!field) continue;
      if (field instanceof RadioNodeList) { field.value = String(value); if (typeof editedSetupFields !== "undefined") editedSetupFields.add(name); continue; }
      if (field.type === "hidden" && !field.hasAttribute("data-presets") && name !== "power_state_start") continue;
      if (field.tagName === "SELECT" && ![...field.options].some(option => option.value === String(value))) continue;
      if (field.type === "checkbox") field.checked = Boolean(value); else field.value = String(value);
      if (typeof editedSetupFields !== "undefined") editedSetupFields.add(name);
      const preset = document.querySelector(`[data-preset-for="${name}"]`);
      if (preset) { preset.value = [...preset.options].some(option => option.value === String(value)) ? String(value) : "custom"; preset.dispatchEvent(new Event("change", {bubbles:true})); }
    }
    if (draft.values?.follower_serial === "__manual__") { $("follower-serial-custom").value = draft.values.follower_serial_custom || ""; $("follower-serial-custom").hidden = false; }
    $("scene-note").value = draft.values?.scene_note || "";
    showPower(draft.values?.power_state_start || "");
    autoName = draft.autoName !== false;
    return draft;
  }

  /** The chosen registry task, if any. */
  function chosenTask() { return tasks.find(task => task.task_id === $("task").value) || null; }
  /** Sentence-case task title from its identifier. */
  function taskTitle(task) { return window.SynriaRecord.taskLabel(task); }
  /** Library tasks sit under their goal; built-in tasks come first. */
  function isLibrary(task) { return task.task_source === "library"; }
  /** Plain words for operator-facing task text. */
  function plain(value) { return String(value).replace(/front-camera still/g, "overhead photo").replace(/the operator/g, "you"); }
  function timed(task) { return task.episode_window.min_episode_s !== null; }
  function windowWords(task) { const w = task.episode_window; return timed(task) ? `${w.min_episode_s} to ${w.max_episode_s} s per take` : "Not set yet"; }

  /** Scene text sent to the recorder: the task's scene plus anything different today. */
  function sceneText() {
    const task = chosenTask();
    const base = task ? task.scene_requirements.join(". ") : "";
    const note = $("scene-note").value.trim();
    return base + (note ? `${base ? ". " : ""}Today: ${note}` : "");
  }
  /** Keep the hidden scene field in step with what is shown. */
  function syncScene() { $("new-form").elements.scene.value = sceneText(); }

  /** Show a stored power state through the two-choice control. */
  function showPower(value) {
    const radios = $("new-form").elements.power_choice;
    if (value === STANDARD_POWER) { radios.value = STANDARD_POWER; $("power-other").hidden = true; }
    else if (value) { radios.value = "other"; $("power-other").hidden = false; $("power-other").value = value; }
    $("new-form").elements.power_state_start.value = value;
  }
  /** Read the two-choice control into the hidden field the recorder receives. */
  function readPower() {
    const choice = $("new-form").elements.power_choice.value;
    $("power-other").hidden = choice !== "other";
    $("new-form").elements.power_state_start.value = choice === "other" ? $("power-other").value.trim() : choice;
  }

  /** A session name in words: task, day, and the count of sessions for that task today. */
  function suggestName() {
    const task = chosenTask(); if (!task) return;
    const field = $("new-form").elements.name;
    if (!autoName && field.value && !/^session-\d{4}/.test(field.value)) return;
    const now = new Date(), prefix = `${taskTitle(task)}, ${now.getDate()} ${MONTHS[now.getMonth()]}, `;
    const count = (typeof sessions === "undefined" ? [] : sessions).filter(s => s.task_id === task.task_id && String(s.name).startsWith(prefix)).length + 1;
    field.value = `${prefix}${count}`;
    if (typeof editedSetupFields !== "undefined") editedSetupFields.add("name");
  }

  /** Render one selectable card per registry task; selection only sets the form field. */
  function renderTaskCards() {
    const container = $("task-cards"); container.replaceChildren();
    let group = null;
    for (const task of tasks) {
      const heading = task.group || (isLibrary(task) ? task.goal_title : "Built-in tasks");
      if (heading !== group) { group = heading; const label = text("h4", heading); label.className = "task-group"; container.append(label); }
      const card = document.createElement("button"); card.type = "button"; card.className = "card task-card";
      card.setAttribute("role", "radio"); card.dataset.taskId = task.task_id;
      card.setAttribute("aria-checked", String($("task").value === task.task_id));
      card.append(text("h3", taskTitle(task)), text("p", plain(task.task_text)));
      const chip = text("span", timed(task) ? windowWords(task) : "Practice only until timed"); chip.className = "chip " + (timed(task) ? "chip-ok" : "chip-warn");
      card.append(chip);
      if (isLibrary(task)) { const mark = text("span", "Your task. Takes are saved but do not count toward dataset milestones."); mark.className = "chip library-chip"; card.append(mark); }
      card.onclick = () => { $("task").value = task.task_id; $("task").dispatchEvent(new Event("change", {bubbles:true})); };
      container.append(card);
    }
    if (!tasks.length) container.append(text("p", "No tasks are registered."));
    const entry = document.createElement("a"); entry.className = "card task-card teach-entry"; entry.href = window.SynriaRouter.href(window.SynriaRouter.routes.teach()); entry.dataset.route = window.SynriaRouter.routes.teach();
    entry.append(text("h3", "Teach something new"), text("p", "Type a sentence and get steps you can train one at a time."));
    container.append(entry);
  }

  /** After tasks change on the server: rebuild the select and the cards. */
  function reloadTasks() {
    if (typeof fillTaskOptions === "function") fillTaskOptions();
    renderTaskCards(); renderTaskDescription(); refresh();
  }

  /** Scene bullets and one plain status for the chosen task. */
  function renderTaskDescription() {
    const task = chosenTask(), box = $("task-description"); box.replaceChildren();
    if (!task) return;
    box.append(text("h3", "Scene before each take"));
    const list = document.createElement("ul");
    for (const item of task.scene_requirements) list.append(text("li", plain(item)));
    box.append(list, text("p", `Success: ${plain(task.success_rule)}`));
    if (isLibrary(task)) {
      box.append(text("p", `Your task from the goal "${task.goal_title}", ${window.SynriaRecord.taskLabel(task, {where: true})}, version ${task.library_version}. Takes record the step, its version and its place in the chain so they can be stitched together later.`));
      if (task.start_state) box.append(text("p", `Starts: ${task.start_state} Ends: ${task.end_state}`));
      const timeIt = button("Time it", () => openTiming(task));
      if (state.demo) { timeIt.disabled = true; timeIt.title = "Time it listens to the real arm; it is not available in demo mode."; }
      box.append(timeIt);
    }
    const status = text("p", timed(task) ? `Take length ${windowWords(task)}. Takes recorded in this session count toward the dataset.`
      : "Take length not set yet. You can do practice runs now. To record takes that count, time this task first.");
    status.className = "task-status"; box.append(status);
    if (!timed(task)) box.append(button("How to time this task", () => window.SynriaGuide.openHelp("timing")));
  }

  let timingTask = null, timingPoll = null;
  /** Paint the timing dialog from a status snapshot. */
  function renderTiming(status) {
    $("time-phase").textContent = status.error ? `Stopped: ${status.error}` : status.running ? `Listening. ${status.phase.charAt(0).toUpperCase()}${status.phase.slice(1)}.` : status.runs.length ? "Stopped." : "Not started.";
    const list = $("time-runs"); list.replaceChildren();
    for (const run of status.runs) list.append(text("li", `Move ${run.index}: ${run.duration_s.toFixed(1)} s${run.gripper_cycled ? ", gripper closed and reopened" : ""}, peak ${run.peak_deg.toFixed(0)} degrees on ${run.peak_joint}`));
    const proposal = status.proposed_window;
    $("time-proposal").hidden = !proposal;
    if (proposal && !$("time-min").dataset.touched) { $("time-min").value = String(proposal.min_episode_s); $("time-max").value = String(proposal.max_episode_s); }
    $("time-error").hidden = !status.error; $("time-error").textContent = status.error || "";
    $("time-start").hidden = status.running; $("time-start").textContent = status.runs.length ? "Time it again" : "Start timing";
    $("time-stop").hidden = !status.running;
    $("time-confirm").hidden = !proposal;
    $("time-start").classList.toggle("primary", !proposal);
  }
  /** Open the dialog for one library task and poll its progress while open. */
  function openTiming(task) {
    timingTask = task;
    $("time-title").textContent = `Time: ${taskTitle(task)}`;
    delete $("time-min").dataset.touched;
    $("time-dialog").showModal();
    const poll = async () => { if (!$("time-dialog").open || !timingTask) return; renderTiming(await api(`library/${timingTask.task_id}/time`)); };
    safe(poll);
    clearInterval(timingPoll); timingPoll = setInterval(() => safe(poll), 700);
  }
  function wireTiming() {
    $("time-close").onclick = () => { clearInterval(timingPoll); $("time-dialog").close(); };
    $("time-dialog").addEventListener("close", () => clearInterval(timingPoll));
    $("time-start").onclick = () => safe(async () => { renderTiming(await api(`library/${timingTask.task_id}/time/start`, {topic: formValues().follower_topic || "/joint_states"})); });
    $("time-stop").onclick = () => safe(async () => { renderTiming(await api(`library/${timingTask.task_id}/time/stop`, {})); });
    for (const id of ["time-min", "time-max"]) $(id).addEventListener("input", () => { $("time-min").dataset.touched = "1"; });
    $("time-confirm").onclick = () => safe(async () => {
      const saved = await api(`library/${timingTask.task_id}/time/confirm`, {min_episode_s: Number($("time-min").value), max_episode_s: Number($("time-max").value)});
      clearInterval(timingPoll); $("time-dialog").close();
      tasks = await api("tasks"); reloadTasks();
      info("Take length saved", `${saved.name || timingTask.task_id}: ${saved.episode_window.min_episode_s} to ${saved.episode_window.max_episode_s} s per take, stored with version ${saved.version}.`, window.SynriaRouter.routes.wizard("task"));
    });
  }

  /** After a task is chosen: scene prefill, suggested name, cards and footer. */
  function onTaskChosen() { renderTaskCards(); renderTaskDescription(); syncScene(); suggestName(); $("task-session-fields").hidden = !chosenTask(); }

  /** Required fields still blank, in wizard order; serial and power count for leaving step 2. */
  function missingFields(values) {
    const missing = window.SynriaGuide.missingInputs(values, state.demo || state.simulation).filter(name => name !== "task_id");
    if (!state.demo && !state.simulation) for (const name of ["follower_serial", "power_state_start"]) if (!String(values[name] || "").trim()) missing.push(name);
    const order = ["name", "operator", "gripper_type", "state_source", "wrist_camera", "front_camera", "fps", "scene", "follower_serial", "power_state_start"];
    return order.filter(name => missing.includes(name));
  }

  /** A camera's stable ID without the usb- prefix and video suffix, for a summary line. */
  function shortCamera(id) { return String(id).replace(/^usb-/, "").replace(/-video-index\d+$/, "").slice(0, 40); }
  /** Relative time in words for a UTC timestamp. */
  function ago(iso) {
    const seconds = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000));
    if (!Number.isFinite(seconds)) return "earlier";
    if (seconds < 45) return "just now";
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
    return `at ${new Date(iso).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"})}`;
  }

  /** The container that holds one named field, for moving it inline. */
  function containerFor(name) { const field = $("new-form").elements.namedItem(name); const node = field instanceof RadioNodeList ? field[0] : field; return node ? (node.closest("fieldset.field") || node.closest("label")) : null; }

  /** Show missing required values inline in Your setup; remembered values stay behind Change setup. */
  function renderYourSetup() {
    const values = formValues();
    const parts = [];
    const chosen = chosenTask();
    if (chosen) parts.push(taskTitle(chosen));
    if (values.gripper_type) parts.push(`${values.gripper_type.replace("mm", " mm")} gripper`);
    if (!state.demo && !state.simulation && values.state_source) parts.push(values.state_source === "ros2_control" ? "ros2_control arm feed" : "standalone driver arm feed");
    if (state.demo) parts.push("demo arm feed and cameras");
    if (state.simulation) parts.push("SIMULATION: Isaac sim arm and cameras, leader feed as action");
    else if (values.wrist_camera && values.front_camera) parts.push(`wrist ${shortCamera(values.wrist_camera)}, overhead ${shortCamera(values.front_camera)}`);
    if (values.fps) parts.push(`${values.fps} FPS`);
    if (values.operator) parts.push(`you: ${values.operator}`);
    $("setup-line").textContent = parts.length ? parts.join(", ") + "." : "Nothing remembered yet.";
    const inline = $("setup-inline");
    const wanted = missingFields(values).filter(name => !["scene", "name"].includes(name));
    if (!state.demo && !state.simulation) for (const name of ["follower_serial", "power_state_start"]) if (!wanted.includes(name)) wanted.push(name);
    for (const name of wanted) {
      const box = containerFor(name);
      if (box && box.parentElement !== inline) { inline.append(box); box.hidden = false; }
    }
    const note = $("scene-note").closest("label");
    if (chosenTask() && note && note.parentElement !== inline) inline.append(note);
    inline.hidden = !inline.children.length;
  }

  /** Is the last setup check current for these settings and younger than the expiry? */
  function checkIsCurrent() {
    const report = typeof readinessReport === "undefined" ? null : readinessReport;
    if (!report?.checked_utc || report.status === "expired" || report.status === "checking") return false;
    if (!matchingReadinessSettings(report.form_settings, formValues())) return false;
    const age = (Date.now() - Date.parse(report.checked_utc)) / 1000;
    return Number.isFinite(age) ? age < CHECK_MAX_AGE_S : true;
  }
  /** Does the last report describe these exact settings, whatever its age? */
  function reportMatches() { const report = typeof readinessReport === "undefined" ? null : readinessReport; return Boolean(report?.checked_utc && matchingReadinessSettings(report.form_settings, formValues())); }
  /** All bench checks ticked, or none required in the synthetic demo. */
  function confirmed() { return state.demo || confirmationNodes().every(node => node.checked); }
  /** The checklist that applies to this launch mode: the Isaac list in simulation, the bench list otherwise. */
  function confirmationNodes() { return [...document.querySelectorAll("[data-readiness-confirm]")].filter(node => Boolean(node.hasAttribute("data-sim-confirm")) === Boolean(state.simulation)); }
  window.SynriaConfirmations = confirmationNodes;
  /** Is a check running right now, by either the page or the server? */
  function checkRunning() { return checking || readinessBusy || (typeof readinessReport !== "undefined" && readinessReport?.status === "checking"); }

  /** The tiles: arm feed, wrist camera, overhead camera, storage, each with a state and words. */
  function tileRows() {
    const report = typeof readinessReport === "undefined" ? null : readinessReport;
    const matches = reportMatches(), running = checkRunning();
    const checks = matches ? report.checks : {};
    const paths = window.SynriaGuide.connectionPaths(matches ? report : null, matches);
    const leader = formValues().action_source === "leader";
    const rate = checks.state_rate_evidence?.details?.rate_hz;
    const tiles = [];
    const fromPath = (id, name, okText) => {
      const path = paths.find(item => item.id === id);
      if (running && (!path || ["launching", "not_checked"].includes(path.status))) return {name, state:"checking", text:"Checking"};
      if (!matches || !path || path.status === "not_checked") return {name, state:"unchecked", text:"Not checked yet"};
      if (path.status === "not_ready") return {name, state:"problem", text:path.message, help:path.help};
      return {name, state:"ok", text:okText(path)};
    };
    const arm = fromPath("follower", state.simulation ? "Sim arm feed" : "Arm feed", () => state.demo ? "Synthetic arm feed, fresh sample" : typeof rate === "number" ? `Receiving ${Math.round(rate)} readings per second` : "Receiving readings");
    if (!leader && arm.state === "ok") { const action = paths.find(item => item.id === "action"); if (action?.status === "not_ready") Object.assign(arm, {state:"problem", text:action.message, help:action.help}); }
    tiles.push(arm);
    if (leader) tiles.push(fromPath("action", state.simulation ? "Leader arm (real follower feed)" : "Leader arm", () => "Receiving leader readings"));
    for (const [id, name] of (state.simulation ? [["wrist", "Sim wrist camera"], ["front", "Sim front camera"]] : [["wrist", "Wrist camera"], ["front", "Overhead camera"]])) {
      const tile = fromPath(id, name, () => { const size = checks[`${id}_sample`]?.native_resolution; return `Fresh frame${Array.isArray(size) ? `, ${size.join("×")}` : ""}`; });
      if (matches && (report.thumbnails || []).includes(id)) tile.image = `/media/readiness/${id}?v=${encodeURIComponent(report.checked_utc)}`;
      tiles.push(tile);
    }
    if (typeof state.free_disk_bytes === "number") {
      const free = state.free_disk_bytes, floor = state.min_free_bytes || 0;
      tiles.push(free >= floor ? {name:"Storage", state:"ok", text:`${(free / 1e9).toFixed(1)} GB free`}
        : {name:"Storage", state:"problem", text:`${(free / 1e9).toFixed(1)} GB free, below the recording floor`, help:window.SynriaGuide.issueFor("disk_space", "insufficient disk space")});
    }
    const other = matches ? window.SynriaGuide.otherFailure(report, true) : null;
    if (other && other.name === "source_alignment") {
      const check = checks.source_alignment || {}, skew = check.measured_skew_s, limit = check.max_timestamp_skew_s;
      const text = typeof skew === "number" && typeof limit === "number"
        ? `The newest camera frames and arm reading were ${skew.toFixed(3)} s apart. The limit is ${limit.toFixed(3)} s.`
        : "The newest camera frames and arm reading were taken too far apart.";
      tiles.push({name:"Cameras and arm feed are out of step", state:"problem", text, help:other.help, fix:"Press Check again. If it repeats, open the steps.", action:"check again, the cameras and arm feed were out of step"});
    } else if (other) tiles.push({name:other.help.title, state:"problem", text:other.message, help:other.help, action:other.help.title.charAt(0).toLowerCase() + other.help.title.slice(1)});
    for (const tile of tiles) tile.action = tile.action || `fix the ${tile.name.toLowerCase()}`;
    return tiles;
  }

  const STATE_WORDS = {checking:"Checking", ok:"OK", problem:"Problem", unchecked:"Not checked yet"};

  /** Paint the tiles only when something changed, so thumbnails do not flicker. */
  function renderTiles() {
    const tiles = tileRows();
    const signature = JSON.stringify(tiles.map(tile => [tile.name, tile.state, tile.text, tile.image || ""]));
    if (signature !== lastTiles) {
      lastTiles = signature;
      const container = $("source-tiles"); container.replaceChildren();
      for (const tile of tiles) {
        const card = document.createElement("article"); card.className = "tile"; card.dataset.state = tile.state;
        card.append(text("span", STATE_WORDS[tile.state]), text("h4", tile.name));
        if (tile.image) { const image = document.createElement("img"); image.src = tile.image; image.alt = `${tile.name} sample from the last check`; card.append(image); }
        if (tile.state === "problem" && tile.help) {
          const list = document.createElement("dl");
          list.append(text("dt", "What is wrong"), text("dd", tile.text), text("dt", "Likely cause"), text("dd", tile.help.summary), text("dt", "What to do"), text("dd", tile.fix || tile.help.steps[0]));
          card.append(list, button("All the steps", () => window.SynriaGuide.openHelp(tile.help.key, tile.text)));
        } else card.append(text("p", tile.text));
        container.append(card);
      }
    }
    const report = typeof readinessReport === "undefined" ? null : readinessReport;
    $("sources-line").textContent = checkRunning() ? "Checking cameras and arm feed…" : reportMatches() ? `Last check ${ago(report.checked_utc)}.` : "Starts by itself once your setup is complete.";
    $("check-readiness").textContent = report?.checked_utc ? "Check again" : "Check now";
    $("check-readiness").disabled = checkRunning() || Boolean(state.active_session);
  }

  /** What is left on the current step, in order, as short verb phrases. */
  function remaining(name) {
    const values = formValues(), items = [];
    if (name === "task") { if (!chosenTask()) items.push("choose a task"); return items; }
    if (name === "check") {
      for (const field of missingFields(values)) items.push(FIELD_WORDS[field]);
      if (!confirmed()) items.push(state.simulation ? "tick the box to confirm you have checked the arm" : "tick the bench checks");
      if (state.simulation && window.SynriaIsaac) {
        const phase = window.SynriaIsaac.phase();
        if (phase === "start") { items.push("start Isaac record mode"); return items; }
        if (phase === "waiting") { items.push("wait for Isaac"); return items; }
      }
      if (reportMatches() && !checkRunning()) {
        const tiles = tileRows();
        for (const tile of tiles) if (tile.state === "problem") items.push(tile.action);
        if (!items.length && tiles.some(tile => tile.state !== "ok")) items.push("wait for the check");
      } else if (!items.length) items.push("wait for the check"); // it starts by itself once the rest is done
      if (starting) return ["wait for the check"];
      if (!String(values.name || "").trim()) items.push("name the session");
      // An untimed task or unverified limits lead to the practice run instead of blocking.
      return items;
    }
    return [];
  }

  /** Will takes from this session count toward the dataset, and if not, why not? */
  function countingStatus() {
    const task = chosenTask(), values = formValues();
    if (state.demo) return {counts:false, why:"Demo takes never count toward the dataset.", practiceRun:false};
    if (state.simulation) return {counts:true, why:"SIMULATION takes go to their own dataset and never count as physical data.", practiceRun:false};
    if (task && !timed(task)) return {counts:false, why:"This task is not timed yet. A practice run records one 20 second take and throws it away.", practiceRun:true};
    const needed = window.SynriaGuide.requirements(values).filter(key => key !== "metadata");
    if (needed.includes("limits")) return {counts:false, why:"Takes will not count toward the dataset until the joint limits are verified.", practiceRun:true};
    return {counts:true, why:"", practiceRun:false};
  }
  /** The primary button names what will happen: a counting session, or the practice run. */
  function primaryLabel() { const status = countingStatus(); return status.practiceRun ? "Start practice run" : "Start session"; }
  /** Prefill the wizard for another session of a task, keeping the remembered setup. */
  function prefill(taskId, target) {
    $("task").value = taskId; onTaskChosen();
    if (target) { const field = $("new-form").elements.target_episodes; field.value = String(target); const preset = document.querySelector("[data-preset-for=target_episodes]"); if (preset) { preset.value = [...preset.options].some(o => o.value === String(target)) ? String(target) : "custom"; preset.dispatchEvent(new Event("change", {bubbles:true})); } }
    saveDraft();
  }

  /** Decide whether a step is complete and, if not, say what is left in one plain sentence. */
  function completion(name) {
    const items = remaining(name);
    if (!items.length) return {ok:true};
    return {ok:false, reason:`${items.length === 1 ? "1 thing left" : `${items.length} things left`}: ${items.join(", ")}`, items};
  }

  /** Which steps are reachable: a later step needs every earlier step complete. */
  function furthestAllowed() {
    let allowed = "task";
    for (const name of STEPS.slice(1)) { if (!completion(STEPS[STEPS.indexOf(name) - 1]).ok) break; allowed = name; }
    return allowed;
  }

  /** Paint the stepper, the footer and the step-specific blocks from current state. */
  function refresh() {
    if (!tasks.length || !armed) return;
    const allowedIndex = STEPS.indexOf(furthestAllowed());
    for (const li of document.querySelectorAll("#wizard-steps li")) {
      const name = li.dataset.step, index = STEPS.indexOf(name), control = li.querySelector("button");
      const done = index < STEPS.indexOf(step) && completion(name).ok;
      li.dataset.state = name === step ? "current" : done ? "complete" : "upcoming";
      control.disabled = index > allowedIndex || name === step;
      if (name === step) control.setAttribute("aria-current", "step"); else control.removeAttribute("aria-current");
      control.querySelector("b").textContent = done ? "✓" : String(index + 1);
    }
    if (step === "check") { renderYourSetup(); renderTiles(); }
    const result = completion(step);
    const isaacPhase = step === "check" && state.simulation && window.SynriaIsaac ? window.SynriaIsaac.phase() : "ready";
    $("wizard-primary").disabled = isaacPhase === "start" ? !(window.SynriaIsaac.last() && window.SynriaIsaac.last().available) : (isaacPhase === "waiting" || !result.ok);
    $("wizard-back").textContent = step === "task" ? "Cancel" : "Back";
    $("wizard-primary").textContent = step !== "check" ? "Continue" : isaacPhase === "start" ? "Start Isaac record mode" : isaacPhase === "waiting" ? "Waiting for Isaac" : primaryLabel();
    const counting = step === "check" ? countingStatus() : {counts:true, why:"", practiceRun:false};
    const reason = isaacPhase === "start" ? "Isaac record mode is not running. Start it here; the checks run by themselves once it is up."
      : isaacPhase === "waiting" ? "Isaac is starting; the checks run by themselves when it is up."
      : result.ok ? (step === "check" ? (counting.why || "Everything is in place.") : "") : result.reason;
    if (reason !== lastReason) { $("wizard-reason").textContent = reason; lastReason = reason; }
    $("smoke").hidden = state.demo || step !== "check" || counting.practiceRun;
    $("smoke").disabled = state.demo || Boolean(state.active_session) || starting || missingFields(formValues()).length > 0 || !confirmed();
    $("smoke").title = $("smoke").disabled ? "A practice run needs the setup values, the follower serial and the power state." : "Opens the cameras and arm feed, records one 20 second take and throws it away.";
  }

  /** Show one step: pages, focus, scroll and the live announcement. Never navigates by itself. */
  function show(name) {
    step = STEPS.includes(name) ? name : "task";
    for (const page of STEPS) $(`step-${page}`).hidden = page !== step;
    $("bench-panel").hidden = Boolean(state.demo || state.simulation);
    $("sim-panel").hidden = !state.simulation;
    refresh();
    // The router calls show() once per arrival, so every arrival gets focus and an announcement.
    window.scrollTo({top:0, behavior:"auto"});
    $(`step-${step}-title`).focus({preventScroll:true});
    $("route-announcer").textContent = `Step ${STEPS.indexOf(step) + 1} of 2: ${TITLES[step]}`;
    if (step === "check") {
      const first = [...$("setup-inline").querySelectorAll("input:not([type=hidden]), select, textarea")].find(node => !node.hidden && node.id !== "scene-note" && !node.value && !(node.type === "radio" && $("new-form").elements[node.name].value));
      if (first) first.focus({preventScroll:true});
    }
    saveDraft();
    if (step === "check") maybeRunCheck();
  }

  /** Run the read-only setup check once per entry when the bench checks are confirmed. */
  async function maybeRunCheck() {
    if (step !== "check" || checkRunning() || !confirmed() || checkIsCurrent()) return;
    if (state.simulation && !(window.SynriaIsaac && window.SynriaIsaac.running())) return;
    if (window.SynriaGuide.missingInputs(formValues(), state.demo || state.simulation).length) return;
    checking = true; refresh();
    try { await runReadiness(); } finally { checking = false; refresh(); }
  }

  /** Create the session and open its cameras and arm feed; rerun a stale check in place first. */
  async function startSession() {
    starting = true; refresh();
    try {
      if (!checkIsCurrent()) {
        // Older than a minute: run it again right here; the tiles show Checking, then the result.
        checking = true; refresh();
        try { await runReadiness(); } finally { checking = false; }
        const tiles = tileRows();
        if (tiles.some(tile => tile.state !== "ok")) { starting = false; refresh(); return; }
      }
      const session = await api("sessions", {...formValues(), ...(state.simulation ? {arm_checked: confirmed()} : {})});
      await api(`session/${session.id}/open`, {});
      clearDraft();
      await refreshState(); await refreshSessions();
      window.SynriaRouter.go(window.SynriaRouter.routes.record(session.id));
      error("");
    } finally { starting = false; refresh(); }
  }

  /** Wire the footer, stepper, draft and change-setup disclosure after the page is ready. */
  function initialize() {
    if (armed) return;
    armed = true;
    renderTaskCards();
    wireTiming();
    // The quick trial is the default; an earlier session's target is remembered, a draft wins.
    const last = (typeof sessions === "undefined" ? [] : sessions).find(item => !["trashed", "purged"].includes(item.status));
    if (typeof editedSetupFields !== "undefined") editedSetupFields.add("target_episodes"); // the quick trial beats the runbook's 100
    if (last?.target_episodes) {
      const field = $("new-form").elements.target_episodes; field.value = String(last.target_episodes);
      const preset = document.querySelector("[data-preset-for=target_episodes]"); if (preset) { preset.value = [...preset.options].some(o => o.value === String(last.target_episodes)) ? String(last.target_episodes) : "custom"; preset.dispatchEvent(new Event("change", {bubbles:true})); }
    }
    const draft = restoreDraft();
    if (draft.values?.task_id) { $("task").value = draft.values.task_id; onTaskChosen(); }
    $("wizard-primary").onclick = () => safe(async () => {
      if (step === "check" && state.simulation && window.SynriaIsaac && window.SynriaIsaac.phase() === "start") { await window.SynriaIsaac.start(); refresh(); return; }
      if (!completion(step).ok) return;
      if (step === "task") window.SynriaRouter.go(window.SynriaRouter.routes.wizard("check"));
      else if (countingStatus().practiceRun) $("smoke").click();
      else await startSession();
    });
    $("wizard-back").onclick = () => {
      if (step === "task") window.SynriaRouter.go(window.SynriaRouter.routes.home());
      else window.SynriaRouter.go(window.SynriaRouter.routes.wizard(STEPS[STEPS.indexOf(step) - 1]));
    };
    document.querySelectorAll("[data-wizard-step]").forEach(node => { node.onclick = () => window.SynriaRouter.go(window.SynriaRouter.routes.wizard(node.dataset.wizardStep)); });
    const toggleSetup = open => { $("setup-configure").hidden = !open; $("change-setup").setAttribute("aria-expanded", String(open)); $("change-setup").textContent = open ? "Hide setup fields" : "Change setup"; };
    $("change-setup").onclick = () => toggleSetup($("setup-configure").hidden);
    $("done-setup").onclick = () => { toggleSetup(false); refresh(); $("step-check-title").focus(); };
    $("new-form").addEventListener("invalid", () => toggleSetup(true), true);
    let editTimer = null;
    const onEdit = event => {
      if (event.target.name === "power_choice" || event.target.id === "power-other") readPower();
      if (event.target.id === "scene-note") syncScene();
      if (event.target.name === "name" && event.isTrusted) autoName = false;
      if (event.target.name === "task_id") onTaskChosen();
      saveDraft(); refresh();
      // Once the last missing value is in, the check starts by itself after typing settles.
      clearTimeout(editTimer); editTimer = setTimeout(() => safe(maybeRunCheck), 1200);
    };
    $("new-form").addEventListener("input", onEdit);
    $("new-form").addEventListener("change", onEdit);
    $("readiness-confirmations").addEventListener("change", () => { refresh(); safe(maybeRunCheck); });
    $("new-form").onsubmit = event => { event.preventDefault(); safe(async () => { if (step === "check" && completion("check").ok) await startSession(); }); };
    window.addEventListener("console-readiness", refresh);
    window.addEventListener("console-isaac", () => { if (!$("new").hidden) { refresh(); safe(maybeRunCheck); } });
    window.addEventListener("console-state", () => { if (!$("new").hidden) refresh(); });
    window.addEventListener("console-sessions", () => { if (autoName && chosenTask()) { suggestName(); if (!$("new").hidden) refresh(); } });
  }

  return {STEPS, show, refresh, completion, remaining, initialize, furthestAllowed, current: () => step, checkIsCurrent, sceneText, tileRows, prefill, countingStatus, reloadTasks,
    flags: () => ({checking, starting, armed})};
})();
globalThis.SynriaWizard = SynriaWizard;
