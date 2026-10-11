"use strict";
/** The new-session wizard: three linear steps, one fixed footer, nothing automatic but the check.
 *
 * Choose task, Check setup, Start. The primary button is disabled until the step is complete
 * and the footer says why in plain words. Back never loses input: the draft lives in local
 * storage and survives refresh. The only automatic action is the read-only setup check, which
 * runs on entering Check setup once the operator has confirmed the bench checks. Expiry of a
 * check never changes the step; Start reruns it in place instead.
 */
const SynriaWizard = (() => {
  const STEPS = ["task", "check", "start"];
  const TITLES = {task:"Choose task", check:"Check setup", start:"Start"};
  const DRAFT_KEY = "synria-wizard-draft";
  const CHECK_MAX_AGE_S = 60;
  let step = "task", checking = false, starting = false, armed = false, lastReason = "";

  /** Storage wrappers: a blocked store must never break the flow, only forget the draft. */
  function readDraft() { try { return JSON.parse(localStorage.getItem(DRAFT_KEY) || "null") || {}; } catch (reason) { return {}; } }
  function writeDraft(draft) { try { localStorage.setItem(DRAFT_KEY, JSON.stringify(draft)); } catch (reason) { /* storage blocked: the draft only lives in the page */ } }
  function clearDraft() { try { localStorage.removeItem(DRAFT_KEY); } catch (reason) { /* nothing to clear */ } }

  /** Save every named field except safety confirmations, which are never restored. */
  function saveDraft() {
    const values = {};
    for (const field of $("new-form").elements) {
      if (!field.name || (field.type === "hidden" && !field.hasAttribute("data-presets"))) continue;
      const value = field.type === "checkbox" ? field.checked : field.value;
      if (value === "") continue; // blanks are not choices; suggestions may still fill them
      values[field.name] = value;
    }
    if (values.follower_serial === "__manual__") values.follower_serial_custom = $("follower-serial-custom").value;
    writeDraft({values, step});
  }

  /** Put saved values back without touching confirmations or inventing identity. */
  function restoreDraft() {
    const draft = readDraft();
    for (const [name, value] of Object.entries(draft.values || {})) {
      const field = $("new-form").elements.namedItem(name);
      if (!field || (field.type === "hidden" && !field.hasAttribute("data-presets"))) continue;
      if (field.tagName === "SELECT" && ![...field.options].some(option => option.value === String(value))) continue;
      if (field.type === "checkbox") field.checked = Boolean(value); else field.value = String(value);
      if (typeof editedSetupFields !== "undefined") editedSetupFields.add(name);
      const preset = document.querySelector(`[data-preset-for="${name}"]`);
      if (preset) { preset.value = [...preset.options].some(option => option.value === String(value)) ? String(value) : "custom"; preset.dispatchEvent(new Event("change", {bubbles:true})); }
    }
    if (draft.values?.follower_serial === "__manual__") { $("follower-serial-custom").value = draft.values.follower_serial_custom || ""; $("follower-serial-custom").hidden = false; }
    return draft;
  }

  /** The chosen registry task, if any. */
  function chosenTask() { return tasks.find(task => task.task_id === $("task").value) || null; }

  /** Describe a task window for operators: timed, or not timed yet. */
  function windowWords(task) {
    const w = task.episode_window;
    return w.min_episode_s === null ? "Not timed yet" : `${w.min_episode_s} to ${w.max_episode_s} s per take`;
  }

  /** Render one selectable card per registry task; selection only sets the form field. */
  function renderTaskCards() {
    const container = $("task-cards"); container.replaceChildren();
    for (const task of tasks) {
      const card = document.createElement("button"); card.type = "button"; card.className = "card task-card";
      card.setAttribute("role", "radio"); card.dataset.taskId = task.task_id;
      card.setAttribute("aria-checked", String($("task").value === task.task_id));
      card.append(text("h3", task.task_id.replaceAll("_", " ")), text("p", task.task_text), text("p", `Success: ${task.success_rule}`));
      const timing = text("span", windowWords(task)); timing.className = "chip " + (task.episode_window.min_episode_s === null ? "chip-warn" : "chip-ok");
      card.append(timing);
      card.onclick = () => { $("task").value = task.task_id; $("task").dispatchEvent(new Event("change", {bubbles:true})); renderTaskCards(); saveDraft(); refresh(); };
      container.append(card);
    }
    if (!tasks.length) container.append(text("p", "No tasks are registered."));
  }

  /** Words for the saved setup summary; missing required values are called out as the one thing to fix. */
  function renderSetupSummary() {
    const values = formValues();
    const rowsOut = [["Session name", values.name], ["Operator", values.operator], ["Gripper", values.gripper_type],
      ["Arm feed", state.demo ? "Synthetic demo" : values.state_source], ["Wrist camera", values.wrist_camera], ["Front camera", values.front_camera],
      ["Rate and image", `${values.fps || "?"} FPS · ${values.image_width}×${values.image_height}`], ["Follower serial", values.follower_serial], ["Starting power state", values.power_state_start], ["Scene", values.scene]];
    const fieldFor = {"Session name":"name", Operator:"operator", Gripper:"gripper_type", "Arm feed":"state_source", "Wrist camera":"wrist_camera", "Front camera":"front_camera", Scene:"scene"};
    const missing = window.SynriaGuide.missingInputs(values, state.demo);
    const list = $("setup-summary"); list.replaceChildren();
    for (const [label, value] of rowsOut) {
      const blank = value === undefined || value === null || String(value).trim() === "";
      const dt = text("dt", label), dd = text("dd", blank ? "not set" : String(value));
      if (blank) dd.className = missing.includes(fieldFor[label]) ? "missing" : "optional";
      list.append(dt, dd);
    }
    $("setup-missing").hidden = !missing.length;
    const names = missing.map(name => name.replaceAll("_", " ")).join(", ");
    if (missing.length) $("setup-missing").textContent = `${missing.length === 1 ? "One thing to fix" : "Still needed"}: ${names}. Press Change setup to fill it in.`;
  }

  /** Summarise what Start will do with the values the operator can see. */
  function renderStartSummary() {
    const values = formValues(), task = chosenTask();
    const list = $("start-summary"); list.replaceChildren();
    const rowsOut = [["Task", task ? task.task_id.replaceAll("_", " ") : "none"], ["Take length", task ? windowWords(task) : "unknown"],
      ["Target", `${values.target_episodes} takes`], ["Setup", `${values.gripper_type || "?"} gripper · ${values.fps} FPS · wrist ${values.wrist_camera || "?"} · front ${values.front_camera || "?"}`],
      ["Session name", values.name]];
    for (const [label, value] of rowsOut) list.append(text("dt", label), text("dd", String(value)));
  }

  /** Is the last setup check current for these settings and younger than the expiry? */
  function checkIsCurrent() {
    const report = typeof readinessReport === "undefined" ? null : readinessReport;
    if (!report?.checked_utc || report.status === "expired" || report.status === "checking") return false;
    if (!matchingReadinessSettings(report.form_settings, formValues())) return false;
    const age = (Date.now() - Date.parse(report.checked_utc)) / 1000;
    return Number.isFinite(age) ? age < CHECK_MAX_AGE_S : true;
  }

  /** All bench checks ticked, or none required in the synthetic demo. */
  function confirmed() { return state.demo || [...document.querySelectorAll("[data-readiness-confirm]")].every(node => node.checked); }

  /** The first failing camera or arm feed path, in words. */
  function failingPath() {
    const report = typeof readinessReport === "undefined" ? null : readinessReport;
    const paths = window.SynriaGuide.connectionPaths(report, true);
    return paths.find(path => path.status === "not_ready") || null;
  }

  /** Decide whether a step is complete and, if not, say why in one plain sentence. */
  function completion(name) {
    const values = formValues();
    if (name === "task") return chosenTask() ? {ok:true} : {ok:false, reason:"Choose a task to continue"};
    const missing = window.SynriaGuide.missingInputs(values, state.demo);
    if (missing.length) return {ok:false, reason:`Fill in ${missing.map(m => m.replaceAll("_", " ")).join(", ")} under Change setup to continue`};
    if (name === "check") {
      if (!confirmed()) return {ok:false, reason:"Tick the five bench checks to continue. The setup check starts when all five are ticked"};
      if (checking || readinessBusy) return {ok:false, reason:"Checking cameras and arm feed…"};
      const report = typeof readinessReport === "undefined" ? null : readinessReport;
      if (!report?.checked_utc || !matchingReadinessSettings(report.form_settings, values)) return {ok:false, reason:"The setup check has not run for these values yet"};
      if (report.status === "checking") return {ok:false, reason:"Checking cameras and arm feed…"};
      // Judge the recorded checks themselves: an expired snapshot keeps its verdict and is rerun at Start.
      const failing = failingPath();
      if (failing) return {ok:false, reason:`${failing.title}: ${failing.message}`};
      const other = window.SynriaGuide.otherFailure(report, true);
      if (other) return {ok:false, reason:`${other.help.title}: ${other.message}`};
      const paths = window.SynriaGuide.connectionPaths(report, true);
      if (!paths.every(path => path.status === "ready")) return {ok:false, reason:"A check did not finish. See the item above and its fix"};
      return {ok:true};
    }
    if (name === "start") {
      if (starting) return {ok:false, reason:"Checking cameras and arm feed…"};
      const requirements = window.SynriaGuide.requirements(values);
      if (requirements.length) {
        const words = {metadata:"follower serial and starting power state", timing:"task timing registered in the repository", limits:"operator-verified joint limits"};
        return {ok:false, reason:`A session that counts toward the dataset needs ${requirements.map(key => words[key]).join(", ")}. ${state.demo ? "" : "You can do a practice run instead."}`.trim()};
      }
      return {ok:true};
    }
    return {ok:false, reason:""};
  }

  /** Which steps are reachable: a later step needs every earlier step complete. */
  function furthestAllowed() {
    let allowed = "task";
    for (const name of STEPS) { if (!completion(name === "task" ? "task" : STEPS[STEPS.indexOf(name) - 1]).ok && name !== "task") break; allowed = name; }
    return allowed;
  }

  /** Paint the stepper, the footer and the step-specific summaries from current state. */
  function refresh() {
    if (!tasks.length) return;
    const allowedIndex = STEPS.indexOf(furthestAllowed());
    for (const li of document.querySelectorAll("#wizard-steps li")) {
      const name = li.dataset.step, index = STEPS.indexOf(name), button = li.querySelector("button");
      const done = index < STEPS.indexOf(step) && completion(name).ok;
      li.dataset.state = name === step ? "current" : done ? "complete" : "upcoming";
      button.disabled = index > allowedIndex || name === step;
      if (name === step) button.setAttribute("aria-current", "step"); else button.removeAttribute("aria-current");
      button.querySelector("b").textContent = done ? "✓" : String(index + 1);
    }
    if (step === "check") renderSetupSummary();
    if (step === "start") renderStartSummary();
    const result = completion(step);
    $("wizard-primary").disabled = !result.ok;
    $("wizard-primary").textContent = step === "start" ? "Start session" : "Continue";
    $("wizard-back").textContent = step === "task" ? "Exit to Home" : "Back";
    const reason = result.ok ? (step === "start" ? "Everything is in place." : "") : result.reason;
    if (reason !== lastReason) { $("wizard-reason").textContent = reason; lastReason = reason; }
    $("smoke").disabled = state.demo || Boolean(state.active_session) || starting || window.SynriaGuide.missingInputs(formValues(), state.demo).length > 0 || window.SynriaGuide.requirements(formValues()).includes("metadata");
    $("smoke").title = $("smoke").disabled ? "A practice run needs the setup values, the follower serial and the starting power state." : "Opens the cameras and arm feed, records one 20 second take and throws it away.";
    $("collection-requirements").textContent = step === "start" && !result.ok && !starting ? "" : "";
  }

  /** Show one step: pages, focus, scroll and the live announcement. Never navigates by itself. */
  function show(name) {
    step = STEPS.includes(name) ? name : "task";
    for (const page of ["task", "check", "start"]) $(`step-${page}`).hidden = page !== step;
    refresh();
    // The router calls show() once per arrival, so every arrival gets focus and an announcement.
    window.scrollTo({top:0, behavior:"auto"});
    $(`step-${step}-title`).focus({preventScroll:true});
    $("route-announcer").textContent = `Step ${STEPS.indexOf(step) + 1} of 3: ${TITLES[step]}`;
    saveDraft();
    if (step === "check") maybeRunCheck();
  }

  /** Run the read-only setup check once per entry when the bench checks are confirmed. */
  async function maybeRunCheck() {
    if (step !== "check" || checking || readinessBusy || !confirmed() || checkIsCurrent()) return;
    if (window.SynriaGuide.missingInputs(formValues(), state.demo).length) return;
    checking = true; refresh();
    try { await runReadiness(); } finally { checking = false; refresh(); }
  }

  /** Create the session and open its cameras and arm feed; rerun a stale check in place first. */
  async function startSession() {
    starting = true; refresh();
    $("start-check").hidden = true;
    try {
      if (!checkIsCurrent()) {
        $("start-check").hidden = false; $("start-check").dataset.level = "checking";
        $("start-check-title").textContent = "Checking cameras and arm feed"; $("start-check-message").textContent = "The last check is older than a minute, so it runs again before the session opens.";
        await runReadiness();
        const failing = failingPath();
        const report = typeof readinessReport === "undefined" ? null : readinessReport;
        if (failing || !report?.checked_utc || report.status === "blocked") {
          $("start-check").dataset.level = "not_ready";
          $("start-check-title").textContent = failing ? failing.title : "The setup check did not pass";
          $("start-check-message").textContent = failing ? `${failing.message} Fix this, then press Start session again.` : "See the check details under Check setup, fix the cause, then press Start session again.";
          return;
        }
        $("start-check").hidden = true;
      }
      const session = await api("sessions", formValues());
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
    renderTaskCards();
    const draft = restoreDraft();
    if (draft.values?.task_id) { $("task").value = draft.values.task_id; $("task").dispatchEvent(new Event("change", {bubbles:true})); renderTaskCards(); }
    $("wizard-primary").onclick = () => safe(async () => {
      if (!completion(step).ok) return;
      if (step === "task") window.SynriaRouter.go(window.SynriaRouter.routes.wizard("check"));
      else if (step === "check") window.SynriaRouter.go(window.SynriaRouter.routes.wizard("start"));
      else await startSession();
    });
    $("wizard-back").onclick = () => {
      if (step === "task") window.SynriaRouter.go(window.SynriaRouter.routes.home());
      else window.SynriaRouter.go(window.SynriaRouter.routes.wizard(STEPS[STEPS.indexOf(step) - 1]));
    };
    document.querySelectorAll("[data-wizard-step]").forEach(node => { node.onclick = () => window.SynriaRouter.go(window.SynriaRouter.routes.wizard(node.dataset.wizardStep)); });
    const toggleSetup = open => { $("setup-configure").hidden = !open; $("change-setup").setAttribute("aria-expanded", String(open)); $("change-setup").textContent = open ? "Hide setup fields" : "Change setup"; };
    $("change-setup").onclick = () => toggleSetup($("setup-configure").hidden);
    $("done-setup").onclick = () => { toggleSetup(false); renderSetupSummary(); refresh(); $("step-check-title").focus(); };
    $("new-form").addEventListener("invalid", () => toggleSetup(true), true);
    $("new-form").addEventListener("input", () => { saveDraft(); refresh(); });
    $("new-form").addEventListener("change", () => { saveDraft(); refresh(); });
    $("readiness-confirmations").addEventListener("change", () => { refresh(); safe(maybeRunCheck); });
    $("new-form").onsubmit = event => { event.preventDefault(); safe(async () => { if (completion("start").ok) await startSession(); }); };
    window.addEventListener("console-readiness", refresh);
    window.addEventListener("console-state", () => { if (!$("new").hidden) refresh(); });
    armed = true;
  }

  return {STEPS, show, refresh, completion, initialize, furthestAllowed, current: () => step, checkIsCurrent};
})();
globalThis.SynriaWizard = SynriaWizard;
