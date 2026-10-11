"use strict";
/** Teach something new: type a sentence, get a breakdown, edit the cards, save the tasks.
 *
 * The planner runs on the server; this module only edits the proposal in the page. Nothing
 * is saved until Save tasks is pressed, which is the only primary on the screen.
 */
const SynriaTeach = (() => {
  const KINDS = [["skill", "Skill"], ["reset", "Reset"], ["recovery", "Recovery"]];

  /** A new empty card in the schema shape. */
  function blankStep(name = "New step") {
    return {name, instruction: "", start_state: "", end_state: "", success_rule: "",
      scene_before_take: ["Gripper empty, arm at ready."], kind: "skill", objects_grasped: [],
      holds_at_start: false, holds_at_end: false, camera_can_confirm: true, suggested_takes: 20,
      optional: false, primitives: []};
  }

  /** Pure list operations on the proposal; each returns a new array. */
  function moveStep(steps, index, delta) {
    const target = index + delta;
    if (target < 0 || target >= steps.length) return steps.slice();
    const copy = steps.slice(); const [item] = copy.splice(index, 1); copy.splice(target, 0, item); return copy;
  }
  function deleteStep(steps, index) { return steps.filter((_, i) => i !== index); }
  function splitStep(steps, index) {
    const copy = steps.slice(); const step = copy[index];
    const second = {...blankStep(`${step.name} (part 2)`), start_state: step.end_state, end_state: step.end_state,
      scene_before_take: step.scene_before_take.slice(), kind: step.kind};
    copy.splice(index + 1, 0, second); return copy;
  }
  function mergeWithNext(steps, index) {
    if (index + 1 >= steps.length) return steps.slice();
    const first = steps[index], second = steps[index + 1];
    const merged = {...first, name: `${first.name} and ${second.name.charAt(0).toLowerCase()}${second.name.slice(1)}`,
      instruction: `${first.instruction.replace(/\.\s*$/, "")}, then ${second.instruction.charAt(0).toLowerCase()}${second.instruction.slice(1)}`,
      end_state: second.end_state, success_rule: second.success_rule,
      objects_grasped: [...new Set([...first.objects_grasped, ...second.objects_grasped])],
      holds_at_end: second.holds_at_end, suggested_takes: Math.max(first.suggested_takes, second.suggested_takes),
      primitives: [...first.primitives, ...second.primitives]};
    const copy = steps.slice(); copy.splice(index, 2, merged); return copy;
  }

  /** Build the payload Save tasks posts, from the goal fields and the edited steps. */
  function payload(goal, steps) {
    return {schema_version: "synria_task_breakdown_v1",
      goal: {title: goal.title, sentence: goal.sentence, rationale: goal.rationale || ""},
      sub_tasks: steps.map(step => ({...step, suggested_takes: Number(step.suggested_takes) || 1,
        scene_before_take: step.scene_before_take.map(s => s.trim()).filter(Boolean),
        objects_grasped: step.objects_grasped.map(s => s.trim()).filter(Boolean)}))};
  }

  /** Plain-words source line for the proposal. */
  function sourceLine(result) {
    if (!result) return "";
    if (result.fallback_reason) return `Suggested by the built-in planner because the external task planner command failed: ${result.fallback_reason}`;
    return result.source === "built-in planner" ? "Suggested by the built-in planner." : "Suggested by the external task planner command.";
  }

  let goal = {title: "", sentence: "", rationale: ""}, steps = [], result = null, saving = false;

  /** Read the edited cards back into the model before any operation. */
  function readCards() {
    document.querySelectorAll("#teach-steps [data-step]").forEach(card => {
      const index = Number(card.dataset.step), step = steps[index]; if (!step) return;
      for (const name of ["name", "instruction", "start_state", "end_state", "success_rule"]) step[name] = card.querySelector(`[data-field=${name}]`).value;
      step.kind = card.querySelector("[data-field=kind]").value;
      step.suggested_takes = Number(card.querySelector("[data-field=suggested_takes]").value) || 1;
      step.scene_before_take = card.querySelector("[data-field=scene]").value.split("\n");
      step.objects_grasped = card.querySelector("[data-field=objects]").value.split(",");
      step.camera_can_confirm = card.querySelector("[data-field=camera]").checked;
      step.optional = card.querySelector("[data-field=optional]").checked;
    });
  }

  /** One editable card per step. */
  function renderCards() {
    const container = $("teach-steps"); container.replaceChildren();
    steps.forEach((step, index) => {
      const card = document.createElement("article"); card.className = "card teach-card"; card.dataset.step = String(index);
      const head = document.createElement("div"); head.className = "teach-head";
      head.append(text("span", `Step ${index + 1} of ${steps.length}`));
      const kind = document.createElement("select"); kind.dataset.field = "kind"; kind.setAttribute("aria-label", "Kind");
      for (const [value, label] of KINDS) { const option = text("option", label); option.value = value; kind.append(option); }
      kind.value = step.kind; head.append(kind);
      const optional = document.createElement("label"); optional.className = "check small"; const optionalBox = document.createElement("input"); optionalBox.type = "checkbox"; optionalBox.checked = step.optional; optionalBox.dataset.field = "optional"; optional.append(optionalBox, "Optional, outside the chain"); head.append(optional);
      card.append(head);
      const field = (label, name, value, kindOf = "input") => { const wrap = document.createElement("label"); wrap.append(label); const node = document.createElement(kindOf); if (kindOf === "textarea") node.rows = 2; node.value = value; node.dataset.field = name; wrap.append(node); return wrap; };
      card.append(field("Short name", "name", step.name));
      card.append(field("Instruction shown to you and stored with every take", "instruction", step.instruction, "textarea"));
      const states = document.createElement("div"); states.className = "split";
      states.append(field("Start state", "start_state", step.start_state, "textarea"), field("End state the overhead camera can confirm", "end_state", step.end_state, "textarea"));
      card.append(states);
      card.append(field("Success rule", "success_rule", step.success_rule, "textarea"));
      card.append(field("Scene before each take, one line each", "scene", step.scene_before_take.join("\n"), "textarea"));
      const row = document.createElement("div"); row.className = "teach-row";
      row.append(field("Objects grasped, comma separated", "objects", step.objects_grasped.join(", ")));
      const takes = field("Suggested takes", "suggested_takes", String(step.suggested_takes)); takes.querySelector("input").type = "number"; takes.querySelector("input").min = "1"; row.append(takes);
      const camera = document.createElement("label"); camera.className = "check"; const cameraBox = document.createElement("input"); cameraBox.type = "checkbox"; cameraBox.checked = step.camera_can_confirm; cameraBox.dataset.field = "camera"; camera.append(cameraBox, "The overhead camera can confirm the end state"); row.append(camera);
      card.append(row);
      const timing = text("p", "Take length: not timed yet. Time it after saving, from the task's card."); timing.className = "small-print"; card.append(timing);
      const actions = document.createElement("div"); actions.className = "actions";
      actions.append(button("Move up", () => { readCards(); steps = moveStep(steps, index, -1); renderCards(); }),
        button("Move down", () => { readCards(); steps = moveStep(steps, index, 1); renderCards(); }),
        button("Split", () => { readCards(); steps = splitStep(steps, index); renderCards(); }),
        button("Merge with next", () => { readCards(); steps = mergeWithNext(steps, index); renderCards(); }),
        button("Delete", () => { readCards(); steps = deleteStep(steps, index); renderCards(); }, "danger"));
      card.append(actions);
      container.append(card);
    });
    $("teach-result").hidden = !steps.length;
    $("teach-save").disabled = !steps.length || saving;
    $("teach-add").hidden = !steps.length;
  }

  /** Show the planner's proposal, or its questions. */
  function showResult(response) {
    result = response;
    $("teach-source").textContent = sourceLine(response);
    const questions = $("teach-questions"); questions.replaceChildren(); questions.hidden = !response.questions?.length;
    for (const question of response.questions || []) {
      const wrap = document.createElement("label"); wrap.append(question.prompt);
      const input = document.createElement("input"); input.dataset.answer = question.key; input.placeholder = `for example ${question.example}`; wrap.append(input); questions.append(wrap);
    }
    if (response.questions?.length) questions.append(button("Break it down with these answers", () => safe(breakDown), "primary"));
    $("teach-notes").textContent = (response.notes || []).join(" ");
    if (response.breakdown) {
      goal = {...response.breakdown.goal};
      steps = response.breakdown.sub_tasks.map(step => ({...step}));
      $("teach-goal-title").value = goal.title;
      $("teach-rationale").textContent = goal.rationale || "";
    } else { steps = []; }
    renderWarnings(response.warnings || []);
    renderCards();
  }

  /** The chain check's warnings, in words. */
  function renderWarnings(list) {
    const node = $("teach-warnings"); node.replaceChildren(); node.hidden = !list.length;
    if (!list.length) return;
    node.append(text("h3", "Chain check"));
    const ul = document.createElement("ul");
    for (const warning of list) ul.append(text("li", warning.message));
    node.append(ul);
  }

  /** Ask the server's planner. */
  async function breakDown() {
    const sentence = $("teach-sentence").value.trim();
    if (!sentence) { $("teach-sentence").focus(); return; }
    const answers = {};
    document.querySelectorAll("#teach-questions [data-answer]").forEach(node => { if (node.value.trim()) answers[node.dataset.answer] = node.value.trim(); });
    $("teach-break").disabled = true;
    try { showResult(await api("library/plan", {sentence, answers})); }
    finally { $("teach-break").disabled = false; }
  }

  /** Save the edited breakdown as library tasks and return to Choose task. */
  async function save() {
    readCards();
    goal.title = $("teach-goal-title").value.trim() || goal.title;
    goal.sentence = $("teach-sentence").value.trim() || goal.sentence;
    saving = true; $("teach-save").disabled = true;
    try {
      const saved = await api("library/save", {breakdown: payload(goal, steps), planner_source: result?.fallback_reason ? "built-in planner" : (result?.source || "operator")});
      renderWarnings(saved.warnings || []);
      tasks = await api("tasks");
      window.SynriaWizard.reloadTasks();
      const first = saved.goal.tasks[0];
      if (first) window.SynriaWizard.prefill(first.task_id);
      info("Tasks saved", `${saved.goal.tasks.length} tasks saved under ${saved.goal.title}. They are library tasks, not in the published registry.`, window.SynriaRouter.routes.wizard("task"));
      window.SynriaRouter.go(window.SynriaRouter.routes.wizard("task"));
    } finally { saving = false; $("teach-save").disabled = !steps.length; }
  }

  /** Show the rules the planner follows. */
  async function initialize() {
    try { const library = await api("library"); const list = $("teach-rules"); list.replaceChildren(); for (const rule of library.rules) list.append(text("li", rule)); } catch (reason) { /* rules stay empty until the server answers */ }
    $("teach-break").onclick = () => safe(breakDown);
    $("teach-save").onclick = () => safe(save);
    $("teach-add").onclick = () => { readCards(); steps = [...steps, blankStep()]; renderCards(); };
    $("teach-sentence").addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); safe(breakDown); } });
  }

  return {blankStep, moveStep, deleteStep, splitStep, mergeWithNext, payload, sourceLine, initialize, showResult, breakDown, save};
})();
globalThis.SynriaTeach = SynriaTeach;
