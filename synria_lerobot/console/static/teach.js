"use strict";
/** Teach something new: type a sentence, read the proposal as compact cards, edit if you want, save.
 *
 * The planner runs on the server; this module only edits the proposal in the page. The default
 * view is a summary a novice can accept without touching a field; Edit opens the full form on
 * one card. Nothing is saved until Save tasks, the only primary once a breakdown exists.
 */
const SynriaTeach = (() => {
  const KINDS = [["skill", "Skill"], ["reset", "Reset"], ["recovery", "Recovery"]];
  const EXAMPLES = ["roll a die", "put the die in the cup", "move the cup to the tray"];
  const UNDERSTANDS = "The built-in planner understands roll a die, put something in something, and move something to somewhere. For anything else, answer these questions or add the steps by hand.";

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

  /** Chain numbering: required steps count 1..n, optional ones are labelled Optional. */
  function chainLabels(steps) {
    let n = 0;
    return steps.map(step => step.optional ? "Optional" : String(++n));
  }

  /** One line naming the chain, for the strip at the top. */
  function chainStrip(steps) {
    const labels = chainLabels(steps);
    const chain = steps.map((step, i) => ({step, label: labels[i]})).filter(item => !item.step.optional)
      .map(item => `${item.label} ${item.step.name}${item.step.kind !== "skill" ? ` (${item.step.kind})` : ""}`);
    const optional = steps.filter(step => step.optional).map(step => `${step.name} (${step.kind})`);
    return {chain: chain.join(", then "), optional: optional.join(", ")};
  }

  /** Build the payload Save tasks posts, from the goal fields and the edited steps. */
  function payload(goal, steps) {
    return {schema_version: "synria_task_breakdown_v1",
      goal: {title: goal.title, sentence: goal.sentence, rationale: goal.rationale || ""},
      sub_tasks: steps.map(step => ({...step, suggested_takes: Number(step.suggested_takes) || 1,
        scene_before_take: step.scene_before_take.map(s => s.trim()).filter(Boolean),
        objects_grasped: step.objects_grasped.map(s => s.trim()).filter(Boolean)}))};
  }

  /** What stops a save: only an empty name or instruction; warnings never block. */
  function blockers(steps) {
    const items = [];
    steps.forEach((step, i) => {
      if (!String(step.name || "").trim()) items.push(`step ${i + 1} needs a name`);
      if (!String(step.instruction || "").trim()) items.push(`step ${i + 1} needs an instruction`);
    });
    return items;
  }

  /** Plain-words source line for the proposal. */
  function sourceLine(result) {
    if (!result) return "";
    if (result.fallback_reason) return `Suggested by the built-in planner because the external task planner command failed: ${result.fallback_reason}`;
    return result.source === "built-in planner" ? "Suggested by the built-in planner." : "Suggested by the external task planner command.";
  }

  let goal = {title: "", sentence: "", rationale: ""}, steps = [], result = null, warnings = [], saving = false, editing = new Set();

  /** Read the edited form of one card back into the model. */
  function readCard(card) {
    const index = Number(card.dataset.step), step = steps[index]; if (!step || card.dataset.mode !== "edit") return;
    for (const name of ["name", "instruction", "start_state", "end_state", "success_rule"]) step[name] = card.querySelector(`[data-field=${name}]`).value;
    step.kind = card.querySelector("[data-field=kind]").value;
    step.suggested_takes = Number(card.querySelector("[data-field=suggested_takes]").value) || 1;
    step.scene_before_take = card.querySelector("[data-field=scene]").value.split("\n");
    step.objects_grasped = card.querySelector("[data-field=objects]").value.split(",");
    step.camera_can_confirm = card.querySelector("[data-field=camera]").checked;
    step.optional = card.querySelector("[data-field=optional]").checked;
  }
  function readCards() { document.querySelectorAll("#teach-steps [data-step]").forEach(readCard); }

  /** The full form for one step, used inside Edit. */
  function editForm(step, index) {
    const form = document.createElement("div"); form.className = "teach-form";
    const head = document.createElement("div"); head.className = "teach-head";
    const kind = document.createElement("select"); kind.dataset.field = "kind"; kind.setAttribute("aria-label", "Kind");
    for (const [value, label] of KINDS) { const option = text("option", label); option.value = value; kind.append(option); }
    kind.value = step.kind; head.append(text("span", "Kind"), kind);
    const optional = document.createElement("label"); optional.className = "check small"; const optionalBox = document.createElement("input"); optionalBox.type = "checkbox"; optionalBox.checked = step.optional; optionalBox.dataset.field = "optional"; optional.append(optionalBox, "Optional, outside the chain"); head.append(optional);
    form.append(head);
    const field = (label, name, value, kindOf = "input") => { const wrap = document.createElement("label"); wrap.append(label); const node = document.createElement(kindOf); if (kindOf === "textarea") node.rows = 2; node.value = value; node.dataset.field = name; wrap.append(node); return wrap; };
    form.append(field("Short name", "name", step.name));
    form.append(field("Instruction shown to you and stored with every take", "instruction", step.instruction, "textarea"));
    const states = document.createElement("div"); states.className = "split";
    states.append(field("Starts with", "start_state", step.start_state, "textarea"), field("Ends with (the overhead camera can confirm this)", "end_state", step.end_state, "textarea"));
    form.append(states);
    form.append(field("Success rule", "success_rule", step.success_rule, "textarea"));
    form.append(field("Scene before each take, one line each", "scene", step.scene_before_take.join("\n"), "textarea"));
    const row = document.createElement("div"); row.className = "teach-row";
    row.append(field("Objects grasped, comma separated", "objects", step.objects_grasped.join(", ")));
    const takes = field("Suggested takes", "suggested_takes", String(step.suggested_takes)); takes.querySelector("input").type = "number"; takes.querySelector("input").min = "1"; row.append(takes);
    const camera = document.createElement("label"); camera.className = "check"; const cameraBox = document.createElement("input"); cameraBox.type = "checkbox"; cameraBox.checked = step.camera_can_confirm; cameraBox.dataset.field = "camera"; camera.append(cameraBox, "The overhead camera can confirm the end state"); row.append(camera);
    form.append(row);
    const actions = document.createElement("div"); actions.className = "actions";
    const act = (label, fn, kindOf = "") => button(label, () => { readCards(); fn(); renderCards(); }, kindOf);
    actions.append(act("Done", () => editing.delete(index), "primary"),
      act("Move up", () => { steps = moveStep(steps, index, -1); editing = new Set([Math.max(0, index - 1)]); }),
      act("Move down", () => { steps = moveStep(steps, index, 1); editing = new Set([Math.min(steps.length - 1, index + 1)]); }),
      act("Split", () => { steps = splitStep(steps, index); }),
      act("Merge with next", () => { steps = mergeWithNext(steps, index); }),
      act("Delete", () => { steps = deleteStep(steps, index); editing.delete(index); }, "danger"));
    form.append(actions);
    return form;
  }

  /** One card per step: compact and read-only by default, the full form when editing. */
  function renderCards() {
    const container = $("teach-steps"); container.replaceChildren();
    const labels = chainLabels(steps);
    const strip = chainStrip(steps);
    $("teach-chain").textContent = steps.length ? strip.chain : "";
    $("teach-chain-optional").textContent = strip.optional ? `Optional: ${strip.optional}` : "";
    $("teach-chain-optional").hidden = !strip.optional;
    steps.forEach((step, index) => {
      const card = document.createElement("article"); card.className = "card teach-card"; card.dataset.step = String(index);
      card.dataset.mode = editing.has(index) ? "edit" : "view";
      const head = document.createElement("div"); head.className = "teach-head";
      head.append(text("span", labels[index] === "Optional" ? "Optional" : `Step ${labels[index]}`));
      const chip = text("span", step.kind.charAt(0).toUpperCase() + step.kind.slice(1)); chip.className = "chip"; head.append(chip);
      head.append(text("small", `${step.suggested_takes} takes suggested`));
      if (card.dataset.mode === "view") head.append(button("Edit", () => { readCards(); editing.add(index); renderCards(); }));
      card.append(head);
      if (card.dataset.mode === "view") {
        card.append(text("h3", step.name || "(no name yet)"), text("p", step.instruction || "(no instruction yet)"));
        card.append(text("p", `Starts with: ${step.start_state}`), text("p", `Ends with: ${step.end_state}`));
      } else card.append(editForm(step, index));
      const own = warnings.filter(w => w.step === index + 1);
      if (own.length) { const note = document.createElement("div"); note.className = "route-note teach-warning"; for (const w of own) note.append(text("p", w.message)); card.append(note); }
      container.append(card);
    });
    const general = warnings.filter(w => w.step === null || w.step === undefined);
    const node = $("teach-warnings"); node.replaceChildren(); node.hidden = !general.length;
    for (const w of general) node.append(text("p", w.message));
    $("teach-result").hidden = !steps.length;
    renderFooter();
  }

  /** The fixed footer: Cancel, a reason, Save tasks only once a breakdown exists. */
  function renderFooter() {
    const has = steps.length > 0;
    $("teach-save").hidden = !has;
    const blocking = has ? blockers(steps) : [];
    $("teach-save").disabled = !has || saving || blocking.length > 0;
    $("teach-reason").textContent = !has ? "" : blocking.length ? `${blocking.length === 1 ? "1 thing left" : `${blocking.length} things left`}: ${blocking.join(", ")}` : (warnings.length ? "Warnings are shown on the cards. You can still save." : "Accept the steps as they are, or press Edit on a card.");
    $("teach-break").classList.toggle("primary", !has && !$("teach-questions").children.length);
    $("teach-break").textContent = has ? "Break it down again" : "Break it down";
    $("teach-add").hidden = !has;
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
    if (response.questions?.length) {
      questions.append(button("Break it down with these answers", () => safe(breakDown), "primary"));
      const byHand = button("Add steps by hand", () => { steps = [blankStep()]; editing = new Set([0]); warnings = []; goal = {title: $("teach-sentence").value.trim().slice(0, 80) || "New goal", sentence: $("teach-sentence").value.trim(), rationale: ""}; $("teach-goal-title").value = goal.title; $("teach-rationale").textContent = ""; questions.hidden = true; renderCards(); }); byHand.className = "text-button"; questions.append(byHand);
    }
    $("teach-notes").textContent = response.questions?.length ? `The planner could not work out the object and where it goes from that sentence. ${UNDERSTANDS}` : (response.notes || []).join(" ");
    if (response.breakdown) {
      goal = {...response.breakdown.goal};
      steps = response.breakdown.sub_tasks.map(step => ({...step}));
      editing = new Set();
      $("teach-goal-title").value = goal.title;
      $("teach-rationale").textContent = goal.rationale || "";
    } else { steps = []; }
    warnings = response.warnings || [];
    renderCards();
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

  /** Save the edited breakdown as library tasks and go forward to Choose task. */
  async function save() {
    readCards();
    goal.title = $("teach-goal-title").value.trim() || goal.title;
    goal.sentence = $("teach-sentence").value.trim() || goal.sentence;
    if (blockers(steps).length) { renderFooter(); return; }
    saving = true; renderFooter();
    try {
      const saved = await api("library/save", {breakdown: payload(goal, steps), planner_source: result?.fallback_reason ? "built-in planner" : (result?.source || "operator")});
      tasks = await api("tasks");
      window.SynriaWizard.reloadTasks();
      const first = saved.goal.tasks[0];
      if (first) window.SynriaWizard.prefill(first.task_id);
      const chain = steps.filter(step => !step.optional).length, optional = steps.length - chain;
      info("Tasks saved", `${chain} task${chain === 1 ? "" : "s"} saved under ${saved.goal.title}${optional ? ` (plus ${optional} optional)` : ""}.`, window.SynriaRouter.routes.wizard("task"));
      steps = []; warnings = []; result = null; editing = new Set(); renderCards();
      window.SynriaRouter.go(window.SynriaRouter.routes.wizard("task"));
    } finally { saving = false; renderFooter(); }
  }

  /** Example chips the built-in planner really handles. */
  function renderExamples() {
    const row = $("teach-examples"); row.replaceChildren();
    for (const example of EXAMPLES) row.append(button(example, () => { $("teach-sentence").value = example; $("teach-sentence").focus(); }));
  }

  /** Show the rules the planner follows and wire the controls. */
  async function initialize() {
    renderExamples();
    try { const library = await api("library"); const list = $("teach-rules"); list.replaceChildren(); for (const rule of library.rules) list.append(text("li", rule)); } catch (reason) { /* rules stay empty until the server answers */ }
    $("teach-break").onclick = () => safe(breakDown);
    $("teach-save").onclick = () => safe(save);
    $("teach-cancel").onclick = () => window.SynriaRouter.go(window.SynriaRouter.routes.wizard("task"));
    $("teach-add").onclick = () => { readCards(); steps = [...steps, blankStep()]; editing = new Set([steps.length - 1]); renderCards(); };
    $("teach-sentence").addEventListener("keydown", event => { if (event.key === "Enter") { event.preventDefault(); safe(breakDown); } });
    renderFooter();
  }

  return {blankStep, moveStep, deleteStep, splitStep, mergeWithNext, chainLabels, chainStrip, blockers, payload, sourceLine, EXAMPLES, initialize, showResult, breakDown, save};
})();
globalThis.SynriaTeach = SynriaTeach;
