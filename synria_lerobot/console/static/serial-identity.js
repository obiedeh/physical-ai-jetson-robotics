"use strict";
/** List USB connection names without opening a port or substituting them for arm identity. */
let setupSuggestions = null;
const editedSetupFields = new Set();

/** Preserve an explicit choice while refreshing; never select the only device automatically. */
async function refreshSerialConnections() {
  const inventory = await api("serial-connections");
  const select = $("follower-usb-choice"), previous = select.value;
  select.replaceChildren();
  const unknown = text("option", "Unknown / unverified · no connection selected");
  unknown.value = "";
  select.append(unknown);
  for (const path of inventory.connections) {
    const option = text("option", path);
    option.value = path;
    select.append(option);
  }
  select.value = inventory.connections.includes(previous) ? previous : "";
  $("serial-note").textContent = inventory.message;
  if (select.value !== previous) select.dispatchEvent(new Event("change", {bubbles:true}));
}

$("refresh-serial").onclick = () => safe(refreshSerialConnections);

/** Set a candidate through the existing preset control, keeping its submitted scalar in sync. */
function suggestSetupValue(name, value, force = false) {
  const field = $("new-form").elements.namedItem(name);
  if (!field || (!force && editedSetupFields.has(name))) return;
  if (field.tagName === "SELECT" && ![...field.options].some(option => option.value === String(value))) return;
  const preset = document.querySelector(`[data-preset-for="${name}"]`);
  if (preset) {
    preset.value = [...preset.options].some(option => option.value === String(value)) ? String(value) : "custom";
    preset.dispatchEvent(new Event("change", {bubbles:true}));
  }
  field.value = String(value);
  field.dispatchEvent(new Event("change", {bubbles:true}));
}

/** Apply documented recording candidates without inventing physical identity or observed state. */
function applyRecommendedSetup(force = false) {
  if (!setupSuggestions) return;
  for (const [name, value] of Object.entries(setupSuggestions.recommended)) suggestSetupValue(name, value, force);
}

/** Populate only available facts and explicit prior entries after the normal form has loaded. */
async function loadSetupSuggestions() {
  setupSuggestions = await api("setup-suggestions");
  await refreshSerialConnections();
  const prior = setupSuggestions.previous_operator_entries;
  $("setup-facts").textContent = `Host: ${setupSuggestions.host} · account: ${setupSuggestions.account} · free storage: ${(setupSuggestions.free_disk_bytes / 1e9).toFixed(1)} GB`;
  $("setup-guidance").textContent = setupSuggestions.notice + " Restored fields: " + (Object.keys(prior).join(", ") || "none") + ".";
  // Prior declarations take precedence over general candidates and remain visibly unverified today.
  for (const [name, value] of Object.entries(prior)) suggestSetupValue(name, value);
  applyRecommendedSetup();
  const usb = $("follower-usb-choice");
  if (usb.options.length === 2) suggestSetupValue("follower_usb_id", usb.options[1].value);
  if (!$("new-form").elements.operator.value) suggestSetupValue("operator", setupSuggestions.account);
  if (!$("new-form").elements.name.value) suggestSetupValue("name", `session-${new Date().toISOString().replace(/[:.]/g, "-")}`);
}

$("new-form").addEventListener("input", event => {
  const name = event.target.name || event.target.dataset.presetFor;
  if (name) editedSetupFields.add(name);
});
$("new-form").addEventListener("change", event => {
  const name = event.target.name || event.target.dataset.presetFor;
  if (name) editedSetupFields.add(name);
});
$("use-recommended").onclick = () => safe(async () => { applyRecommendedSetup(true); });
window.addEventListener("console-ready", () => safe(loadSetupSuggestions), {once:true});
if (window.synriaConsoleReady) safe(loadSetupSuggestions);
