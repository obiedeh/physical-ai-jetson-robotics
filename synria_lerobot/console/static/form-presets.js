"use strict";
/** Present documented presets without replacing the recorder's named fields or custom validation. */

/** Enhance one suggested-value input with a dropdown while retaining its original form payload. */
function addPresetSelect(input) {
  const options = document.getElementById(input.getAttribute("list"));
  const select = document.createElement("select");
  const inputType = input.type;
  select.required = input.required;
  select.setAttribute("aria-label", `${input.getAttribute("aria-label")} preset`);
  select.dataset.presetFor = input.name;
  const choices = [["", "Choose a value"],
    ...Array.from(options.options, option => [option.value, option.label || option.textContent]),
    ["custom", "Custom value…"]];
  for (const [value, label] of choices) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.append(option);
  }
  select.value = choices.some(([value]) => value === input.value) ? input.value : "custom";
  input.before(select);

  /** Preserve custom values and native constraints; presets only change the submitted scalar. */
  function applySelection() {
    const custom = select.value === "custom";
    input.type = custom ? inputType : "hidden";
    if (!custom) input.value = select.value;
  }

  select.addEventListener("change", () => {
    applySelection();
    if (select.value === "custom") input.focus();
  });
  applySelection();
}

document.querySelectorAll("input[data-presets]").forEach(addPresetSelect);
