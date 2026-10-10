/** Verify automatic setup candidates without browser permissions or device access. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(process.argv[2], "utf8");
const fields = {};
const edited = new Set();
const options = values => values.map(value => ({value}));

/** Create an inert field whose change events model user-edit protection. */
function field(name, values = null) {
  return fields[name] = {name, value:"", tagName:values ? "SELECT" : "INPUT",
    options:values ? options(values) : [],
    replaceChildren() { this.options = []; }, append(option) { this.options.push(option); },
    dispatchEvent() { edited.add(name); },
  };
}

for (const name of ["fps", "operator", "name", "power_state_start"]) field(name);
field("follower_serial", [""]);
field("follower_serial_custom");
field("gripper_type", ["", "50mm", "100mm"]);
field("wrist_camera", ["", "fake-wrist"]);
const usb = field("follower_usb_id", [""]);
const nodes = {"follower-usb-choice":usb, "serial-note":{}, "follower-serial-choice":fields.follower_serial,
  "follower-serial-custom":fields.follower_serial_custom,
  "follower-serial-help":{}, "setup-facts":{}, "setup-guidance":{},
  "new-form":{elements:{...fields, namedItem:name => fields[name]}},
};
const preset = {options:options(["", "15", "30", "custom"]), value:"",
  dispatchEvent() { fields.fps.value = this.value; edited.add("fps"); },
};
const suggestions = {host:"fake-host", account:"fake-account", free_disk_bytes:1e9,
  notice:"Review candidates", recommended:{fps:15},
  known_follower_serials:["ADF-previously-recorded"],
  previous_operator_entries:{operator:"prior operator", gripper_type:"50mm", wrist_camera:"fake-wrist"},
};
let connections = ["fake-usb"];
const calls = [];
const context = vm.createContext({
  setupSuggestions:null, editedSetupFields:edited,
  $:id => nodes[id], text:(tag, textContent) => ({tag, textContent}),
  Event:class {}, document:{querySelector:selector => selector.includes('"fps"') ? preset : null},
  api:async path => { calls.push(path); return path === "setup-suggestions" ? suggestions :
    {connections, message:"USB identity is unverified"}; },
});
vm.runInContext(source.slice(source.indexOf("async function refreshSerialConnections()"),
  source.indexOf('$("refresh-serial")')), context);
vm.runInContext(source.slice(source.indexOf("function suggestSetupValue("),
  source.indexOf('$("new-form").addEventListener("input"')), context);

/** Prefill known declarations and candidates while leaving unknown safety facts untouched. */
async function verifySetup() {
  await context.loadSetupSuggestions();
  assert.equal(fields.fps.value, "15");
  assert.equal(fields.operator.value, "prior operator");
  assert.equal(fields.gripper_type.value, "50mm");
  assert.equal(fields.wrist_camera.value, "fake-wrist");
  assert.equal(usb.value, "fake-usb", "one ID is only a candidate, not a manufacturer serial");
  assert.equal(fields.follower_serial.value, "");
  assert.deepEqual(fields.follower_serial.options.map(option => option.value),
    ["", "ADF-previously-recorded", "__manual__"], "saved serials are dropdown choices");
  assert.equal(fields.follower_serial.options[1].textContent, "ADF-previously-recorded",
    "previous physical serials are selectable suggestions");
  fields.follower_serial.value = "__manual__";
  fields.follower_serial.onchange();
  assert.equal(fields.follower_serial_custom.hidden, false, "manual fallback is an explicit option");
  assert.equal(fields.power_state_start.value, "");
  assert.match(fields.name.value, /^session-/);
  assert.deepEqual(calls, ["setup-suggestions", "serial-connections"]);
  fields.fps.value = "30";
  context.applyRecommendedSetup();
  assert.equal(fields.fps.value, "30", "automatic loading preserves edits");
  context.applyRecommendedSetup(true);
  assert.equal(fields.fps.value, "15", "explicit recommendation action may replace an edit");
  connections = ["second-usb", "third-usb"];
  await context.refreshSerialConnections();
  assert.equal(usb.value, "", "disconnected or ambiguous devices are not reassigned");
  assert.equal(fields.follower_serial.value, "__manual__",
    "refreshing USB identities does not overwrite the chosen manufacturer-serial path");
  assert.equal(fields.follower_serial_custom.hidden, false);
}

verifySetup().catch(error => { console.error(error); process.exitCode = 1; });
