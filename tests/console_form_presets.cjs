/** Exercise preset selection and original form serialization without a browser or hardware. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const fields = new Map();
const suggestions = {options: [{value:"15", label:"15 FPS"}, {value:"30", label:"30 FPS"}]};

/** Model only browser properties used by the enhancement, retaining native validation attributes. */
function input(name, value, type = "number") {
  const node = {name, value, type, required:true, min:"1", step:"1", dataset:{},
    getAttribute(key) { return key === "list" ? "presets" : name; },
    before(select) { this.select = select; },
    focus() { this.focused = true; },
  };
  fields.set(name, node);
  return node;
}

const fps = input("fps", "");
const custom = input("custom", "12");
const context = vm.createContext({document: {
  querySelectorAll() { return [fps, custom]; },
  getElementById() { return suggestions; },
  createElement() { return {dataset:{}, options:[], setAttribute() {},
    append(option) { this.options.push(option); },
    addEventListener(name, callback) { this[name] = callback; },
  }; },
}});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
assert.equal(fps.select.required, true);
assert.equal(fps.select.value, "", "required FPS remains explicitly unselected");
assert.equal(fps.select.name, undefined, "preset controls must not add payload keys");
assert.equal(fps.type, "hidden");
assert.equal(custom.select.value, "custom", "non-preset existing values remain unchanged");
assert.equal(custom.type, "number");
assert.equal(custom.value, "12");
assert.deepEqual(fps.select.options.map(option => option.value), ["", "15", "30", "custom"]);
fps.select.value = "30";
fps.select.change();
assert.equal(fps.value, "30");
fps.select.value = "custom";
fps.select.change();
assert.equal(fps.type, "number");
assert.equal(fps.focused, true);
assert.equal(fps.required, true);
assert.equal(fps.min, "1");
assert.equal(fps.step, "1");
fps.value = "12";

const values = {fps:fps.value, action_lookahead_steps:"1", image_width:"224", image_height:"224",
  state_startup_timeout_s:"10", target_episodes:"100", guard_command_topic:" /extra ",
  action_source:"next_state", follower_topic:"/joint_states"};
context.$ = () => ({elements:{state_has_velocity:{checked:false}}});
context.FormData = class {
  /** Yield the same named scalar pairs as an HTML form with a hidden preset-backed input. */
  [Symbol.iterator]() { return Object.entries(values)[Symbol.iterator](); }
};
const app = fs.readFileSync(process.argv[3], "utf8");
vm.runInContext(app.slice(app.indexOf("function formValues()"), app.indexOf("function taskDescription()")), context);
const result = context.formValues();
assert.equal(result.fps, 12);
assert.equal(result.image_width, 224);
assert.equal(result.action_lookahead_steps, 1);
assert.equal(result.action_source, "next_state");
assert.equal(result.state_has_velocity, false);
assert.deepEqual(Array.from(result.guard_command_topic), ["/extra"]);
fps.select.value = "15";
fps.select.change();
assert.equal(fps.value, "15");
assert.equal(fps.type, "hidden");
