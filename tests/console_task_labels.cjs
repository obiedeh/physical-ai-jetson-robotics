/** Feed the real server task payload to the page's label helper and check every label reads cleanly. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const [bannerPath, payloadPath] = process.argv.slice(2);
const context = vm.createContext({window: {}});
vm.runInContext(fs.readFileSync(bannerPath, "utf8"), context);
const record = context.window.SynriaRecord;
const tasks = JSON.parse(fs.readFileSync(payloadPath, "utf8"));
assert.ok(tasks.length >= 4, "the payload holds built-in and taught tasks");

/** A label must never show an id, an undefined, a doubled number or a leading comma. */
function clean(label, where) {
  assert.equal(typeof label, "string", where);
  assert.ok(!/undefined|null/.test(label), `${where}: ${label}`);
  assert.ok(!/^\d+\. \d+\./.test(label), `${where} numbers once: ${label}`);
  assert.ok(!/^[,\s]/.test(label), `${where} no leading comma: ${label}`);
  assert.ok(!/^lib_|_\d+$/.test(label), `${where} no id: ${label}`);
}
for (const task of tasks) {
  const plain = record.taskLabel(task), withGoal = record.taskLabel(task, {goal: true});
  clean(plain, `${task.task_id} plain`); clean(withGoal, `${task.task_id} with goal`);
  clean(`${plain}, 50 mm gripper`, `${task.task_id} setup line`);
  if (task.task_source === "library") {
    assert.equal(plain, task.display_name, "a taught task shows the server's display name");
    assert.ok(withGoal.endsWith(`(${task.goal_title})`), withGoal);
    const where = record.taskLabel(task, {where: true});
    if (task.optional) { assert.match(plain, /^Optional: \S/); assert.equal(where, "optional step"); }
    else { assert.match(plain, /^\d+\. \S/); assert.equal(where, `step ${task.chain_position}`); }
  } else {
    assert.equal(plain, withGoal, "a built-in task has no goal suffix");
    assert.match(plain, /^[A-Z][a-z ]+$/);
  }
}
const taught = tasks.filter(task => task.task_source === "library");
assert.ok(taught.some(task => task.optional), "the seed includes an optional step");
console.log(`task labels verified for ${tasks.length} tasks`);
