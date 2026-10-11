/** Check the Teach screen's pure list edits and payload shape without a browser or a planner. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const context = vm.createContext({document: undefined});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
const teach = context.SynriaTeach;

const a = {...teach.blankStep("Die into cup"), instruction: "Pick the die and drop it into the cup.",
  start_state: "Die in zone, gripper empty, arm at ready", end_state: "Die in cup, gripper empty, arm at ready",
  success_rule: "The die is in the cup.", objects_grasped: ["die"], primitives: ["pick", "drop into"]};
const b = {...teach.blankStep("Die back to start"), instruction: "Pick the die and place it in the zone.",
  start_state: a.end_state, end_state: a.start_state, success_rule: "The die is in the zone.", kind: "reset",
  objects_grasped: ["die"], primitives: ["pick", "place"]};

let steps = [a, b];
assert.equal(JSON.stringify(teach.moveStep(steps, 0, 1).map(s => s.name)), JSON.stringify(["Die back to start", "Die into cup"]));
assert.equal(JSON.stringify(teach.moveStep(steps, 0, -1).map(s => s.name)), JSON.stringify(["Die into cup", "Die back to start"]), "a move off the end changes nothing");
assert.equal(teach.deleteStep(steps, 1).length, 1);
const split = teach.splitStep(steps, 0);
assert.equal(split.length, 3);
assert.equal(split[1].name, "Die into cup (part 2)");
assert.equal(split[1].start_state, a.end_state, "the new part starts where the first part ends");
const merged = teach.mergeWithNext(steps, 0);
assert.equal(merged.length, 1);
assert.equal(merged[0].name, "Die into cup and die back to start");
assert.match(merged[0].instruction, /^Pick the die and drop it into the cup, then pick the die and place it in the zone\.$/);
assert.equal(merged[0].end_state, b.end_state);
assert.equal(JSON.stringify(merged[0].objects_grasped), JSON.stringify(["die"]), "one object listed once");
assert.equal(merged[0].kind, "skill", "the first step's kind is kept");
assert.equal(teach.mergeWithNext(steps, 1).length, 2, "the last step has nothing to merge with");

const payload = teach.payload({title: "Roll a die", sentence: "train to roll a dice"}, [
  {...a, suggested_takes: "12", scene_before_take: [" The cup is on its mark. ", ""], objects_grasped: ["die", " "]},
]);
assert.equal(payload.schema_version, "synria_task_breakdown_v1");
assert.equal(payload.goal.rationale, "");
assert.equal(payload.sub_tasks[0].suggested_takes, 12, "takes are numbers");
assert.equal(JSON.stringify(payload.sub_tasks[0].scene_before_take), JSON.stringify(["The cup is on its mark."]), "blank lines are dropped");
assert.equal(JSON.stringify(payload.sub_tasks[0].objects_grasped), JSON.stringify(["die"]));

assert.equal(teach.sourceLine({source: "built-in planner"}), "Suggested by the built-in planner.");
assert.equal(teach.sourceLine({source: "external task planner command"}), "Suggested by the external task planner command.");
assert.match(teach.sourceLine({source: "built-in planner", fallback_reason: "timed out"}), /^Suggested by the built-in planner because the external task planner command failed: timed out$/);
console.log("teach screen edits verified");

// Chain numbering and the strip: required steps count, optional ones are labelled.
const recovery = {...teach.blankStep("Stand the cup up"), kind: "recovery", optional: true, instruction: "Stand it up."};
assert.equal(JSON.stringify(teach.chainLabels([a, b, recovery])), JSON.stringify(["1", "2", "Optional"]));
const strip = teach.chainStrip([a, b, recovery]);
assert.equal(strip.chain, "1 Die into cup, then 2 Die back to start (reset)");
assert.equal(strip.optional, "Stand the cup up (recovery)");
assert.equal(JSON.stringify(teach.blockers([a, {...b, name: " "}, {...recovery, instruction: ""}])), JSON.stringify(["step 2 needs a name", "step 3 needs an instruction"]));
assert.equal(teach.blockers([a, b]).length, 0);
assert.equal(JSON.stringify(teach.EXAMPLES), JSON.stringify(["roll a die", "put the die in the cup", "move the cup to the tray"]));
console.log("teach chain labels verified");
