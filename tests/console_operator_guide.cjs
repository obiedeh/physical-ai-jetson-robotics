/** Attack pure operator guidance with failed, stale, incomplete and contradictory check fixtures. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const context = vm.createContext({});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
const guide = context.SynriaGuide;
const cases = [
  ["state_rate", "No state sample received on /joint_states after waiting 10 s", "timeout"],
  ["wrist_camera", "Camera device busy", "camera"],
  ["front_sample", "Camera sample is black", "camera"],
  ["camera_mapping", "Choose two distinct cameras", "mapping"],
  ["command_publishers", "publisher exists", "publishers"],
  ["follower_sample", "ROS header arrival disagreement", "freshness"],
  ["state_rate", "requested fps exceeds measured rate", "rate"],
  ["runtime_dependencies", "rclpy unavailable", "software"],
  ["recording_metadata", "follower_serial missing", "metadata"],
  ["qualifying_configuration", "episode window unset", "timing"],
  ["limits_verification", "limits unverified", "limits"],
  ["disk_space", "insufficient disk space", "disk"],
  ["dataset_lock_and_recovery", "journal blocked", "recovery"],
  ["", "physical contract mismatch", "contract"],
  ["", "save failed; pending frames retained", "save"],
  ["", "Failed to fetch", "network"],
  ["", "launch token is invalid", "auth"],
  ["", "unexpected refusal", "general"],
];
for (const [name, message, expected] of cases) {
  const issue = guide.issueFor(name, message);
  assert.equal(issue.key, expected, `${name}: ${message}`);
  assert.ok(issue.title && issue.summary && issue.steps.length >= 3 && issue.doc);
}
const checks = {state_subscription:{passed:true, status:"configured", message:"Subscription created"}};
assert.ok(guide.connectionPaths({checks}).every(row => row.status === "not_checked"), "construction is not a sample");
checks.state_rate = {passed:false, status:"not_ready", message:"No state sample received"};
assert.equal(guide.connectionPaths({checks})[0].status, "not_ready");
assert.equal(guide.connectionPaths({checks})[2].status, "not_checked");
checks.wrist_camera = {passed:true, message:"opened"};
assert.equal(guide.connectionPaths({checks})[2].status, "not_checked", "opening is not a verified frame");
checks.wrist_sample = {passed:true, status:"ready", message:"Fresh frame"};
assert.equal(guide.connectionPaths({checks})[2].status, "ready");
assert.ok(guide.connectionPaths({checks}, false).every(row => row.status === "not_checked"), "stale and changed settings never stay green");
checks.action_sample = {passed:false, status:"not_checked", message:"Follower unavailable"};
assert.equal(guide.connectionPaths({checks})[1].status, "not_checked");
checks.front_sample = {passed:false, status:"launching", message:"Checking frame"};
assert.equal(guide.connectionPaths({checks})[3].status, "launching");
const settings = {name:"test", task_id:"die_into_cup", gripper_type:"50mm", fps:15,
  operator:"test", scene:"test", state_source:"ros2_control", wrist_camera:"fake:wrist", front_camera:"fake:front"};
const base = {settings, demo:false, report:{checks, checked_utc:"test", status:"blocked"}, softwareReady:true, current:true, confirmed:true};
assert.equal(guide.nextStep(base).help, "timeout");
assert.equal(guide.nextStep({...base, current:false}).help, undefined);
assert.equal(guide.nextStep({...base, softwareReady:false}).help, "software");
assert.equal(guide.nextStep({...base, active:{id:"pending"}}).step, "record");
assert.equal(guide.nextStep({...base, settings:{...settings, fps:0}}).step, "configure");
assert.equal(guide.nextStep({...base, settings:{...settings, fps:15.5}}).step, "configure");
assert.equal(guide.nextStep({...base, confirmed:false}).step, "check");
assert.equal(guide.nextStep({...base, report:{checks:{}, status:"checking", checked_utc:"test"}}).title, "Connection checks in progress");
assert.ok(guide.issueFor("", "Failed to fetch").steps.join(" ").includes("Do not assume the recorder stopped"));
assert.ok(guide.issueFor("", "journal blocked").steps.join(" ").includes("Do not remove the lock"));
assert.equal(settings.follower_serial, undefined, "guidance never manufactures recording identity");

let modal=false, control=false, hidden=false;const commands=[];
const keyboard=vm.createContext({
  document:{querySelector:()=>modal,activeElement:{closest:()=>control}},
  $:()=>({hidden}), lastCapture:"stopped", state:{capture:{controls:{success:true,failure:true,start:true}}},
  safe:operation=>operation(), command:name=>commands.push(name),
});
const app=fs.readFileSync(process.argv[3],"utf8");
vm.runInContext(app.slice(app.indexOf("function recordingShortcut("),app.indexOf('document.addEventListener("keydown",recordingShortcut)')),keyboard);
const event={key:"s",code:"KeyS",preventDefault(){}};
for(const flag of ["ctrlKey","metaKey","altKey","isComposing","repeat"])keyboard.recordingShortcut({...event,[flag]:true});
modal=true;keyboard.recordingShortcut(event);modal=false;
control=true;keyboard.recordingShortcut(event);control=false;
hidden=true;keyboard.recordingShortcut(event);hidden=false;
assert.equal(commands.length,0,"dialogs, focused controls, hidden recording view and browser shortcuts never issue a recording action");
keyboard.recordingShortcut(event);assert.deepEqual(commands,["success"]);
keyboard.state.capture.controls.success=false;keyboard.recordingShortcut(event);
assert.deepEqual(commands,["success"],"server-provided control availability remains authoritative");
