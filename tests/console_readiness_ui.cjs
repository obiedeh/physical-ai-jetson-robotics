/** Check readiness rendering and consent without browser permissions, sockets or real devices. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const nodes = new Map();
const confirmations = [{checked:true, dataset:{readinessConfirm:"secured"}}];
let settings = {fps:15, gripper_type:"50mm"};
const report = {status:"ready", label:"Ready for session preflight", percent:100, checked_utc:"test",
  form_settings:{fps:15, gripper_type:"50mm"}, checks:{}, passed_count:4, check_count:4,
  scope:"Snapshot only", expires_after_s:60};
const calls = [];
const context = vm.createContext({
  readinessBusy:false, readinessReport:null, state:{demo:false},
  document:{querySelectorAll() { return confirmations; }, createElement() { return {}; }},
  formValues:() => settings,
  api:async (...args) => { calls.push(args); return report; },
  $:id => {
    if (!nodes.has(id)) nodes.set(id, {dataset:{}, style:{}, textContent:"",
      children:[], append(node) { this.children.push(node); },
      setAttribute(key, value) { this[key] = value; }, replaceChildren() { this.children = []; },
      reportValidity() { return true; },
    });
    return nodes.get(id);
  },
});
const source = fs.readFileSync(process.argv[2], "utf8");
vm.runInContext(source.slice(source.indexOf("function matchingReadinessSettings("),
  source.indexOf("async function refreshCameraMapping()")), context);
context.renderReadiness(report);
assert.equal(nodes.get("readiness-meter")["aria-valuenow"], "100");
assert.equal(calls.length, 0, "rendering and reload must never open sources");
context.renderReadiness({...report, checks:{
  follower_sample:{passed:false, message:"Not run"},
  recording_metadata:{passed:false, message:"Required before recording: follower_serial"},
}});
assert.match(nodes.get("readiness-results").children[0].textContent, /^NOT CHECKED/);
assert.match(nodes.get("readiness-results").children[1].textContent, /^NEEDS ATTENTION/);
settings = {gripper_type:"50mm", fps:15};
context.renderReadiness(report);
assert.equal(nodes.get("readiness-meter")["aria-valuenow"], "100", "key order is irrelevant");
settings.fps = 30;
context.renderReadiness(report);
assert.equal(nodes.get("readiness-meter")["aria-valuenow"], "0");
settings.fps = 15;
confirmations[0].checked = false;
context.renderReadiness(report);
assert.equal(nodes.get("readiness-meter")["aria-valuenow"], "0");

/** Require deliberate confirmation before sending even an otherwise valid diagnostic request. */
async function explicitRequest() {
  await assert.rejects(context.runReadiness(), /physical safety confirmations/);
  assert.equal(calls.length, 0);
  confirmations[0].checked = true;
  await context.runReadiness();
  assert.equal(calls.length, 2);
  assert.equal(calls[0][0], "readiness");
  assert.equal(calls[0][1].settings.fps, 15);
  assert.equal(calls[0][1].confirmations.secured, true);
  assert.equal(calls[1][1], undefined, "completion polls existing results only");
  context.renderReadiness({...report, status:"expired", percent:0, label:"Check expired"});
  assert.equal(nodes.get("readiness-meter")["aria-valuenow"], "0");
  assert.equal(context.readinessBusy, false);
}

explicitRequest().catch(error => { console.error(error); process.exitCode = 1; });
