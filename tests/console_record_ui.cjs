/** Check the Record screen's plain-language banner states without a browser or any device. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const context = vm.createContext({window: {}});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), context);
const record = context.window.SynriaRecord;
const active = {id: "s1", name: "Session", target_episodes: 10};
const base = {state: "idle", episode_index: 1, elapsed_s: 0, pending_frame_count: 0,
  min_episode_s: 20, max_episode_s: 20, remaining_s: 20, countdown_s: null, saving: false,
  error: null, catalog_error: null, recovery_blocked: false, recording_purpose: "qualifying",
  counters: {saved: 1, success: 1, failure: 0, discarded: 1, save_failed: 0},
  controls: {start: true}, last_episode: null, stop_reason: null, smoke_complete: false};

let view = record.describe({state: "closed"}, null);
assert.equal(view.phase, "No session");
assert.match(view.message, /Open a named session/);
assert.equal(view.counters, "");
assert.equal(view.reasons.start, "No session open");

view = record.describe(base, active);
assert.equal(view.phase, "Ready");
assert.equal(view.message, "Ready. Press Start recording (Space) to record.");
assert.equal(view.counters, "Saved 1 (1 success, 0 failure) · Discarded 1 · Target 10");
assert.equal(view.countdownS, null);
assert.equal(view.action, null);
assert.equal(view.reasons.start, "");
assert.equal(view.reasons.success, "Nothing stopped to save");

view = record.describe({...base, state: "recording", elapsed_s: 8, remaining_s: 12, pending_frame_count: 120,
  controls: {stop: true}}, active);
assert.equal(view.phase, "RECORDING");
assert.equal(view.message, "12 s left. Press Stop (Space) when the move is done.");
assert.equal(view.countdownS, 12);
assert.equal(view.reasons.start, "Already recording");

view = record.describe({...base, state: "stopped", elapsed_s: 14.2, pending_frame_count: 213,
  stop_reason: "operator", controls: {success: true, failure: true, discard: true}}, active);
assert.equal(view.phase, "Stopped");
assert.equal(view.message, "Stopped after 14.2 s with 213 frames. Save it: Save as success (S) or Save as failure (F). Discard only for a faulty capture.");
assert.equal(view.reasons.success, "");
assert.equal(view.reasons.start, "Save or discard the stopped episode first");

view = record.describe({...base, state: "stopped", elapsed_s: 20, pending_frame_count: 300,
  stop_reason: "hard cap", controls: {success: true, failure: true, discard: true}}, active);
assert.match(view.message, /^The 20 s hard cap ended this capture\. Stopped after 20\.0 s with 300 frames\. Save it/);

view = record.describe({...base, state: "stopped", pending_frame_count: 0, controls: {discard: true}}, active);
assert.match(view.message, /no frames captured\. Discard this empty capture/);
assert.equal(view.reasons.success, "Nothing stopped to save");

view = record.describe({...base, state: "stopped", pending_frame_count: 300, error: "save failed: disk full.",
  controls: {retry: true, discard: true}}, active);
assert.equal(view.message, "Save failed: save failed: disk full. Pending frames are kept. Press Retry save, or Discard only for a faulty capture.");
assert.equal(view.reasons.retry, "");

view = record.describe({...base, state: "stopped", pending_frame_count: 300, saving: true, controls: {}}, active);
assert.equal(view.phase, "Saving");
assert.equal(view.message, "Saving episode 1…");

view = record.describe({...base, countdown_s: 3.2, controls: {cancel_countdown: true}}, active);
assert.equal(view.phase, "Starting");
assert.equal(view.message, "Starting in 3.2 s. Press Cancel countdown to abort.");
assert.equal(view.countdownS, 3.2);
assert.equal(view.reasons.cancel_countdown, "");

const saved = {episode_index: 1, operator_label: "success", gate_passed: false, failed_gates: ["state_action_limits"]};
view = record.describe({...base, episode_index: 2, last_episode: saved}, active);
assert.equal(view.phase, "Ready");
assert.equal(view.message, "Saved episode 1 as success. Quality checks: 8 of 9 passed, failed: joint limits. Ready. Press Start recording (Space) to record.");
assert.equal(view.lastSave, "Saved episode 1 as success. Quality checks: 8 of 9 passed, failed: joint limits.");
assert.equal(record.gateSummary({gate_passed: true, failed_gates: []}), "Quality checks: 9 of 9 passed");
assert.equal(record.gateSummary({gate_passed: false, failed_gates: ["timestamp_skew", "state_action_limits"]}),
  "Quality checks: 7 of 9 passed, failed: timestamp alignment, joint limits");
assert.equal(record.gateSummary({operator_label: "success"}), "Quality checks: not evaluated");

view = record.describe({...base, recording_purpose: "disposable_smoke", smoke_complete: true, last_episode: saved,
  counters: {saved: 1, success: 1, failure: 0, discarded: 0}, controls: {}}, active);
assert.equal(view.phase, "Practice run finished");
assert.equal(view.message, "Practice run finished. Close sources to run another or to start a real session.");
assert.equal(JSON.stringify(view.action), JSON.stringify({label: "Close sources", command: "close"}));
assert.equal(view.counters, "Saved 1 (1 success, 0 failure) · Discarded 0 · Practice run, not counted");
assert.equal(view.reasons.start, "Practice run finished; close sources");

view = record.describe({...base, recovery_blocked: true, controls: {}}, active);
assert.equal(view.phase, "Recording blocked");
assert.match(view.message, /do not remove the lock/);

view = record.describe({...base, catalog_error: "catalog write failed.", controls: {}}, active);
assert.equal(view.phase, "Catalog needs attention");

view = record.describe({...base, state: "closed", controls: {}}, active);
assert.equal(view.phase, "Sources closed");

assert.equal(record.cameraBrief({}), "no sample yet");
assert.equal(record.cameraBrief({error: "device busy"}), "no frames: device busy");
assert.equal(record.cameraBrief({age_s: 0.034, native_resolution: [640, 480]}), "0.03 s old · 640×480");
console.log("record banner states verified");
