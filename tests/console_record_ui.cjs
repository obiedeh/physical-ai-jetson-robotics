/** Check the Record screen's plain-language states and visible actions without a browser or any device. */
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
const commands = view => view.actions.map(action => action.command).join(",");
const labels = view => view.actions.map(action => action.label).join("|");

let view = record.describe({state: "closed"}, null);
assert.equal(view.phase, "no-session");
assert.match(view.message, /Open a session from Home or Sessions/);
assert.equal(view.counters, "");
assert.equal(commands(view), "", "no session shows no actions");
assert.equal(view.reasons.start, "No session open");

view = record.describe(base, active);
assert.equal(view.phase, "ready");
assert.equal(view.headline, "Ready");
assert.equal(view.message, "Set the scene, then press Start recording.");
assert.equal(view.take, 2, "takes count from 1 on screen");
assert.equal(view.target, 10);
assert.equal(view.betweenTakes, true, "the scene reset list shows between takes");
assert.equal(commands(view), "start", "ready shows one large Start recording");
assert.equal(labels(view), "Start recording (Space)");
assert.equal(view.counters, "Saved 1 (1 success, 0 failure) · Thrown away 1 · Target 10");
assert.equal(view.countdownS, null);
assert.equal(view.reasons.success, "Nothing stopped to save");

view = record.describe({...base, counters: {saved: 1, success: 1, failure: 0, discarded: 0}}, {...active, counts: {saved: 4, success: 3, failure: 1}});
assert.equal(view.counters, "Saved 4 (3 success, 1 failure) · Target 10", "session totals show after a continue, not the run's own count");

view = record.describe({...base, countdown_s: 3.2, controls: {cancel_countdown: true}}, active);
assert.equal(view.phase, "countdown");
assert.equal(view.headline, "Get ready");
assert.equal(view.countdownS, 3.2);
assert.equal(view.countdownUnit, "", "the count stands alone during the countdown");
assert.equal(commands(view), "cancel_countdown", "Cancel countdown is the only action during a countdown");
assert.equal(view.reasons.cancel_countdown, "");

view = record.describe({...base, state: "recording", elapsed_s: 8, remaining_s: 12, pending_frame_count: 120,
  controls: {stop: true}}, active);
assert.equal(view.phase, "recording");
assert.equal(view.recording, true);
assert.equal(view.betweenTakes, false);
assert.equal(view.message, "Press Stop (Space) when the move is done.");
assert.equal(view.countdownS, 12);
assert.equal(commands(view), "stop", "recording shows one large Stop");
assert.equal(view.reasons.start, "Already recording");

view = record.describe({...base, state: "stopped", elapsed_s: 14.2, pending_frame_count: 213,
  stop_reason: "operator", controls: {success: true, failure: true, discard: true}}, active);
assert.equal(view.phase, "did-it-work");
assert.equal(view.headline, "Did it work?");
assert.equal(view.message, "A real failure of the task is saved as failure. Throw away only a bad capture.");
assert.equal(commands(view), "success,failure,discard");
assert.equal(labels(view), "Yes, save as success (S)|No, save as failure (F)|Throw away, bad capture");
assert.equal(view.reasons.success, "");
assert.equal(view.reasons.start, "Save or throw away the stopped take first");

view = record.describe({...base, state: "stopped", elapsed_s: 20, pending_frame_count: 300,
  stop_reason: "hard cap", controls: {success: true, failure: true, discard: true}}, active);
assert.match(view.message, /^The 20 s limit ended this take\. A real failure/);

view = record.describe({...base, state: "stopped", pending_frame_count: 0, controls: {discard: true}}, active);
assert.equal(view.phase, "empty");
assert.match(view.message, /Stopped with no frames\. Throw it away and start again\./);
assert.equal(labels(view), "Throw away empty capture");
assert.equal(view.reasons.success, "Nothing stopped to save");

view = record.describe({...base, state: "stopped", pending_frame_count: 300, error: "save failed: disk full.",
  controls: {retry: true, discard: true}}, active);
assert.equal(view.phase, "save-failed");
assert.equal(view.message, "save failed: disk full. The frames are kept. Press Retry save; throw away only a bad capture.");
assert.equal(commands(view), "retry,discard", "Retry save appears only after a failed save");
assert.equal(view.reasons.retry, "");

view = record.describe({...base, state: "stopped", pending_frame_count: 300, saving: true, controls: {}}, active);
assert.equal(view.phase, "saving");
assert.equal(view.message, "Saving take 2…");
assert.equal(commands(view), "");

const saved = {episode_index: 1, operator_label: "success", gate_passed: false, failed_gates: ["state_action_limits"]};
view = record.describe({...base, episode_index: 2, last_episode: saved}, active, {showSaved: true});
assert.equal(view.phase, "saved");
assert.equal(view.message, "Take 2 saved as success. Quality check failed: joint limits");
assert.equal(commands(view), "start", "a saved take goes straight back to Start recording");
view = record.describe({...base, episode_index: 2, last_episode: saved}, active, {showSaved: false});
assert.equal(view.phase, "ready", "after the saved message the banner returns to Ready");
assert.equal(view.lastSave, "Take 2 saved as success. Quality check failed: joint limits");
assert.equal(record.gateSummary({gate_passed: true, failed_gates: []}), "Quality checks passed");
assert.equal(record.gateSummary({gate_passed: true, failed_gates: []}, true), "Quality checks passed (demo data)");
assert.equal(record.gateSummary({gate_passed: false, failed_gates: ["timestamp_skew", "state_action_limits"]}),
  "Quality checks failed: timestamp alignment, joint limits");
assert.equal(record.gateSummary({gate_passed: 1, failed_gates_json: []}), "Quality checks passed", "a stored 1 is a pass");
assert.equal(record.gateSummary({gate_passed: 0, failed_gates_json: ["visual_sanity"]}), "Quality check failed: image brightness", "a stored 0 is a fail");
assert.equal(record.gateSummary({operator_label: "success"}), "Quality checks not run yet");
assert.equal(record.savedSentence({episode_index: 0, operator_label: "failure"}), "Take 1 saved as failure. Quality checks not run yet");
assert.equal(record.taskTitle("die_into_cup"), "Die into cup");
assert.equal(record.NAMES.front, "Overhead camera");

view = record.describe({...base, recording_purpose: "disposable_smoke", smoke_complete: true, last_episode: saved,
  counters: {saved: 1, success: 1, failure: 0, discarded: 0}, controls: {}}, active);
assert.equal(view.phase, "practice-finished");
assert.equal(view.message, "The take is not kept. Finish to run another or to start a real session.");
assert.equal(labels(view), "Finish practice run");
assert.equal(view.counters, "Saved 1 (1 success, 0 failure) · Practice run, not counted");
assert.equal(view.reasons.start, "Practice run finished");

view = record.describe({...base, recovery_blocked: true, controls: {}}, active);
assert.equal(view.phase, "blocked");
assert.match(view.message, /do not remove the lock/);
assert.equal(commands(view), "");

view = record.describe({...base, catalog_error: "catalog write failed.", controls: {}}, active);
assert.equal(view.phase, "blocked");
assert.equal(view.headline, "Catalog needs attention");

view = record.describe({...base, state: "closed", controls: {}}, active);
assert.equal(view.phase, "paused");
assert.match(view.message, /Cameras are released and saved takes are kept/);

assert.equal(record.takeNumber(0), 1);
assert.equal(record.cameraBrief({}), "no sample yet");
assert.equal(record.cameraBrief({error: "device busy"}), "no frames: device busy");
assert.equal(record.cameraBrief({age_s: 0.034, native_resolution: [640, 480]}), "0.03 s old · 640×480");
console.log("record banner states verified");
