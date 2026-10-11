"use strict";
/** Describe the Record screen in plain words: one phase, one next action, worded counters.
 *
 * This module only reads the server's capture snapshot. It never changes a recorder
 * transition, guard or keyboard shortcut; the server's controls remain authoritative.
 */
(function () {
  const GATE_WORDS = {
    frame_count: "frame count", finite_state_action: "finite values",
    cameras_present: "both cameras present", state_action_limits: "joint limits",
    timestamp_skew: "timestamp alignment", source_staleness: "source freshness",
    episode_length: "episode length", gripper_dimensionality: "state size",
    visual_sanity: "image brightness",
  };
  const GATE_COUNT = Object.keys(GATE_WORDS).length;

  /** Say how many quality checks passed and name the failed ones in plain words. */
  function gateSummary(episode) {
    if (!episode || typeof episode.gate_passed !== "boolean") return "Quality checks: not evaluated";
    const failed = (episode.failed_gates || []).map(name => GATE_WORDS[name] || name);
    const passed = GATE_COUNT - failed.length;
    if (!failed.length) return `Quality checks: ${GATE_COUNT} of ${GATE_COUNT} passed`;
    return `Quality checks: ${passed} of ${GATE_COUNT} passed, failed: ${failed.join(", ")}`;
  }

  /** One sentence for the last saved episode. */
  function savedSentence(episode) {
    return `Saved episode ${episode.episode_index} as ${episode.operator_label}. ${gateSummary(episode)}.`;
  }

  /** Worded counters: saved, labels, discards and the target or practice note. */
  function counters(capture, active) {
    const c = capture.counters || active.counts || {};
    const target = capture.recording_purpose === "disposable_smoke"
      ? "Practice run, not counted" : `Target ${active.target_episodes ?? "unset"}`;
    return `Saved ${c.saved || 0} (${c.success || 0} success, ${c.failure || 0} failure)`
      + ` · Discarded ${c.discarded || 0} · ${target}`;
  }

  /** Build the banner phase, message, countdown, optional action and tooltip reasons. */
  function describe(capture, active) {
    const c = capture || {};
    const reasons = {start: "", stop: "Not recording", success: "Nothing stopped to save",
      failure: "Nothing stopped to save", retry: "No failed save to retry",
      discard: "No pending frames", cancel_countdown: "No countdown pending"};
    const view = {phase: "", message: "", countdownS: null, action: null, reasons,
      counters: active ? counters(c, active) : "",
      lastSave: c.last_episode ? savedSentence(c.last_episode) : ""};
    if (!active) {
      view.phase = "No session";
      view.message = "Open a named session, or run a disposable smoke, to begin.";
      Object.keys(reasons).forEach(key => { reasons[key] = "No session open"; });
      return view;
    }
    const elapsed = (c.elapsed_s || 0).toFixed(1);
    const frames = c.pending_frame_count || 0;
    const remaining = Math.max(0, c.remaining_s ?? 0);
    if (c.state === "closed") {
      view.phase = "Sources closed";
      view.message = "Sources are closed. Open a session again to record.";
      Object.keys(reasons).forEach(key => { reasons[key] = "Sources closed"; });
    } else if (c.recovery_blocked) {
      view.phase = "Recording blocked";
      view.message = "The dataset needs recovery review. Close sources and follow the recovery procedure; do not remove the lock.";
      reasons.start = "Recovery blocked";
    } else if (c.catalog_error) {
      view.phase = "Catalog needs attention";
      view.message = `Catalog error: ${c.catalog_error} Resolve pending frames, then close, reindex and reopen.`;
      reasons.start = "Catalog needs reconciliation";
    } else if (c.countdown_s !== null && c.countdown_s !== undefined) {
      view.phase = "Starting";
      view.message = `Starting in ${c.countdown_s.toFixed(1)} s. Press Cancel countdown to abort.`;
      view.countdownS = c.countdown_s;
      reasons.start = "Countdown pending";
      reasons.cancel_countdown = "";
    } else if (c.state === "recording") {
      view.phase = "RECORDING";
      view.message = `${remaining.toFixed(0)} s left. Press Stop (Space) when the move is done.`;
      view.countdownS = remaining;
      reasons.start = "Already recording";
      reasons.stop = "";
      reasons.discard = "Stop first";
    } else if (c.saving) {
      view.phase = "Saving";
      view.message = `Saving episode ${c.episode_index ?? ""}…`;
      reasons.start = "Saving";
      reasons.success = reasons.failure = "Saving";
    } else if (c.state === "stopped") {
      view.phase = "Stopped";
      const capped = c.stop_reason === "hard cap" ? `The ${c.max_episode_s ?? ""} s hard cap ended this capture. ` : "";
      if (c.error) {
        view.message = `Save failed: ${c.error} Pending frames are kept. Press Retry save, or Discard only for a faulty capture.`;
        reasons.retry = "";
      } else if (!frames) {
        view.message = `${capped}Stopped with no frames captured. Discard this empty capture, then press Start recording (Space).`;
      } else {
        view.message = `${capped}Stopped after ${elapsed} s with ${frames} frames. Save it: Save as success (S) or Save as failure (F). Discard only for a faulty capture.`;
        reasons.success = reasons.failure = "";
      }
      reasons.discard = "";
      reasons.start = "Save or discard the stopped episode first";
    } else if (c.recording_purpose === "disposable_smoke" && c.smoke_complete) {
      view.phase = "Practice run finished";
      view.message = "Practice run finished. Close sources to run another or to start a real session.";
      view.action = {label: "Close sources", command: "close"};
      reasons.start = "Practice run finished; close sources";
    } else {
      view.phase = "Ready";
      view.message = (c.last_episode ? savedSentence(c.last_episode) + " " : "")
        + "Ready. Press Start recording (Space) to record.";
    }
    return view;
  }

  /** Short caption for one camera: age and native size, or the error. */
  function cameraBrief(info) {
    if (!info || !Object.keys(info).length) return "no sample yet";
    if (info.error) return `no frames: ${info.error}`;
    const age = typeof info.age_s === "number" ? `${info.age_s.toFixed(2)} s old` : "live";
    const size = Array.isArray(info.native_resolution) ? ` · ${info.native_resolution.join("×")}` : "";
    return age + size;
  }

  window.SynriaRecord = {GATE_WORDS, GATE_COUNT, gateSummary, savedSentence, counters, describe, cameraBrief};
})();
