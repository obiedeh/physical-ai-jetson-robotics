"use strict";
/** Describe the Record screen in plain words: one phase, one message, only the actions that apply.
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
  /** The one place the two camera roles are named for people. */
  const NAMES = {wrist: "Wrist camera", front: "Overhead camera"};
  /** Sentence-case title for a task identifier such as die_into_cup. */
  function taskTitle(taskId) { const words = String(taskId || "").replaceAll("_", " "); return words.charAt(0).toUpperCase() + words.slice(1); }
  /** The one place a task is named for people.
   *
   * A taught task carries the server's display name, already numbered once ("1. Die into cup")
   * or marked optional ("Optional: Stand the cup up"); a built-in task is titled from its id.
   * With goal set, a taught task also names its goal; with where set, its place in the chain.
   */
  function taskLabel(task, options = {}) {
    if (!task) return "unknown task";
    const taught = task.task_source === "library";
    const base = taught && task.display_name ? task.display_name : taskTitle(task.task_id);
    if (!taught) return base;
    if (options.where) return task.optional ? "optional step" : `step ${task.chain_position}`;
    return options.goal && task.goal_title ? `${base} (${task.goal_title})` : base;
  }
  const LABELS = {
    start: "Start recording (Space)", stop: "Stop (Space)", cancel_countdown: "Cancel countdown",
    success: "Yes, save as success (S)", failure: "No, save as failure (F)", retry: "Retry save",
    abort: "Aborted, throw away (A)", discard: "Throw away, bad capture", redo: "Run another practice run",
    close: "Finish practice run",
  };

  /** Say how many quality checks passed and name the failed ones in plain words. */
  function gateSummary(episode, demo = false) {
    const verdict = episode ? episode.gate_passed : undefined;
    const known = verdict === true || verdict === false || verdict === 0 || verdict === 1;
    if (!known) return "Quality checks not run yet";
    const failed = (episode.failed_gates || episode.failed_gates_json || []).map(name => GATE_WORDS[name] || name);
    const suffix = demo ? " (demo data)" : "";
    if (verdict === true || verdict === 1) return `Quality checks passed${suffix}`;
    if (!failed.length) return `Quality check failed${suffix}`;
    return `Quality check${failed.length === 1 ? "" : "s"} failed: ${failed.join(", ")}${suffix}`;
  }

  /** Takes count from 1 on screen; the stored index stays zero-based. */
  function takeNumber(index) { return (Number(index) || 0) + 1; }

  /** One sentence for the last saved take. */
  function savedSentence(episode, demo = false) {
    return `Take ${takeNumber(episode.episode_index)} saved as ${episode.operator_label}. ${gateSummary(episode, demo)}`;
  }

  /** Worded counters: saved, labels, discards and the target or practice note. */
  function counters(capture, active) {
    // Session totals come from the catalog; the capture run only knows its own discards.
    const practice = capture.recording_purpose === "disposable_smoke";
    const c = (!practice && active.counts) || capture.counters || {};
    const discarded = capture.counters?.discarded || 0;
    const target = practice ? "Practice run, not counted" : `Target ${active.target_episodes ?? "unset"}`;
    return `Saved ${c.saved || 0} (${c.success || 0} success, ${c.failure || 0} failure)`
      + (discarded ? ` · Thrown away ${discarded}` : "") + ` · ${target}`;
  }

  /** Build the phase, message, countdown, actions that apply and tooltip reasons for the rest. */
  function describe(capture, active, options = {}) {
    const c = capture || {};
    const demo = Boolean(options.demo);
    const reasons = {start: "", stop: "Not recording", success: "Nothing stopped to save",
      failure: "Nothing stopped to save", retry: "No failed save to retry",
      abort: "Nothing stopped to throw away", discard: "No pending frames", cancel_countdown: "No countdown pending"};
    const view = {phase: "", headline: "", message: "", countdownS: null, countdownUnit: "s left", actions: [], reasons,
      recording: false, betweenTakes: false,
      counters: active ? counters(c, active) : "",
      lastSave: c.last_episode ? savedSentence(c.last_episode, demo) : "",
      take: takeNumber(c.episode_index), target: active?.target_episodes ?? null};
    const act = (...commands) => { view.actions = commands.map(command => ({command, label: LABELS[command]})); };
    if (!active) {
      view.phase = "no-session"; view.headline = "No session";
      view.message = "Open a session from Home or Sessions to record.";
      Object.keys(reasons).forEach(key => { reasons[key] = "No session open"; });
      return view;
    }
    const frames = c.pending_frame_count || 0;
    const remaining = Math.max(0, c.remaining_s ?? 0);
    if (c.state === "closed") {
      view.phase = "paused"; view.headline = "Session paused";
      view.message = "Cameras are released and saved takes are kept. Continue from Home or Sessions.";
      Object.keys(reasons).forEach(key => { reasons[key] = "Session paused"; });
    } else if (c.recovery_blocked) {
      view.phase = "blocked"; view.headline = "Recording blocked";
      view.message = "The dataset needs recovery review. Pause the session and follow the recovery procedure; do not remove the lock.";
      reasons.start = "Recovery blocked";
    } else if (c.catalog_error) {
      view.phase = "blocked"; view.headline = "Catalog needs attention";
      view.message = `Catalog error: ${c.catalog_error} Resolve pending frames, then pause, reindex and open again.`;
      reasons.start = "Catalog needs reconciliation";
    } else if (c.countdown_s !== null && c.countdown_s !== undefined) {
      view.phase = "countdown"; view.headline = "Get ready";
      view.message = "Recording starts when the count reaches zero.";
      view.countdownS = c.countdown_s; view.countdownUnit = "";
      reasons.start = "Countdown pending"; reasons.cancel_countdown = "";
      act("cancel_countdown");
    } else if (c.state === "recording") {
      view.phase = "recording"; view.headline = "Recording"; view.recording = true;
      view.message = "Press Stop (Space) when the move is done.";
      view.countdownS = remaining;
      reasons.start = "Already recording"; reasons.stop = ""; reasons.discard = "Stop first";
      act("stop");
    } else if (c.saving) {
      view.phase = "saving"; view.headline = "Saving";
      view.message = `Saving take ${view.take}…`;
      reasons.start = "Saving"; reasons.success = reasons.failure = "Saving";
    } else if (c.state === "stopped") {
      const capped = c.stop_reason === "hard cap" ? `The ${c.max_episode_s ?? ""} s limit ended this take. ` : "";
      reasons.discard = ""; reasons.abort = ""; reasons.start = "Save or throw away the stopped take first";
      if (c.error) {
        view.phase = "save-failed"; view.headline = "Save failed";
        view.message = `${c.error} The frames are kept. Press Retry save; throw away only a bad capture.`;
        reasons.retry = ""; act("retry", "discard");
      } else if (!frames) {
        view.phase = "empty"; view.headline = "Nothing captured";
        view.message = `${capped}Stopped with no frames. Throw it away and start again.`;
        act("discard"); view.actions[0].label = "Throw away empty capture";
      } else {
        view.phase = "did-it-work"; view.headline = "Did it work?";
        view.message = `${capped}A real failure of the task is saved as failure. Aborted means you never attempted it; nothing is saved. Throw away only a bad capture.`;
        reasons.success = reasons.failure = ""; act("success", "failure", "abort", "discard");
      }
    } else if (c.recording_purpose === "disposable_smoke" && c.smoke_complete) {
      view.phase = "practice-finished"; view.headline = "Practice run finished";
      view.message = "The take is not kept. Run another with the same setup, or finish to start a real session.";
      reasons.start = "Practice run finished"; act("redo", "close");
    } else if (options.showSaved && c.last_episode) {
      view.phase = "saved"; view.headline = "Saved"; view.betweenTakes = true;
      view.message = savedSentence(c.last_episode, demo);
      act("start");
    } else {
      view.phase = "ready"; view.headline = "Ready"; view.betweenTakes = true;
      view.message = "Set the scene, then press Start recording.";
      act("start");
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

  window.SynriaRecord = {GATE_WORDS, GATE_COUNT, LABELS, NAMES, taskTitle, taskLabel, gateSummary, savedSentence, takeNumber, counters, describe, cameraBrief};
})();
