"use strict";
/** Render the Home screen: the one next move, a status strip in words, and the launch-key notice.
 *
 * Everything shown here is read from state the page already polls. Nothing on Home opens a
 * source, starts a check or changes a session; the cards only navigate.
 */
(function () {
  /** Describe one status row in words and a level, never colour alone. */
  function row(level, name, detail) { return {level, name, detail}; }

  /** Derive the status rows from the last known server state, software check and setup check. */
  function statusRows(current, software, report) {
    const rows = [];
    const active = current.active_session;
    rows.push(active ? row("ready", "Session", `${active.name} is open`) : row("neutral", "Session", "None open"));
    if (!software) rows.push(row("neutral", "Software", "Not checked yet"));
    else rows.push(row(software.status === "ready" ? "ready" : software.status === "launching" ? "checking" : "attention", "Software",
      software.status === "ready" ? "Recording libraries found" : software.status === "launching" ? "Checking" : (software.message || "Something is missing")));
    const checks = report?.checks || {};
    const stale = !report?.checked_utc || report.status === "expired";
    const when = report?.checked_utc ? `last setup check ${report.checked_utc.replace("T", " ").slice(0, 16)} UTC` : "no setup check yet";
    const describe = (name, keys) => {
      const sample = checks[keys[0]];
      if (stale || !sample || sample.status === "not_checked" || sample.message === "Not run") return row("neutral", name, `Not checked (${when})`);
      const failed = keys.find(key => checks[key]?.passed === false && !["not_checked", "launching"].includes(checks[key]?.status));
      if (failed) return row("attention", name, checks[failed].message);
      return row(sample.passed ? "ready" : "neutral", name, sample.passed ? `Fresh sample (${when})` : sample.message);
    };
    rows.push(describe("Arm feed", ["follower_sample", "state_rate"]));
    rows.push(describe("Wrist camera", ["wrist_sample", "wrist_camera"]));
    rows.push(describe("Overhead camera", ["front_sample", "front_camera"]));
    if (typeof current.free_disk_bytes === "number") {
      const free = current.free_disk_bytes, floor = current.min_free_bytes || 0;
      rows.push(row(free >= floor ? "ready" : "attention", "Storage", `${(free / 1e9).toFixed(1)} GB free${free >= floor ? "" : ", below the recording floor"}`));
    }
    return rows;
  }

  const levelWords = {ready:"OK", attention:"Needs attention", checking:"Checking", neutral:"Unknown"};

  /** Paint the strip and the session card from current page state. */
  function renderHome() {
    if (typeof state === "undefined" || !document.getElementById("status-strip")) return;
    const software = window.SynriaSoftware || null;
    const report = typeof readinessReport === "undefined" ? null : readinessReport;
    const strip = document.getElementById("status-strip");
    const rows = statusRows(state, software, report);
    const signature = JSON.stringify(rows);
    if (strip.dataset.signature !== signature) {
      strip.dataset.signature = signature; strip.replaceChildren();
      for (const item of rows) {
        const li = document.createElement("li"); li.dataset.level = item.level;
        li.append(text("b", item.name), text("span", `${levelWords[item.level]} · ${item.detail}`));
        strip.append(li);
      }
    }
    const link = document.getElementById("home-session-link"), active = state.active_session;
    const r = window.SynriaRouter.routes;
    if (active) {
      document.getElementById("home-session-title").textContent = "Continue session";
      document.getElementById("home-session-text").textContent = `${active.name} is open. Go back to recording.`;
      link.textContent = "Continue recording"; link.href = window.SynriaRouter.href(r.record(active.id)); link.dataset.route = r.record(active.id);
    } else {
      document.getElementById("home-session-title").textContent = "New session";
      document.getElementById("home-session-text").textContent = "Choose the task, check the cameras and arm feed, then record takes.";
      link.textContent = "New session"; link.href = window.SynriaRouter.href(r.wizard("task")); link.dataset.route = r.wizard("task");
    }
    const count = typeof sessions === "undefined" ? 0 : sessions.filter(s => !["trashed", "purged"].includes(s.status)).length;
    document.getElementById("home-sessions-text").textContent = count ? `${count} session${count === 1 ? "" : "s"} on this workspace.` : "No sessions yet.";
    document.getElementById("token-notice").hidden = Boolean(token);
  }

  window.SynriaHome = {statusRows, renderHome};
  if (typeof window !== "undefined" && typeof document !== "undefined") {
    for (const name of ["console-state", "console-readiness", "console-sessions", "console-route"]) window.addEventListener(name, renderHome);
  }
})();
