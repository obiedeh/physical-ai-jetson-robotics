"use strict";
/** Render Home: three things to do, a first-run strip, and a status strip in neutral words.
 *
 * Everything shown here is read from state the page already polls. The cards only navigate
 * or reopen a paused session through the existing API; Check now runs the same read-only
 * setup check the wizard runs, and only where its bench checks are not required.
 */
(function () {
  const FIRST_RUN_KEY = "synria-first-run-dismissed";
  /** Describe one status row in words and a level, never colour alone. */
  function row(level, name, detail) { return {level, name, detail}; }
  /** Relative time in words for a UTC timestamp. */
  function ago(iso) {
    const seconds = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 1000));
    if (!Number.isFinite(seconds)) return "earlier";
    if (seconds < 45) return "just now";
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 48) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
    return `on ${new Date(iso).toLocaleDateString()}`;
  }

  /** Derive the status rows from the last known software check and setup check. */
  function statusRows(current, software, report) {
    const rows = [];
    if (software && software.status !== "ready") {
      rows.push(row(software.status === "launching" ? "checking" : "attention", "Software",
        software.status === "launching" ? "Checking" : (software.message || "Something is missing")));
    }
    const checks = report?.checks || {};
    const stale = !report?.checked_utc || report.status === "expired";
    const describe = (name, keys) => {
      const sample = checks[keys[0]];
      if (stale || !sample || sample.status === "not_checked" || sample.message === "Not run") return row("neutral", name, "Not checked yet");
      const failed = keys.find(key => checks[key]?.passed === false && !["not_checked", "launching"].includes(checks[key]?.status));
      if (failed) return row("attention", name, checks[failed].message);
      if (report.status === "checking" || sample.status === "launching") return row("checking", name, "Checking");
      return row(sample.passed ? "ready" : "neutral", name, sample.passed ? "OK" : sample.message);
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

  /** The session worth continuing: the open one, else the most recent unfinished one. */
  function continuable(current, list) {
    if (current.active_session && current.active_session.id !== "smoke") return current.active_session;
    const usable = list.filter(s => !["trashed", "purged"].includes(s.status) && s.source_kind !== undefined);
    return usable.find(s => (s.counts?.saved ?? 0) < (s.target_episodes ?? Infinity)) || null;
  }

  /** Build the three cards: new session, continue or practice run, review. */
  function cards(current, list, demo) {
    const r = window.SynriaRouter.routes;
    const out = [{kind:"link", title:"New session", text:"Choose the task, check the cameras and arm feed, then record takes.", verb:"Start a new session", route:r.wizard("task"), primary:true}];
    const session = continuable(current, list);
    if (session) {
      const saved = session.counts?.saved ?? 0, target = session.target_episodes ?? "?";
      const when = session.updated_utc ? `, last used ${ago(session.updated_utc)}` : "";
      out.push({kind:"continue", title:"Continue recording", text:`${String(session.task_id).replaceAll("_", " ")}, ${saved} of ${target} takes saved${when}.`, verb:"Continue", sessionId:session.id, open:current.active_session?.id === session.id});
    } else if (demo) {
      out.push({kind:"disabled", title:"Practice run", text:"Not available in demo mode.", verb:"Start a practice run"});
    } else {
      out.push({kind:"link", title:"Practice run", text:"20 seconds, not kept. Choose the task, then pick the practice run on Check setup.", verb:"Start a practice run", route:r.wizard("task")});
    }
    const takes = list.some(s => !["trashed", "purged"].includes(s.status) && (s.counts?.saved ?? 0) > 0);
    out.push(takes ? {kind:"link", title:"Review recordings", text:"Watch saved takes frame by frame.", verb:"Open review", route:r.sessions()}
      : {kind:"disabled", title:"Review recordings", text:"No takes recorded yet.", verb:"Open review"});
    return out;
  }

  const levelWords = {ready:"OK", attention:"Needs attention", checking:"Checking", neutral:"Not checked yet"};

  /** Paint the cards, the first-run strip and the status strip from current page state. */
  function renderHome() {
    if (typeof state === "undefined" || !document.getElementById("status-strip")) return;
    const software = window.SynriaSoftware || null;
    const report = typeof readinessReport === "undefined" ? null : readinessReport;
    const list = typeof sessions === "undefined" ? [] : sessions;
    const strip = document.getElementById("status-strip");
    const rows = statusRows(state, software, report);
    const signature = JSON.stringify(rows);
    if (strip.dataset.signature !== signature) {
      strip.dataset.signature = signature; strip.replaceChildren();
      for (const item of rows) {
        const li = document.createElement("li"); li.dataset.level = item.level;
        li.append(text("b", item.name), text("span", item.detail === levelWords[item.level] ? item.detail : `${levelWords[item.level]} · ${item.detail}`));
        strip.append(li);
      }
    }
    document.getElementById("status-note").textContent = report?.checked_utc && report.status !== "expired" ? `Last check: ${ago(report.checked_utc)}.` : "Not checked yet. Checked automatically when you start a session.";
    const container = document.getElementById("home-cards");
    const wanted = cards(state, list, state.demo);
    const cardSignature = JSON.stringify(wanted);
    if (container.dataset.signature !== cardSignature) {
      container.dataset.signature = cardSignature; container.replaceChildren();
      for (const card of wanted) {
        let node;
        if (card.kind === "link") { node = document.createElement("a"); node.href = window.SynriaRouter.href(card.route); node.dataset.route = card.route; }
        else { node = document.createElement("button"); node.type = "button"; if (card.kind === "disabled") { node.disabled = true; node.title = card.text; } }
        node.className = "home-card" + (card.primary ? " primary-card" : "");
        node.append(text("h3", card.title), text("p", card.text), text("span", card.verb));
        node.lastChild.className = "verb";
        if (card.kind === "continue") node.onclick = () => safe(async () => {
          if (!card.open) { await api(`session/${card.sessionId}/open`, {}); await refreshState(); await refreshSessions(); }
          window.SynriaRouter.go(window.SynriaRouter.routes.record(card.sessionId));
        });
        if (card.kind === "link" && card.primary) node.id = "home-session-link";
        container.append(node);
      }
    }
    let dismissed = false; try { dismissed = localStorage.getItem(FIRST_RUN_KEY) === "1"; } catch (reason) { dismissed = false; }
    const active = list.filter(s => !["trashed", "purged"].includes(s.status));
    document.getElementById("first-run-strip").hidden = dismissed || active.length > 0;
    document.getElementById("token-notice").hidden = Boolean(token);
    document.getElementById("home-check-now").disabled = Boolean(state.active_session) || (typeof readinessBusy !== "undefined" && readinessBusy);
  }

  /** Check now: run the read-only check in place when no bench checks are needed, else go where they are. */
  async function checkNow() {
    if (state.demo) { await runReadiness(); return; }
    window.SynriaRouter.go(window.SynriaRouter.routes.wizard("check"));
  }

  window.SynriaHome = {statusRows, cards, continuable, renderHome};
  if (typeof window !== "undefined" && typeof document !== "undefined") {
    for (const name of ["console-state", "console-readiness", "console-sessions", "console-route"]) window.addEventListener(name, renderHome);
    window.addEventListener("console-ready", () => {
      document.getElementById("home-check-now").onclick = () => safe(checkNow);
      document.getElementById("dismiss-first-run").onclick = () => { try { localStorage.setItem(FIRST_RUN_KEY, "1"); } catch (reason) { /* storage blocked */ } renderHome(); };
    }, {once:true});
  }
})();
