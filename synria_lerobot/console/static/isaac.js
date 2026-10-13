/** Isaac record mode panel for the simulation console: live state, one button, Stop, auto check.
 *
 * Polls /api/isaac while the console runs in simulation mode. The button never sends text to
 * the server: start and stop are fixed actions. When the state turns to running, the setup
 * check runs by itself once.
 */
(function () {
  let last = null, autoRan = false, busy = false, timer = null;
  const levelClass = {running: "", starting: "", other_running: "demo", not_running: "demo"};

  /** The latest Isaac state seen, for Home and tests. */
  function lastState() { return last; }

  /** Words for one instance line: kind, start time, GPU memory, record-mode flag. */
  function instanceLine(item) {
    const when = item.started_utc ? new Date(item.started_utc).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "unknown time";
    const gpu = typeof item.gpu_mb === "number" ? `${(item.gpu_mb / 1024).toFixed(1)} GB GPU` : "GPU use unknown";
    return `${item.kind}${item.record_mode ? " (record mode)" : ""}, started ${when}, ${gpu}, pid ${item.pid}`;
  }

  /** Progress words for the Starting state. */
  function progressLines(progress, status) {
    const tick = ok => (ok ? "done" : "waiting");
    const rate = typeof progress.state_rate_hz === "number" ? `${progress.state_rate_hz.toFixed(1)} Hz` : "no messages yet";
    if (status === "not_running") return [];
    return [`Process up: ${tick(progress.process_up)}`, `Scene loaded: ${tick(progress.scene_loaded)}`,
      `Sim topics publishing: ${tick(progress.topics_started)} (${rate}, needs 15 Hz)`];
  }

  /** Draw the panel from a state object; pure DOM, no requests. */
  function render(isaac) {
    last = isaac;
    const panel = document.getElementById("isaac-panel");
    if (!panel) return;
    const sim = typeof state !== "undefined" && state && state.simulation;
    panel.hidden = !sim || !isaac;
    if (!sim || !isaac) return;
    const status = document.getElementById("isaac-status");
    status.textContent = isaac.label; status.className = `badge ${levelClass[isaac.status] || ""}`;
    const progress = document.getElementById("isaac-progress"); progress.replaceChildren();
    for (const line of progressLines(isaac.progress || {}, isaac.status)) { const li = document.createElement("li"); li.textContent = line; progress.append(li); }
    const list = document.getElementById("isaac-instances"); list.replaceChildren();
    for (const item of isaac.instances || []) { const li = document.createElement("li"); li.textContent = instanceLine(item); list.append(li); }
    const driver = isaac.driver || {};
    document.getElementById("isaac-driver").textContent = driver.running
      ? `Arm driver is running (reused, never restarted by this page).`
      : `Arm driver is not running. The button starts only the read-only form: ${driver.read_only_command || ""}`;
    const driverStart = document.getElementById("isaac-driver-start");
    driverStart.hidden = Boolean(driver.running); driverStart.disabled = busy || !driver.can_start;
    const closes = document.getElementById("isaac-closes");
    const willClose = (isaac.button && isaac.button.closes) || [];
    closes.hidden = willClose.length === 0;
    closes.textContent = willClose.length ? "This will close: " + willClose.map(instanceLine).join("; ") + "." : "";
    const action = document.getElementById("isaac-action");
    action.textContent = isaac.button ? isaac.button.label : "Start Isaac record mode";
    action.disabled = !isaac.available || busy || !(isaac.button && isaac.button.action);
    const stop = document.getElementById("isaac-stop"); stop.hidden = !isaac.can_stop; stop.disabled = busy;
    const note = document.getElementById("isaac-note");
    note.textContent = !isaac.available ? "Launcher path is not configured: set SYNRIA_ISAAC_LAUNCHER or isaac_launcher.json in the workspace."
      : isaac.failure ? `Last launch failed: ${isaac.failure}. Log: ${isaac.launch_log || "unknown"}`
      : isaac.status === "running" ? `Teleop commit ${isaac.teleop_commit}. The setup check runs by itself.` : "";
  }

  /** Fetch and draw; once the state is running, run the setup check by itself one time. */
  async function poll() {
    if (typeof state === "undefined" || !state || !state.simulation) return;
    try { render(await api("isaac")); } catch (reason) { return; }
    if (typeof window !== "undefined") window.dispatchEvent(new Event("console-isaac"));
    if (last && last.status === "running" && !autoRan && typeof runReadiness === "function"
        && !document.getElementById("step-check")?.hidden && !(typeof readinessBusy !== "undefined" && readinessBusy)) {
      autoRan = true;
      try { await runReadiness(); } catch (reason) { /* shown by the readiness panel */ }
    }
    if (last && last.status !== "running") autoRan = false;
  }

  /** Send one fixed action; the server accepts no parameters. */
  async function act(name) {
    busy = true;
    if (name === "start" && last) {
      // Say Starting at once; the server's own state replaces this on the next poll.
      render({...last, status: "starting", label: "Starting Isaac record mode", button: {action: null, label: "Starting, please wait", closes: []}, failure: null});
      if (typeof window !== "undefined") window.dispatchEvent(new Event("console-isaac"));
    } else render(last);
    try { render(await api(`isaac/${name}`, {})); } finally { busy = false; render(last); if (typeof window !== "undefined") window.dispatchEvent(new Event("console-isaac")); }
  }

  /** Whether a healthy record-mode instance is up, as the server last reported. */
  function running() { return Boolean(last && last.status === "running"); }
  /** The phase the wizard footer should name: start, waiting or ready. */
  function phase() { if (!last || last.status === "not_running" || last.status === "other_running") return "start"; return last.status === "starting" ? "waiting" : "ready"; }

  window.SynriaIsaac = {last: lastState, render, progressLines, instanceLine, running, phase, start: () => act("start")};
  if (typeof window !== "undefined" && typeof document !== "undefined") {
    window.addEventListener("console-ready", () => {
      document.getElementById("isaac-action").onclick = () => safe(() => act("start"));
      document.getElementById("isaac-stop").onclick = () => safe(async () => { if (confirm("Stop Isaac record mode? Takes already saved are kept.")) await act("stop"); });
      document.getElementById("isaac-driver-start").onclick = () => safe(() => act("start-driver"));
      timer = setInterval(() => safe(poll), 2000); safe(poll);
    });
  }
})();
