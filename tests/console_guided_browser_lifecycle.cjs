/** Test owned browser teardown ordering with inert child-process lifecycle events. */
const assert = require("node:assert/strict");
const {EventEmitter} = require("node:events");
const {ownedBrowserShutdown} = require("./console_guided_browser.cjs");

/** Resolve after one event-loop turn without creating a browser or any device interface. */
const tick = () => new Promise(resolve => setImmediate(resolve));

/** Exercise shutdown contracts using only the child instance supplied to the owner. */
async function check(scenario) {
  const child = new EventEmitter();
  const signals = [];
  child.kill = signal => { signals.push(signal); return true; };
  const stop = ownedBrowserShutdown(child);
  if (scenario === "closed") {
    child.emit("close", 0);
    await stop(20);
    assert.deepEqual(signals, [], "a closed child receives no signal");
  } else if (scenario === "delayed_close" || scenario === "exit_before_close") {
    let finished = false;
    const stopped = stop(100).then(() => { finished = true; });
    if (scenario === "exit_before_close") child.emit("exit", 0);
    await tick();
    assert.equal(finished, false, "process exit alone cannot release the profile");
    child.emit("close", 0);
    await stopped;
    assert.deepEqual(signals, [], "graceful close avoids unnecessary signals");
  } else if (scenario === "signal_escalation") {
    child.kill = signal => {
      signals.push(signal);
      if (signal === "SIGKILL") setImmediate(() => child.emit("close", null, signal));
      return true;
    };
    await stop(5);
    assert.deepEqual(signals, ["SIGTERM", "SIGKILL"]);
  } else if (scenario === "stuck") {
    await assert.rejects(stop(5), /profile retained/);
    assert.deepEqual(signals, ["SIGTERM", "SIGKILL"]);
  } else {
    throw Error("unknown lifecycle scenario");
  }
  console.log("Owned browser lifecycle contract passed: " + scenario);
}

check(process.argv[2]).catch(error => { console.error(error); process.exitCode = 1; });
