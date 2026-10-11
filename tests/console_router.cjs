/** Check hash-route parsing, addresses and the Home status strip without a browser or devices. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const [routerPath, homePath] = process.argv.slice(2);

const context = vm.createContext({window: {addEventListener() {}}, location: {hash: "", pathname: "/"}, history: {replaceState() {}}});
vm.runInContext(fs.readFileSync(routerPath, "utf8"), context);
const router = context.window.SynriaRouter;

assert.equal(router.parse(""), null, "an empty hash has no route");
assert.equal(router.parse("#token=abc"), null, "a launch token is not a route");
assert.equal(router.parse("#/nowhere"), null, "unknown places are rejected, not guessed");
assert.equal(router.parse("#/home").name, "home");
assert.equal(router.parse("#/sessions/").name, "sessions", "a trailing slash is tolerated");
assert.equal(router.parse("#/help").name, "help");
assert.equal(JSON.stringify(router.parse("#/new")), JSON.stringify({name: "new", path: "new", step: "task"}), "the wizard opens on its first step");
assert.equal(router.parse("#/new/check").step, "check");
assert.equal(router.parse("#/new/other"), null, "only the three wizard steps exist");
const record = router.parse("#/session/abc%20def/record");
assert.equal(record.name, "record"); assert.equal(record.sessionId, "abc def");
const review = router.parse("#/session/s1/review/4");
assert.equal(review.name, "review"); assert.equal(review.sessionId, "s1"); assert.equal(review.episode, 4);
assert.equal(router.parse("#/session/s1/review").episode, null);
assert.equal(router.href("sessions"), "#/sessions");
assert.equal(router.href("#/sessions"), "#/sessions", "href never doubles the prefix");
assert.equal(router.routes.record("a b"), "session/a%20b/record", "session identities are encoded in addresses");
assert.equal(router.routes.review("s1", 2), "session/s1/review/2");
assert.equal(router.routes.wizard(), "new/task");
for (const path of ["home", "sessions", "help", "new/task", "new/check", "new/start", "session/s1/record", "session/s1/review/0"]) {
  assert.equal(router.parse(router.href(path)).path, path, `round trip for ${path}`);
}

// Home status rows are words plus a level, derived only from state the page already has.
const homeContext = vm.createContext({window: {}, document: undefined, text: (tag, value) => ({tag, value})});
vm.runInContext(fs.readFileSync(homePath, "utf8"), homeContext);
const home = homeContext.window.SynriaHome;
let rows = home.statusRows({active_session: null, free_disk_bytes: 5e10, min_free_bytes: 1e9}, null, null);
assert.equal(rows[0].detail, "None open");
assert.equal(rows[1].detail, "Not checked yet");
assert.ok(rows.slice(2, 5).every(row => row.level === "neutral" && row.detail.startsWith("Not checked")), "no check means unknown, never failed");
assert.equal(rows[5].level, "ready"); assert.match(rows[5].detail, /^50\.0 GB free$/);
const checks = {follower_sample: {passed: true, status: "ready", message: "Fresh sample"},
  wrist_sample: {passed: false, status: "not_ready", message: "Camera device busy"},
  front_sample: {passed: false, status: "not_checked", message: "Not run"}};
rows = home.statusRows({active_session: {name: "Morning"}, free_disk_bytes: 1e8, min_free_bytes: 1e9},
  {status: "ready"}, {checked_utc: "2026-10-11T10:00:00Z", status: "blocked", checks});
assert.equal(rows[0].detail, "Morning is open");
assert.equal(rows[1].detail, "Recording libraries found");
assert.equal(rows[2].level, "ready"); assert.match(rows[2].detail, /Fresh sample \(last setup check 2026-10-11 10:00 UTC\)/);
assert.equal(rows[3].level, "attention"); assert.equal(rows[3].detail, "Camera device busy");
assert.equal(rows[4].level, "neutral");
assert.equal(rows[5].level, "attention"); assert.match(rows[5].detail, /below the recording floor/);
rows = home.statusRows({active_session: null}, {status: "not_ready", message: "Missing rclpy"}, {checked_utc: "2026-10-11T10:00:00Z", status: "expired", checks});
assert.equal(rows[1].level, "attention"); assert.equal(rows[1].detail, "Missing rclpy");
assert.ok(rows[2].detail.startsWith("Not checked"), "an expired check is not shown as a fresh sample");
console.log("router and home status verified");
