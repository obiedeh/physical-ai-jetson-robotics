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
let rows = home.statusRows({active_session: null, free_disk_bytes: 5e10, min_free_bytes: 1e9}, {status: "ready"}, null);
assert.equal(rows.map(r => r.name).join(), "Arm feed,Wrist camera,Overhead camera,Storage", "four tiles, software hidden when it is fine");
assert.ok(rows.slice(0, 3).every(r => r.level === "neutral" && r.detail === "Not checked yet"), "no check means Not checked yet, never a fault");
assert.equal(rows[3].level, "ready"); assert.match(rows[3].detail, /^50\.0 GB free$/);
const simRows = home.statusRows({active_session: null, simulation: true, free_disk_bytes: 5e10, min_free_bytes: 1e9}, {status: "ready"}, null);
assert.equal(simRows.map(r => r.name).join(), "Sim arm feed,Sim wrist camera,Sim front camera,Storage", "simulation mode names the sim sources");
const checks = {follower_sample: {passed: true, status: "ready", message: "Fresh sample"},
  wrist_sample: {passed: false, status: "not_ready", message: "Camera device busy"},
  front_sample: {passed: false, status: "not_checked", message: "Not run"}};
const fresh = new Date(Date.now() - 2 * 60 * 1000).toISOString();
rows = home.statusRows({active_session: {name: "Morning"}, free_disk_bytes: 1e8, min_free_bytes: 1e9},
  {status: "not_ready", message: "Missing rclpy"}, {checked_utc: fresh, status: "expired", checks});
assert.equal(rows[0].name, "Software"); assert.equal(rows[0].level, "attention"); assert.equal(rows[0].detail, "Missing rclpy");
assert.equal(rows[1].level, "ready"); assert.equal(rows[1].detail, "OK, 2 minutes ago", "a recent result keeps its verdict and age even after the server's 60 s expiry");
assert.equal(rows[2].level, "attention"); assert.equal(rows[2].detail, "Camera device busy, 2 minutes ago");
assert.equal(rows[3].level, "neutral");
assert.equal(rows[4].level, "attention"); assert.match(rows[4].detail, /below the recording floor/);
rows = home.statusRows({active_session: null}, {status: "ready"}, {checked_utc: "2026-10-11T10:00:00Z", status: "ready", checks});
assert.equal(rows[0].detail, "Not checked recently", "a result older than ten minutes is not shown as current");

// Cards: things to do, with Continue only when an unfinished session exists.
homeContext.window.SynriaRouter = {routes: {wizard: step => `new/${step}`, sessions: () => "sessions", record: id => `session/${id}/record`}, href: path => "#/" + path};
let cards = home.cards({active_session: null}, [], false);
assert.equal(cards.map(c => c.title).join("|"), "New session|Practice run|Review recordings");
assert.equal(cards.map(c => c.verb).join("|"), "Start a new session|Start a practice run|Open review");
assert.equal(cards[2].kind, "disabled", "review is disabled until a take exists");
assert.equal(cards[0].primary, true, "with nothing to continue, New session is the highlighted card");
cards = home.cards({active_session: null}, [], true);
assert.equal(cards[1].kind, "disabled", "demo mode has no practice run");
const paused = {id: "s1", name: "Morning", task_id: "die_into_cup", status: "closed", source_kind: "physical", target_episodes: 10, counts: {saved: 4}, updated_utc: new Date().toISOString()};
cards = home.cards({active_session: null}, [paused], false);
assert.equal(cards[1].title, "Continue recording"); assert.equal(cards[1].verb, "Continue");
assert.equal(cards[1].primary, true, "Continue is the highlighted card when a session can be continued"); assert.equal(cards[0].primary, false);
assert.match(cards[1].text, /^die into cup, 4 of 10 takes saved, last used just now\.$/);
assert.equal(cards[1].open, false, "a paused session is reopened before recording");
assert.equal(cards[2].kind, "link", "saved takes enable review");
const done = {...paused, counts: {saved: 10}};
cards = home.cards({active_session: null}, [done], false);
assert.equal(cards[1].title, "Practice run", "a session at its target is not offered to continue");
cards = home.cards({active_session: null}, [{...paused, status: "finished"}], false);
assert.equal(cards[1].title, "Practice run", "a finished session is not offered to continue");
cards = home.cards({active_session: {id: "s1"}}, [paused], false);
assert.equal(cards[1].open, true);
const finishedLater = {...paused, id: "s2", status: "finished", counts: {saved: 10}};
cards = home.cards({active_session: null}, [finishedLater, paused], false);
assert.equal(cards[1].title, "Continue recording", "the latest paused session is still offered");
assert.equal(cards[1].primary, false, "but not highlighted when the most recent session was finished");
assert.equal(cards[0].primary, true, "New session is the highlight again");
console.log("router and home status verified");
