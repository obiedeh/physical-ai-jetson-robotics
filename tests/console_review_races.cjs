/** Verify browser review request ordering without a browser, network, or timing sleeps. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync(process.argv[2], "utf8");

/** Drain promise continuations without introducing a wall-clock delay. */
const flush = () => new Promise(resolve => setImmediate(resolve));

/** Build minimal inert DOM objects so exact pair updates can be observed. */
function environment() {
  const nodes = new Map();
  const renders = [];
  const context = {
    selected: {id: "one", ep: {episode_index: 0}}, frame: 0, frameGeneration: 0,
    playGeneration: 0, playing: false, rows: [], sessions: [], state: {},
    demoLabel: () => "", trace: (...values) => renders.push(values),
    $: id => {
      if (!nodes.has(id)) nodes.set(id, {
        id, textContent: "", max: 0, value: 0,
        replaceWith(image) { nodes.set(id, image); renders.push([id, image.src]); },
      });
      return nodes.get(id);
    },
    Promise, Number, Math, JSON,
  };
  return {context: vm.createContext(context), nodes, renders};
}

/** Older image requests cannot overwrite a later complete pair or paint only one view. */
async function pairedFrames() {
  const {context, nodes, renders} = environment();
  const pending = [];
  context.Image = class {
    decode() { return new Promise(resolve => pending.push({image: this, resolve})); }
  };
  context.api = async path => ({frame_index: Number(path.split("/").pop()), frame_count: 3,
    state_monotonic_s: [1], wrist_monotonic_s: [1], front_monotonic_s: [1], action_monotonic_s: [2]});
  vm.runInContext(source.slice(source.indexOf("async function showFrame()"),
    source.indexOf("async function advance()")), context);
  const older = context.showFrame(); await flush();
  assert.equal(pending.length, 2);
  context.frame = 1;
  const newer = context.showFrame(); await flush();
  assert.equal(pending.length, 4);
  pending[2].resolve(); await flush();
  assert.equal(renders.length, 0, "one decoded image must not paint independently");
  pending[3].resolve(); await newer;
  assert.equal(nodes.get("play-wrist").src, "/media/frame/one/0/1/wrist");
  assert.equal(nodes.get("play-front").src, "/media/frame/one/0/1/front");
  assert.equal(JSON.parse(nodes.get("frame-data").textContent).frame_index, 1);
  const rendered = renders.length;
  pending[0].resolve(); pending[1].resolve(); await older;
  assert.equal(renders.length, rendered, "late old frames must not replace the selected pair");
}

/** Changing episodes while rows or provenance are pending cannot mix the two records. */
async function selectedEpisode() {
  const {context, nodes} = environment();
  const pending = [];
  context.api = path => new Promise(resolve => pending.push({path, resolve}));
  context.showFrame = async () => { context.$("scrub").max = 0; };
  vm.runInContext(source.slice(source.indexOf("async function selectEpisode("),
    source.indexOf("async function showFrame()")), context);
  const older = context.selectEpisode("one", {episode_index: 0, operator_label: "success"});
  pending[0].resolve({from: "one"}); await flush();
  assert.match(pending[1].path, /frame\/one/);
  const newer = context.selectEpisode("two", {episode_index: 1, operator_label: "failure"});
  pending[1].resolve({"observation.state": [99], action: [99]}); await older;
  assert.equal(context.rows.length, 0, "old episode rows must not enter a new selection");
  pending[2].resolve({from: "two"}); await flush();
  pending[3].resolve({"observation.state": [2], action: [3]}); await newer;
  assert.equal(context.rows.length, 1);
  assert.equal(context.rows[0].state[0], 2);
  assert.equal(JSON.parse(nodes.get("episode-provenance").textContent).from, "two");
}

Promise.resolve().then(pairedFrames).then(selectedEpisode).catch(error => {
  console.error(error); process.exitCode = 1;
});
