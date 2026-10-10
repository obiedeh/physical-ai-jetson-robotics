/** Exercise shipped browser UI with synthetic HTTP failures; no real source or robot API exists here. */
const assert = require("node:assert/strict"), fs = require("node:fs"), http = require("node:http");
const os = require("node:os"), path = require("node:path"), {spawn} = require("node:child_process");
const [staticRoot, browser, screenshot] = process.argv.slice(2);
const calls = [], exceptions = [];
let offline = false, scenario = "timeout", allowSave = false;
let report = {status:"unchecked", label:"Not checked", percent:0, checks:{}, form_settings:{}, checked_utc:null,
  passed_count:0, check_count:0, demo:false, scope:"Synthetic test backend · no hardware", expires_after_s:60};
const state = {demo:false, active_session:null, capture:{state:"closed"}, preflight:{}, free_disk_bytes:1e11, min_free_bytes:1e9};
const task = {task_id:"die_into_cup", task_text:"Synthetic fixture: put die into cup", success_rule:"Operator and still", scene_requirements:["Fixed synthetic scene"], episode_window:{min_episode_s:null, max_episode_s:null}};

/** Return a realistic refusal snapshot from fake state without constructing any runtime. */
function diagnostic(settings) {
  const check = (passed, message) => ({passed, status:passed?"ready":"not_ready", message});
  const checks = {operator_safety:check(true,"Confirmed"), form_configuration:check(true,"Validated"),
    recording_metadata:check(false,"Required before recording: follower_serial, power_state_start"),
    qualifying_configuration:check(false,"episode window unset; operator timing required"), limits_verification:check(false,"limits unverified"),
    state_subscription:{...check(true,"Subscription created; not proof of data"),status:"configured"}};
  for (const name of ["follower_sample","action_sample","wrist_sample","front_sample"])
    checks[name] = {passed:false,status:"not_checked",message:"Not run"};
  if (scenario === "timeout") checks.state_rate = check(false,"No state sample received on /joint_states after waiting 10 s");
  if (scenario === "camera") {
    for (const name of ["follower_sample","action_sample","front_sample"]) checks[name] = check(true,"Fresh synthetic sample");
    checks.wrist_sample = check(false,"Camera device busy; cannot open selected wrist camera");
  }
  if (scenario === "recovery") checks.dataset_lock_and_recovery = check(false,"Recovery blocked: journal requires review");
  return {status:"blocked", label:"Connection checks incomplete", percent:25, checks,
    form_settings:settings, checked_utc:new Date().toISOString(), passed_count:3, check_count:Object.keys(checks).length,
    demo:false, scope:"Synthetic test backend · no hardware", expires_after_s:60};
}

/** Serve packaged assets and only inert API fixtures on an ephemeral loopback listener. */
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  if (!url.pathname.startsWith("/api/")) {
    const file = url.pathname === "/" ? "index.html" : path.basename(url.pathname);
    if(!fs.existsSync(path.join(staticRoot,file))){res.writeHead(404);res.end();return;}
    const content = fs.readFileSync(path.join(staticRoot, file));
    res.writeHead(200,{"Content-Type":file.endsWith(".js")?"text/javascript":file.endsWith(".css")?"text/css":"text/html"});
    res.end(content);return;
  }
  const route = url.pathname.slice(5); let value;
  if (offline) {res.writeHead(503,{"Content-Type":"application/json"});res.end(JSON.stringify({error:"Network connection lost; state unavailable"}));return;}
  if (req.method === "POST") {
    let body="";for await (const part of req) body+=part;
    const payload=JSON.parse(body);calls.push({route,payload});
    if(route==="readiness"){report=diagnostic(payload.settings);value=report;}
    else if(route==="record/retry") {
      if(!allowSave){state.capture.error="save failed: disk full; pending frames retained";res.writeHead(409,{"Content-Type":"application/json"});res.end(JSON.stringify({error:state.capture.error}));return;}
      state.capture={state:"idle",pending_frame_count:0,controls:{start:true},last_episode:{episode_index:0,operator_label:"failure"}};
      value=state;
    }
    else {res.writeHead(409,{"Content-Type":"application/json"});res.end(JSON.stringify({error:"Unexpected mutation in fake browser test"}));return;}
  } else if(route==="state")value=state;
  else if(route==="sessions")value=[];
  else if(route==="tasks")value=[task];
  else if(route==="readiness")value=report;
  else if(route==="cameras")value={cameras:["fake-wrist-video-index0","fake-front-video-index0"],message:"Synthetic identifiers"};
  else if(route==="serial-connections")value={connections:[],message:"No physical connections in test"};
  else if(route==="software")value={status:"ready",label:"Ready",dependencies:{lerobot:true,cv2:true,rclpy:true,sensor_msgs:true,ffmpeg:true},message:"Fake dependency discovery"};
  else if(route==="operator-guide")value={environment:{host:"synthetic-test",python:"fake-python",ros_domain:"23",ros_distribution:"fake",middleware:"fake"},limits_verified:false,documents:{}};
  else if(route.startsWith("help/"))value={path:"docs/ludo_flagship/D1_OPERATOR_RUNBOOK.md",text:"Synthetic reference fixture. Never start a second follower."};
  else if(route==="setup-suggestions")value={host:"synthetic-test",account:"test",free_disk_bytes:1e11,
    recommended:{fps:15,image_width:224,image_height:224,action_lookahead_steps:1,state_startup_timeout_s:10,target_episodes:100},
    previous_operator_entries:{operator:"test",scene:"synthetic",gripper_type:"50mm",state_source:"ros2_control",wrist_camera:"fake-wrist-video-index0",front_camera:"fake-front-video-index0"},notice:"Synthetic fixtures only"};
  else {res.writeHead(404);res.end("{}");return;}
  res.writeHead(200,{"Content-Type":"application/json"});res.end(JSON.stringify(value));
});

/** Launch an isolated installed browser and drive only its synthetic loopback page through its pipe. */
async function run() {
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  const profile=fs.mkdtempSync(path.join(os.tmpdir(),"synria-guide-test-"));
  const child=spawn(browser,["--headless","--disable-gpu","--disable-background-networking","--no-first-run",
    "--no-default-browser-check",`--user-data-dir=${profile}`,"--remote-debugging-pipe"],{stdio:["ignore","ignore","pipe","pipe","pipe"]});
  let sequence=0, buffer="", sessionId, browserError="";const pending=new Map();
  child.stderr.on("data",chunk=>{browserError=(browserError+chunk.toString()).slice(-8000);});
  child.stdio[4].on("data",chunk=>{
    buffer+=chunk.toString();let boundary;
    while((boundary=buffer.indexOf("\0"))>=0){const message=JSON.parse(buffer.slice(0,boundary));buffer=buffer.slice(boundary+1);
      if(message.id){const task=pending.get(message.id);if(task){pending.delete(message.id);clearTimeout(task.timer);message.error?task.reject(Error(JSON.stringify(message.error))):task.resolve(message.result);}}
      else if(message.method==="Runtime.exceptionThrown")exceptions.push(message.params.exceptionDetails);
    }
  });
  /** Send one bounded browser protocol request, keeping target commands scoped to this isolated page. */
  function send(method,params={},target=sessionId){return new Promise((resolve,reject)=>{
    const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error(`Protocol timeout: ${method}\n${browserError}`));},15000);
    pending.set(id,{resolve,reject,timer});child.stdio[3].write(JSON.stringify({id,method,params,...(target?{sessionId:target}:{})})+"\0");
  });}
  /** Evaluate test interactions and propagate browser exceptions rather than treating them as success. */
  async function evaluate(expression){const result=await send("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
  /** Wait on DOM mutations, not arbitrary sleeps, for an asynchronous UI condition to become true. */
  async function until(expression){return evaluate(`new Promise((resolve,reject)=>{const test=()=>(${expression});if(test())return resolve(true);const observer=new MutationObserver(()=>{if(test()){observer.disconnect();clearTimeout(timer);resolve(true);}});observer.observe(document,{subtree:true,childList:true,attributes:true,characterData:true});const timer=setTimeout(()=>{observer.disconnect();reject(Error('UI condition timed out: '+${JSON.stringify(expression)}));},10000);})`);}
  try {
    const target=await send("Target.createTarget",{url:"about:blank"},null);
    sessionId=(await send("Target.attachToTarget",{targetId:target.targetId,flatten:true},null)).sessionId;
    await send("Runtime.enable");await send("Page.enable");
    await send("Emulation.setDeviceMetricsOverride",{width:1360,height:1000,deviceScaleFactor:1,mobile:false});
    const url=`http://127.0.0.1:${server.address().port}/#token=synthetic-test-token`;
    await send("Page.navigate",{url});
    await until("window.synriaConsoleReady && document.querySelector('#environment-facts').textContent.includes('synthetic-test') && document.querySelector('[name=gripper_type]').value==='50mm'");
    assert.equal(await evaluate("document.querySelector('#setup-prepare').hidden"),false);
    assert.equal(calls.length,0,"page load and software check cannot start sources");
    await evaluate("document.querySelector('[data-guide-help=source]').click()");
    assert.match(await evaluate("document.querySelector('#help-content').textContent"),/No verified startup command/);
    assert.match(await evaluate("document.querySelector('#help-content').textContent"),/joint_commands_enabled:=false/);
    assert.match(await evaluate("document.querySelector('#help-content pre').textContent"),/--ros-args \\\n/);
    await evaluate("document.querySelector('#close-help').click();document.querySelector('[data-setup-step=configure]').click()");
    assert.equal(await evaluate("document.querySelector('#setup-configure').hidden"),false);
    assert.equal(await evaluate("new Set([...document.querySelectorAll('[id]')].map(n=>n.id)).size===document.querySelectorAll('[id]').length"),true,"no duplicate controls");
    await evaluate("const fpsPreset=document.querySelector('[data-preset-for=fps]');fpsPreset.value='custom';fpsPreset.dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[name=fps]').value='0';document.querySelector('[data-setup-step=check]').click();document.querySelector('#check-readiness').click()");
    assert.equal(calls.length,0,"invalid native form constraints cannot open sources");
    assert.equal(await evaluate("document.querySelector('#setup-configure').hidden"),false,"invalid hidden form field opens its configuration step");
    await evaluate("document.querySelector('[data-preset-for=fps]').value='15';document.querySelector('[data-preset-for=fps]').dispatchEvent(new Event('change',{bubbles:true}))");
    if(process.env.SYNRIA_GUIDE_EXTRA_SHOTS)fs.writeFileSync(screenshot.replace(/\.png$/,"-configuration.png"),Buffer.from((await send("Page.captureScreenshot",{format:"png"})).data,"base64"));
    await evaluate("document.querySelector('[name=scene]').value='<script>throw Error(99)</script>';document.querySelector('[name=scene]').dispatchEvent(new Event('input',{bubbles:true}));document.querySelector('[data-setup-step=check]').click()");
    await evaluate("document.querySelector('#check-readiness').click()");
    await until("document.querySelector('#notice').textContent.includes('physical safety confirmations')");
    assert.equal(calls.length,0,"missing physical confirmations cannot issue a diagnostic");
    await evaluate("document.querySelectorAll('[data-readiness-confirm]').forEach(n=>{n.checked=true;n.dispatchEvent(new Event('change',{bubbles:true}));});document.querySelector('#check-readiness').click()");
    await until("document.querySelector('#next-action-title').textContent==='No follower state received'");
    assert.equal(await evaluate("document.querySelector('#notice').textContent"),"","a deliberate valid retry supersedes the resolved consent error");
    assert.equal(calls.length,1);assert.equal(calls[0].payload.settings.follower_serial,"");
    assert.match(await evaluate("document.querySelector('#connection-paths').textContent"),/Not ready.*Follower state/s);
    assert.match(await evaluate("document.querySelector('#connection-paths').textContent"),/Not checked.*Wrist camera/s);
    assert.ok(!await evaluate("document.querySelector('#all-check-details').textContent.includes('Ready · state source')"));
    if(process.env.SYNRIA_GUIDE_EXTRA_SHOTS)fs.writeFileSync(screenshot.replace(/\.png$/,"-failure.png"),Buffer.from((await send("Page.captureScreenshot",{format:"png"})).data,"base64"));
    await evaluate("document.querySelector('#next-action-button').click()");
    assert.match(await evaluate("document.querySelector('#help-content').textContent"),/ROS domain/);
    await evaluate("document.querySelector('#close-help').click()");
    scenario="camera";await evaluate("document.querySelector('#check-readiness').click()");
    await until("document.querySelector('#connection-paths').textContent.includes('Camera device busy')");
    assert.match(await evaluate("document.querySelector('#next-action-title').textContent"),/camera/);
    await evaluate("document.querySelector('#next-action-button').click()");
    assert.match(await evaluate("document.querySelector('#help-content').textContent"),/pgrep -a cheese/);
    await evaluate("document.querySelector('#close-help').click();document.querySelector('[name=fps]').value='30';document.querySelector('[name=fps]').dispatchEvent(new Event('change',{bubbles:true}));");
    assert.equal(await evaluate("[...document.querySelectorAll('#connection-paths article')].every(n=>n.dataset.level==='not_checked')"),true,"changed settings invalidate every old path");
    scenario="recovery";await evaluate("document.querySelector('#check-readiness').click()");
    await until("document.querySelector('#next-action-title').textContent==='Preserve the dataset for recovery'");
    await evaluate("document.querySelector('#next-action-button').click()");
    assert.match(await evaluate("document.querySelector('#help-content').textContent"),/Do not remove the lock/);
    await evaluate("document.querySelector('#close-help').click();document.querySelector('[data-setup-step=collect]').click()");
    assert.equal(await evaluate("document.querySelector('#create').disabled"),true,"unset real task timing blocks creation");
    assert.equal(await evaluate("document.querySelector('#smoke').disabled"),true,"missing identity cannot start smoke");
    state.active_session={id:"pending",name:"Synthetic pending session"};
    state.capture={state:"stopped",pending_frame_count:2,controls:{success:true,failure:true,discard:true}};
    await evaluate("(async()=>{await refreshState();screen('record');SynriaGuide.openHelp('save');})()");
    assert.equal(await evaluate("lastCapture"),"stopped");
    assert.equal(await evaluate("state.capture.controls.success"),true);
    const protectedCount=calls.length;
    await evaluate("document.dispatchEvent(new KeyboardEvent('keydown',{code:'Space',key:' '}));document.dispatchEvent(new KeyboardEvent('keydown',{key:'s'}))");
    assert.equal(calls.length,protectedCount,"help-dialog keyboard use cannot label or start a recording");
    await evaluate("document.querySelector('#close-help').click();document.activeElement.blur();document.dispatchEvent(new KeyboardEvent('keydown',{key:'s',ctrlKey:true}))");
    assert.equal(calls.length,protectedCount,"browser shortcuts cannot label an episode");
    state.capture.error="Previous save failed; pending frames retained";
    state.capture.controls.retry=true;
    await evaluate("refreshState()");
    await evaluate("document.querySelector('[data-command=retry]').click()");
    await until("document.querySelector('#notice').textContent.includes('disk full')");
    assert.equal(state.capture.pending_frame_count,2,"failed save retains pending state");
    allowSave=true;
    await evaluate("document.querySelector('[data-command=retry]').click()");
    await until("document.querySelector('#notice').textContent==='' && state.capture.pending_frame_count===0");
    assert.equal(calls.filter(call=>call.route==='record/retry').length,2,"only two deliberate save retries occur");
    state.active_session=null;state.capture={state:"closed"};
    const mutations=calls.length;offline=true;
    await evaluate("refreshState().catch(error)");
    assert.match(await evaluate("document.querySelector('#notice').textContent"),/Reconnect without losing/);
    offline=false;await evaluate("refreshState()");
    assert.match(await evaluate("document.querySelector('#notice').textContent"),/Network connection lost/,"polling cannot erase an actionable error");
    assert.equal(calls.length,mutations,"reconnection never retries a mutation");
    await send("Page.reload");
    await until("window.synriaConsoleReady && document.querySelector('#environment-facts').textContent.includes('synthetic-test')");
    assert.equal(calls.length,mutations,"reload never restarts a diagnostic");
    await evaluate("document.querySelector('[data-setup-step=check]').click()");
    await send("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});
    assert.equal(await evaluate("document.documentElement.scrollWidth<=window.innerWidth+2"),true,"mobile layout must not overflow");
    await send("Emulation.setDeviceMetricsOverride",{width:1360,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate("document.querySelector('#mode').textContent='SYNTHETIC UI TEST · NO HARDWARE';document.body.classList.add('demo');document.querySelector('[data-setup-step=prepare]').click()");
    fs.writeFileSync(screenshot,Buffer.from((await send("Page.captureScreenshot",{format:"png"})).data,"base64"));
    assert.deepEqual(exceptions,[],"no uncaught browser errors");
    console.log("Guided browser: setup, help, state timeout, camera fault, recovery, stale settings, consent, keyboard, save retries, network, reload and mobile checks passed.");
  } finally {
    try{await send("Browser.close",{},null);}catch{}
    child.kill("SIGTERM");
    for(const entry of pending.values())clearTimeout(entry.timer);
    server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
    fs.rmSync(profile,{recursive:true,force:true});
  }
}
run().catch(error=>{console.error(error);server.closeAllConnections();server.close();process.exitCode=1;});
