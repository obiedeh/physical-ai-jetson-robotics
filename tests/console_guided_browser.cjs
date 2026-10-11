/** Exercise shipped browser UI with synthetic HTTP failures; no real source or robot API exists here. */
const assert = require("node:assert/strict"), fs = require("node:fs"), http = require("node:http");
const os = require("node:os"), path = require("node:path"), {spawn} = require("node:child_process");
const [staticRoot, browser, screenshot] = process.argv.slice(2);
const calls = [], exceptions = [];
let offline = false, scenario = "timeout", allowSave = false, expireReports = false;
let report = {status:"unchecked", label:"Not checked", percent:0, checks:{}, form_settings:{}, checked_utc:null,
  passed_count:0, check_count:0, demo:false, scope:"Synthetic test backend · no hardware", expires_after_s:60};
const state = {demo:false, active_session:null, capture:{state:"closed"}, preflight:{}, free_disk_bytes:1e11, min_free_bytes:1e9};
const task = {task_id:"die_into_cup", task_text:"Synthetic fixture: put die into cup", success_rule:"Operator and still", scene_requirements:["Fixed synthetic scene"], episode_window:{min_episode_s:null, max_episode_s:null}};
const sessionList = [];
const libraryTasks = [];
const timing = {running:false,polls:0,runs:[],proposed_window:null,phase:"not started"};
/** A fixed breakdown the fake planner returns for any sentence. */
function fakeBreakdown(sentence){const step=(name,start,end,kind)=>({name,instruction:`Do ${name.toLowerCase()} and return to ready.`,start_state:start,end_state:end,success_rule:`${name} is done.`,scene_before_take:["Gripper empty, arm at ready."],kind,objects_grasped:["die"],holds_at_start:false,holds_at_end:false,camera_can_confirm:true,suggested_takes:5,optional:false,primitives:["pick"]});
  return {schema_version:"synria_task_breakdown_v1",goal:{title:"Roll a die",sentence,rationale:"Two steps so the cycle can repeat."},sub_tasks:[step("Die into cup","Die in start zone, gripper empty, arm at ready","Die in cup, gripper empty, arm at ready","skill"),step("Die back to start","Die in cup, gripper empty, arm at ready","Die in start zone, gripper empty, arm at ready","reset")]};}

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
  if (scenario === "pass") {
    for (const name of ["follower_sample","action_sample","wrist_sample","front_sample"]) checks[name] = check(true,"Fresh synthetic sample");
    checks.state_rate = check(true,"25 Hz measured");
    checks.state_rate_evidence = {...check(true,"Shared recorder check passed"),details:{passed:true,rate_hz:25.04,message_count:50,duration_s:2}};
  }
  const status = scenario === "pass" ? "attention" : "blocked";
  return {status, label:status === "attention" ? "Connected · setup required" : "Connection checks incomplete", percent:status === "attention" ? 70 : 25, checks,
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
    else if(route==="sessions"){const session={id:"s1",name:payload.name,task_id:payload.task_id,status:"active",target_episodes:payload.target_episodes,operator:payload.operator,settings:payload,source_kind:"physical",updated_utc:"2026-10-11T00:00:00Z",counts:{saved:0,success:0,failure:0,excluded:0}};sessionList.push(session);value=session;}
    else if(route==="session/s1/open"){state.active_session=sessionList[0];state.capture={state:"idle",episode_index:0,pending_frame_count:0,min_episode_s:8,max_episode_s:25,remaining_s:25,countdown_s:null,controls:{start:true},counters:{saved:0,success:0,failure:0,discarded:0},recording_purpose:"qualifying"};value=state;}
    else if(route==="session/s1/close"){state.active_session=null;state.capture={state:"closed"};if(sessionList[0])sessionList[0].status="closed";value=state;}
    else if(route==="library/plan"){value=payload.sentence==="do the thing"&&!Object.keys(payload.answers||{}).length?{source:"built-in planner",breakdown:null,questions:[{key:"object",prompt:"Which object does the arm pick up?",example:"die"},{key:"container",prompt:"Which container does it go into, if any?",example:"cup"},{key:"destination",prompt:"Which marked destination does it go to, if there is no container?",example:"tray"}],recipe:"",fallback_reason:"",notes:[]}:{source:"built-in planner",breakdown:fakeBreakdown(payload.sentence),questions:[],recipe:"roll_a_die",fallback_reason:"",notes:[],warnings:[{step:2,code:"no_reset",message:"Step 2 is a reset step, check that it puts the scene back."}],proposed:{goal_id:"lib_roll_a_die",task_ids:["lib_roll_a_die_1","lib_roll_a_die_2"]}};}
    else if(route==="library/save"){libraryTasks.length=0;payload.breakdown.sub_tasks.forEach((step,i)=>libraryTasks.push({task_id:`lib_roll_a_die_${i+1}`,task_text:step.instruction,success_rule:step.success_rule,scene_requirements:step.scene_before_take,declared_fixed_scene:true,requires_unobserved_goal:false,episode_window:{min_episode_s:null,max_episode_s:null},task_source:"library",group:"Roll a die",display_name:`${i+1}. ${step.name}`,optional:false,goal_id:"lib_roll_a_die",goal_title:"Roll a die",library_version:1,chain_position:i+1,kind:step.kind,frozen:false,name:step.name,start_state:step.start_state,end_state:step.end_state,suggested_takes:step.suggested_takes}));value={goal:{id:"lib_roll_a_die",title:payload.breakdown.goal.title,tasks:libraryTasks.map(t=>({task_id:t.task_id,version:1}))},warnings:[]};}
    else if(route.endsWith("/time/start")){timing.running=true;timing.polls=0;timing.runs=[];timing.proposed_window=null;timing.phase="learning rest";value=timing;}
    else if(route.endsWith("/time/stop")){timing.running=false;value=timing;}
    else if(route.endsWith("/time/confirm")){timing.running=false;const t=libraryTasks[0];t.episode_window={min_episode_s:payload.min_episode_s,max_episode_s:payload.max_episode_s};value={...t,version:1,timing_runs:timing.runs,name:t.name};}
    else if(route==="session/s1/finish"){state.active_session=null;state.capture={state:"closed"};if(sessionList[0])sessionList[0].status="finished";value=state;}
    else if(route==="record/retry") {
      if(!allowSave){state.capture.error="save failed: disk full; pending frames retained";res.writeHead(409,{"Content-Type":"application/json"});res.end(JSON.stringify({error:state.capture.error}));return;}
      state.capture={state:"idle",pending_frame_count:0,controls:{start:true},last_episode:{episode_index:0,operator_label:"failure"}};
      if(sessionList[0])sessionList[0].counts={saved:1,success:0,failure:1,excluded:0};
      value=state;
    }
    else {res.writeHead(409,{"Content-Type":"application/json"});res.end(JSON.stringify({error:"Unexpected mutation in fake browser test"}));return;}
  } else if(route==="state")value=state;
  else if(route.startsWith("sessions"))value=sessionList;
  else if(route.startsWith("episodes/"))value=[];
  else if(route==="tasks")value=[{...task,task_source:"registry",group:"Built-in tasks",display_name:"Die into cup"},...libraryTasks];
  else if(route.endsWith("/time")){if(timing.running){timing.polls++;if(timing.polls>=2&&timing.runs.length<3){timing.runs.push({index:timing.runs.length+1,duration_s:10+timing.runs.length,peak_deg:40,peak_joint:"Joint1",gripper_travel_mm:16,gripper_cycled:true});timing.phase="waiting for a move";}if(timing.runs.length>=3)timing.proposed_window={min_episode_s:8,max_episode_s:16};}value={task_id:"lib_roll_a_die_1",...timing,runs_wanted:"3 to 5",rest_learned:true,error:null};}
  else if(route==="library")value={goals:[],rules:["Every sub-task starts and ends with the gripper empty and the arm at the ready pose.","One object grasped per sub-task."],planner:{external_configured:false,source:"built-in planner"}};
  else if(route==="readiness")value=expireReports&&report.checked_utc?{...report,status:"expired",label:"Check expired · run again",percent:0}:report;
  else if(route==="cameras")value={cameras:["fake-wrist-video-index0","fake-front-video-index0"],message:"Synthetic identifiers"};
  else if(route==="serial-connections")value={connections:[],message:"No physical connections in test"};
  else if(route==="software")value={status:"ready",label:"Ready",dependencies:{lerobot:true,cv2:true,rclpy:true,sensor_msgs:true,ffmpeg:true},message:"Fake dependency discovery"};
  else if(route==="operator-guide")value={environment:{host:"synthetic-test",python:"fake-python",ros_domain:"23",ros_distribution:"fake",middleware:"fake"},limits_verified:false,documents:{}};
  else if(route==="help/limits")value={path:"config/synria_limits.yaml",text:"Synthetic candidate ranges; verified_by and verified_on are empty."};
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
  let sequence=0, buffer="", sessionId, browserError="", transportError=null, closing=false;
  const pending=new Map();
  child.stderr.on("data",chunk=>{browserError=(browserError+chunk.toString()).slice(-8000);});
  /** Fail outstanding protocol calls on process or pipe failure while retaining startup evidence. */
  function transportFailed(error){
    transportError=Error(`Isolated browser transport failed: ${error.message}\n${browserError}`);
    for(const entry of pending.values()){clearTimeout(entry.timer);entry.reject(transportError);}
    pending.clear();
  }
  child.on("error",transportFailed);
  child.on("exit",(code,signal)=>{if(!closing)transportFailed(Error(`browser exited ${code ?? signal}`));});
  for(const stream of [child.stderr,child.stdio[3],child.stdio[4]])stream.on("error",transportFailed);
  child.stdio[4].on("data",chunk=>{
    buffer+=chunk.toString();let boundary;
    while((boundary=buffer.indexOf("\0"))>=0){const message=JSON.parse(buffer.slice(0,boundary));buffer=buffer.slice(boundary+1);
      if(message.id){const task=pending.get(message.id);if(task){pending.delete(message.id);clearTimeout(task.timer);message.error?task.reject(Error(JSON.stringify(message.error))):task.resolve(message.result);}}
      else if(message.method==="Runtime.exceptionThrown")exceptions.push(message.params.exceptionDetails);
    }
  });
  /** Send one bounded browser protocol request, keeping target commands scoped to this isolated page. */
  function send(method,params={},target=sessionId,timeoutMs=15000){return new Promise((resolve,reject)=>{
    if(transportError){reject(transportError);return;}
    const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error(`Protocol timeout: ${method}\n${browserError}`));},timeoutMs);
    pending.set(id,{resolve,reject,timer});child.stdio[3].write(JSON.stringify({id,method,params,...(target?{sessionId:target}:{})})+"\0");
  });}
  /** Evaluate test interactions and propagate browser exceptions rather than treating them as success. */
  async function evaluate(expression){const result=await send("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});if(result.exceptionDetails)throw Error(JSON.stringify(result.exceptionDetails));return result.result.value;}
  /** Wait on DOM mutations and hash changes, not arbitrary sleeps, for an asynchronous UI condition to become true. */
  /** Poll a page condition from here, so navigation, reload and loading pages cannot strand the wait. */
  async function until(expression,description=expression){
    const deadline=Date.now()+10000;
    while(Date.now()<deadline){
      let value=false;
      try{value=await evaluate(`(()=>{try{return Boolean(${expression});}catch(e){return false;}})()`);}catch(e){value=false;}
      if(value)return true;
      await new Promise(resolve=>setTimeout(resolve,120));
    }
    throw Error('UI condition timed out: '+description);
  }
  const click=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).click()`);
  const textOf=selector=>evaluate(`document.querySelector(${JSON.stringify(selector)}).textContent`);
  const reason=()=>textOf("#wizard-reason");
  let presses=0;
  /** Press the wizard's primary button and count it toward the Home-to-recording budget. */
  async function pressPrimary(){assert.equal(await evaluate("document.querySelector('#wizard-primary').disabled"),false,`primary must be enabled: ${await reason()}`);presses++;await click("#wizard-primary");}
  try {
    // Hosted browsers can initialize slowly; prove protocol readiness before creating a page.
    const startupDeadline=Date.now()+45000;
    await send("Browser.getVersion",{},null,45000);
    const target=await send("Target.createTarget",{url:"about:blank"},null,
      Math.max(1,startupDeadline-Date.now()));
    sessionId=(await send("Target.attachToTarget",{targetId:target.targetId,flatten:true},null)).sessionId;
    await send("Runtime.enable");await send("Page.enable");
    await send("Emulation.setDeviceMetricsOverride",{width:1360,height:1000,deviceScaleFactor:1,mobile:false});
    const url=`http://127.0.0.1:${server.address().port}/#token=synthetic-test-token`;
    await send("Page.navigate",{url});
    await until("window.synriaConsoleReady && document.querySelector('#environment-facts').textContent.includes('synthetic-test') && document.querySelector('[name=gripper_type]').value==='50mm'");
    assert.equal(await evaluate("location.hash"),"#/intro","the very first visit shows the intro once");
    assert.equal(await evaluate("document.activeElement.id"),"intro-title");
    assert.match(await textOf("#intro"),/Choose what to record.*Check cameras and arm.*Record takes/s);
    await click("#intro-continue");
    await until("location.hash==='#/home'");
    await send("Page.reload");
    await until("window.synriaConsoleReady && location.hash==='#/home'","after the intro is dismissed the page opens on Home");
    assert.equal(await evaluate("location.hash"),"#/home","no open session lands on Home with the token removed from the address");
    assert.equal(await evaluate("document.querySelector('#home').hidden"),false);
    assert.equal(await evaluate("document.querySelector('#global-nav').hidden"),false);
    assert.equal(await evaluate("document.querySelector('#flow-nav').hidden"),true);
    assert.equal(await evaluate("document.querySelectorAll('#global-nav a').length"),3,"three global destinations only");
    assert.equal(await evaluate("document.querySelector('#token-notice').hidden"),true,"a tab with the launch key shows no key notice");
    assert.match(await textOf("#status-strip"),/^Arm feedNot checked yetWrist cameraNot checked yetOverhead cameraNot checked yetStorageOK · 100\.0 GB free$/,"status strip reads in neutral words, software hidden when fine");
    assert.equal(await textOf("#status-note"),"Not checked yet. Checked automatically when you start a session.");
    assert.equal(await evaluate("document.activeElement.id"),"home-title","focus lands on the Home heading");
    assert.equal(await evaluate("document.querySelector('#home-title').tagName"),"H1");
    assert.equal(await evaluate("[...document.querySelectorAll('#home-cards .home-card h3')].map(n=>n.textContent).join('|')"),"New session|Practice run|Review recordings");
    assert.equal(await evaluate("document.querySelector('#home-cards .home-card:last-child').disabled"),true,"review is disabled until a take exists");
    assert.equal(await evaluate("document.querySelector('#first-run-strip').hidden"),false,"first run shows the three-step strip");
    await click("#dismiss-first-run");
    assert.equal(await evaluate("document.querySelector('#first-run-strip').hidden"),true);
    assert.equal(await evaluate("document.querySelector('#quit')"),null,"shut down lives on Help, not in the header");
    assert.match(await textOf("footer"),/^This console only records\. It never moves the arm\.$/);
    // Refresh restores the same place for every route.
    for(const path of ["sessions","help","new/task","session/none/review","home"]){
      await evaluate(`SynriaRouter.go(${JSON.stringify(path)})`);
      await until(`location.hash==='#/${path}'`);
      await send("Page.reload");
      await until("window.synriaConsoleReady");
      assert.equal(await evaluate("location.hash"),`#/${path}`,`refresh keeps ${path}`);
      assert.equal(await evaluate("document.querySelector('.screen:not([hidden])').id"),{sessions:"sessions",help:"help","new/task":"new","session/none/review":"review",home:"home"}[path],`refresh shows ${path}`);
    }
    await evaluate("SynriaRouter.go('new/start')");
    await until("location.hash==='#/new/task'","a later step cannot be reached by address before its earlier steps are complete");
    await evaluate("SynriaRouter.go('home')");await until("location.hash==='#/home'");
    assert.equal(calls.length,0,"navigation and refresh never post anything");
    // Inside a flow the global nav is replaced by a breadcrumb and one exit.
    presses++;await click("#home-session-link");
    await until("location.hash==='#/new/task' && document.querySelector('#global-nav').hidden && !document.querySelector('#flow-nav').hidden");
    assert.equal(await textOf("#breadcrumb"),"HomeNew session");
    assert.equal(await evaluate("document.activeElement.id"),"step-task-title","focus moves to the step heading");
    assert.equal(await textOf("#route-announcer"),"Step 1 of 2: Choose task");
    await click("#exit-flow");
    await until("location.hash==='#/home' && !document.querySelector('#global-nav').hidden");
    await evaluate("history.back()");
    await until("location.hash==='#/new/task' && !document.querySelector('#new').hidden","browser Back returns to the flow");
    // No two controls with the same label lead to different places.
    const labels=await evaluate("JSON.stringify([...document.querySelectorAll('[data-route]')].map(n=>[n.textContent.trim(),n.dataset.route]))");
    const targets=new Map();for(const [label,target] of JSON.parse(labels)){targets.set(label,(targets.get(label)||new Set()).add(target));}
    for(const [label,set] of targets)assert.equal(set.size,1,`label ${label} points at one place`);
    assert.equal(await evaluate("new Set([...document.querySelectorAll('[id]')].map(n=>n.id)).size===document.querySelectorAll('[id]').length"),true,"no duplicate controls");
    // Step 1: the primary waits for a task and says so.
    assert.equal(await evaluate("document.querySelector('#wizard-primary').disabled"),true);
    assert.equal(await reason(),"1 thing left: choose a task");
    assert.equal(await textOf("#wizard-back"),"Cancel","on the first step the footer left button cancels to Home");
    assert.equal(await evaluate("document.querySelectorAll('#new button').length===[...document.querySelectorAll('#new button')].length && [...document.querySelectorAll('#new button')].filter(n=>n.textContent.trim()==='Exit to Home').length"),0,"only the top-right Exit says Exit");
    assert.equal(await evaluate("document.querySelector('[data-wizard-step=check]').disabled"),true,"later steps cannot be clicked");
    await evaluate("SynriaGuide.openHelp('source')");
    assert.match(await textOf("#help-content"),/No verified startup command/);
    assert.match(await textOf("#help-content"),/joint_commands_enabled:=false/);
    assert.match(await evaluate("document.querySelector('#help-content pre').textContent"),/--ros-args \\\n/);
    await click("#close-help");
    assert.match(await textOf("[data-task-id=die_into_cup]"),/^Die into cup/,"sentence case title");
    assert.match(await textOf("[data-task-id=die_into_cup]"),/Practice only until timed/);
    await click("[data-task-id=die_into_cup]");
    assert.equal(await evaluate("document.querySelector('[name=task_id]').value"),"die_into_cup");
    assert.match(await textOf("#task-description"),/Scene before each take.*Fixed synthetic scene.*Take length not set yet\. You can do practice runs now\. To record takes that count, time this task first\./s);
    assert.ok(!(await textOf("#task-description")).includes("DECISIONS.md"),"no internal text in the task panel");
    assert.equal(await evaluate("document.querySelector('#task-session-fields').hidden"),false,"name and target appear under the chosen task");
    assert.match(await evaluate("document.querySelector('[name=name]').value"),/^Die into cup, \d{1,2} [A-Z][a-z]{2}, 1$/,"session name in words");
    assert.equal(await evaluate("document.querySelector('[name=target_episodes]').value"),"10","with no earlier session the target is the quick trial");
    assert.match(await evaluate("[...document.querySelector('[data-preset-for=target_episodes]').options].map(o=>o.textContent).join('|')"),/10 takes \(quick trial\)\|50 takes\|100 takes\|Other/,"target choices in plain words");
    await evaluate("[...document.querySelectorAll('#task-description button')].find(n=>n.textContent==='How to time this task').click()");
    assert.match(await textOf("#help-content"),/config\/synria_tasks.json/);
    assert.match(await textOf("#help-content"),/cannot set or override/);
    await click("#close-help");
    assert.match(await textOf("#help"),/There are no timing inputs here by design/,"timing registration lives on the Help page");
    // Teach something new: one sentence in, editable cards out, Save tasks as the only primary.
    await click("#task-cards .teach-entry");
    await until("location.hash==='#/teach' && !document.querySelector('#teach').hidden");
    assert.equal(await textOf("#breadcrumb"),"HomeTeach something new");
    assert.equal(await evaluate("document.querySelector('#teach-save').hidden"),true,"Save tasks is not shown before a breakdown");
    assert.equal(await evaluate("document.querySelector('#teach-break').classList.contains('primary')"),true,"Break it down is the primary until a breakdown exists");
    assert.equal(await evaluate("[...document.querySelectorAll('#teach-examples button')].map(n=>n.textContent).join('|')"),"roll a die|put the die in the cup|move the cup to the tray","example chips the built-in planner handles");
    assert.match(await textOf("#teach-rules"),/gripper empty/);
    await evaluate("document.querySelector('#teach-sentence').value='do the thing'");
    await click("#teach-break");
    await until("!document.querySelector('#teach-questions').hidden");
    assert.match(await textOf("#teach-notes"),/^The planner could not work out the object and where it goes from that sentence\. The built-in planner understands/);
    assert.equal(await evaluate("[...document.querySelectorAll('#teach-questions button')].filter(n=>n.classList.contains('primary')).map(n=>n.textContent).join('|')"),"Break it down with these answers","in the fallback the only primary is the answered breakdown");
    assert.equal(await evaluate("document.querySelector('#teach-break').classList.contains('primary')"),false);
    await evaluate("[...document.querySelectorAll('#teach-questions button')].find(n=>n.textContent==='Add steps by hand').click()");
    await until("document.querySelectorAll('#teach-steps [data-step]').length===1 && document.querySelector('#teach-steps [data-step]').dataset.mode==='edit'","one blank card opens in Edit");
    assert.match(await textOf("#teach-reason"),/^1 thing left: step 1 needs an instruction$/);
    assert.equal(await evaluate("document.querySelector('#teach-save').disabled"),true,"an empty instruction blocks saving");
    await evaluate("[...document.querySelectorAll('#teach-examples button')].find(n=>n.textContent==='roll a die').click()");
    assert.equal(await evaluate("document.querySelector('#teach-sentence').value"),"roll a die");
    await evaluate("document.querySelector('#teach-sentence').value='train to roll a dice'");
    await click("#teach-break");
    await until("document.querySelectorAll('#teach-steps [data-step]').length===2 && document.querySelector('#teach-steps [data-step]').dataset.mode==='view'","the planner's steps appear as compact cards");
    assert.equal(calls.at(-1).route,"library/plan");assert.equal(calls.at(-1).payload.sentence,"train to roll a dice");
    assert.equal(await textOf("#teach-source"),"Suggested by the built-in planner.");
    assert.equal(await evaluate("document.querySelector('#teach-goal-title').value"),"Roll a die");
    assert.equal(await textOf("#teach-chain"),"1 Die into cup, then 2 Die back to start (reset)");
    assert.match(await textOf("#teach-steps"),/Step 1.*Die into cup.*Starts with: .*Ends with: .*Step 2.*Die back to start/s);
    assert.equal(await evaluate("document.querySelector('#teach-steps [data-field=name]')"),null,"no form fields until Edit");
    assert.match(await textOf("#teach-steps [data-step=\"1\"]"),/Step 2 is a reset step, check that it puts the scene back/,"a warning sits on the card it concerns");
    assert.equal(await evaluate("document.querySelector('#teach-save').hidden"),false);
    assert.equal(await evaluate("document.querySelector('#teach-save').disabled"),false,"warnings do not block saving");
    assert.equal(await textOf("#teach-break"),"Break it down again");
    assert.equal(await evaluate("document.querySelector('#teach-break').classList.contains('primary')"),false,"one primary: Save tasks");
    await evaluate("[...document.querySelectorAll('#teach-steps [data-step=\"0\"] button')].find(n=>n.textContent==='Edit').click()");
    await until("document.querySelector('#teach-steps [data-step=\"0\"]').dataset.mode==='edit'");
    await evaluate("document.querySelector('#teach-steps [data-step=\"0\"] [data-field=name]').value='Die into the cup'");
    await evaluate("[...document.querySelectorAll('#teach-steps [data-step=\"0\"] button')].find(n=>n.textContent==='Move down').click()");
    await until("document.querySelector('#teach-steps [data-step=\"1\"]').dataset.mode==='edit'","the moved card stays in Edit");
    assert.equal(await evaluate("document.querySelector('#teach-steps [data-step=\"1\"] [data-field=name]').value"),"Die into the cup","edits survive a move");
    await evaluate("[...document.querySelectorAll('#teach-steps [data-step=\"1\"] button')].find(n=>n.textContent==='Move up').click()");
    await until("document.querySelector('#teach-steps [data-step=\"0\"]').dataset.mode==='edit'");
    await evaluate("[...document.querySelectorAll('#teach-steps [data-step=\"0\"] button')].find(n=>n.textContent==='Done').click()");
    await until("document.querySelector('#teach-steps [data-step=\"0\"]').dataset.mode==='view'");
    assert.equal(await textOf("#teach-chain"),"1 Die into the cup, then 2 Die back to start (reset)");
    const savesBefore=calls.length;
    await click("#teach-save");
    await until("location.hash==='#/new/task'","Save tasks goes forward to Choose task");
    assert.equal(calls[savesBefore].route,"library/save");
    assert.equal(calls[savesBefore].payload.breakdown.sub_tasks[0].name,"Die into the cup","the edited card is what is saved");
    assert.equal(calls[savesBefore].payload.planner_source,"built-in planner");
    await until("document.querySelector('[data-task-id=lib_roll_a_die_1]')!==null","saved tasks appear as cards under their goal");
    assert.match(await textOf("#task-cards"),/Built-in tasks.*Die into cup.*Roll a die.*1\. Die into the cup.*Your task\. Takes are saved but do not count toward dataset milestones\..*2\. Die back to start/s);
    assert.equal(await evaluate("[...document.querySelectorAll('#task optgroup')].map(g=>g.label).join('|')"),"Built-in tasks|Roll a die","the dropdown is grouped by goal with built-in tasks first");
    assert.equal(await evaluate("document.querySelector('[name=task_id]').value"),"lib_roll_a_die_1","the first saved task is selected");
    assert.match(await textOf("#notice"),/2 tasks saved under Roll a die\./);
    // Time it: listen to the arm feed, three moves, a proposed window, one confirmation.
    assert.match(await textOf("#task-description"),/Your task from the goal "Roll a die", step 1, version 1/);
    await evaluate("[...document.querySelectorAll('#task-description button')].find(n=>n.textContent==='Time it').click()");
    await until("document.querySelector('#time-dialog').open");
    assert.equal(await textOf("#time-title"),"Time: 1. Die into the cup");
    assert.equal(await evaluate("document.querySelector('[name=task_id]').value"),"lib_roll_a_die_1");
    await click("#time-start");
    await until("document.querySelectorAll('#time-runs li').length===3 && !document.querySelector('#time-confirm').hidden","three timed moves and a proposed window");
    assert.match(await textOf("#time-runs"),/Move 1: 10\.0 s, gripper closed and reopened, peak 40 degrees on Joint1/);
    assert.equal(await evaluate("document.querySelector('#time-min').value+'-'+document.querySelector('#time-max').value"),"8-16");
    await evaluate("document.querySelector('#time-max').value='15';document.querySelector('#time-max').dispatchEvent(new Event('input',{bubbles:true}))");
    await click("#time-confirm");
    await until("!document.querySelector('#time-dialog').open");
    assert.equal(calls.at(-1).route,"library/lib_roll_a_die_1/time/confirm");
    assert.equal(JSON.stringify(calls.at(-1).payload),JSON.stringify({min_episode_s:8,max_episode_s:15}),"the operator's edit is what is confirmed");
    await until("document.querySelector('[data-task-id=lib_roll_a_die_1]').textContent.includes('8 to 15 s per take')","the card shows the confirmed take length");
    assert.match(await textOf("#notice"),/8 to 15 s per take, stored with version 1/);
    await click("[data-task-id=die_into_cup]");
    assert.equal(await evaluate("document.querySelector('[name=task_id]').value"),"die_into_cup");
    const teachPosts=calls.length;
    await pressPrimary();
    await until("location.hash==='#/new/check'");
    assert.equal(await textOf("#route-announcer"),"Step 2 of 2: Check setup");
    assert.match(await textOf("#setup-line"),/^Die into cup, 50 mm gripper, ros2_control arm feed, wrist fake-wrist, overhead fake-front, 15 FPS, you: test\./,"the task leads the setup line");
    assert.equal(await evaluate("document.querySelector('#setup-configure').hidden"),true,"remembered values sit behind Change setup");
    assert.equal(await evaluate("document.querySelector('#setup-inline [name=follower_serial]')!==null && document.querySelector('#setup-inline [name=power_choice]')!==null"),true,"serial and power state are inline");
    assert.equal(await evaluate("document.activeElement.name"),"follower_serial","focus lands on the first missing inline field");
    assert.equal(await reason(),"3 things left: choose the follower serial, confirm the power state, tick the bench checks");
    assert.equal(await textOf("#wizard-primary"),"Start practice run","an untimed task leads to the practice run, and the button says so");
    assert.equal(await evaluate("document.querySelector('#smoke').hidden"),true,"no second practice button when the primary is the practice run");
    assert.equal(await textOf("#check-readiness"),"Check now","before any run the button says Check now");
    await evaluate("document.querySelector('#scene-note').value='left-handed today';document.querySelector('#scene-note').dispatchEvent(new Event('input',{bubbles:true}))");
    await send("Page.reload");
    await until("window.synriaConsoleReady && location.hash==='#/new/check' && !document.querySelector('#step-check').hidden","a refresh on Check setup stays on Check setup");
    assert.equal(await evaluate("document.querySelector('[name=task_id]').value"),"die_into_cup","the chosen task survives a refresh");
    assert.equal(await evaluate("document.querySelector('#scene-note').value"),"left-handed today","the scene note survives a refresh");
    assert.match(await evaluate("document.querySelector('[name=name]').value"),/^Die into cup, \d{1,2} [A-Z][a-z]{2}, 1$/);
    assert.equal(await evaluate("document.querySelector('#bench-panel').hidden"),false,"physical mode shows the bench checks");
    assert.equal(await evaluate("[...document.querySelectorAll('#source-tiles .tile')].map(n=>n.dataset.state).join()"),"unchecked,unchecked,unchecked,ok","tiles say Not checked yet, storage is OK");
    assert.equal(calls.length,teachPosts,"entering Check setup without bench checks cannot open sources");
    // An invalid hidden field names itself and opens the form.
    await evaluate("const fpsPreset=document.querySelector('[data-preset-for=fps]');fpsPreset.value='custom';fpsPreset.dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[name=fps]').value='0';document.querySelector('[name=fps]').dispatchEvent(new Event('input',{bubbles:true}))");
    assert.match(await reason(),/choose the rate/);
    await click("#check-readiness");
    assert.equal(calls.length,teachPosts,"invalid native form constraints cannot open sources");
    assert.equal(await evaluate("document.querySelector('#setup-configure').hidden"),false,"invalid hidden form field opens its setup fields");
    await evaluate("document.querySelector('[data-preset-for=fps]').value='15';document.querySelector('[data-preset-for=fps]').dispatchEvent(new Event('change',{bubbles:true}))");
    if(process.env.SYNRIA_GUIDE_EXTRA_SHOTS)fs.writeFileSync(screenshot.replace(/\.png$/,"-configuration.png"),Buffer.from((await send("Page.captureScreenshot",{format:"png"})).data,"base64"));
    await click("#done-setup");
    assert.equal(await evaluate("document.querySelector('#setup-configure').hidden"),true);
    await evaluate("document.querySelector('#scene-note').value='<script>throw Error(99)</script>';document.querySelector('#scene-note').dispatchEvent(new Event('input',{bubbles:true}))");
    assert.equal(await evaluate("SynriaWizard.sceneText()"),"Fixed synthetic scene. Today: <script>throw Error(99)</script>","scene is the task scene plus today's note");
    await click("#check-readiness");
    await until("document.querySelector('#notice').textContent.includes('physical safety confirmations')");
    assert.equal(calls.length,teachPosts,"missing physical confirmations cannot issue a diagnostic");
    // The fifth bench tick starts the read-only check by itself.
    await evaluate("document.querySelectorAll('[data-readiness-confirm]').forEach(n=>{n.checked=true;n.dispatchEvent(new Event('change',{bubbles:true}));})");
    await until("document.querySelector('#wizard-reason').textContent.includes('fix the arm feed')");
    assert.equal(await reason(),"3 things left: choose the follower serial, confirm the power state, fix the arm feed");
    assert.equal(await textOf("#check-readiness"),"Check again");
    assert.equal(await textOf("#notice"),"","a deliberate valid attempt supersedes the resolved consent error");
    assert.equal(calls.length,teachPosts+1);assert.equal(calls[teachPosts].payload.settings.follower_serial,"");
    assert.equal(calls[teachPosts].payload.settings.scene,"Fixed synthetic scene. Today: <script>throw Error(99)</script>");
    assert.equal("power_choice" in calls[teachPosts].payload.settings,false,"page-only controls never reach the server");
    assert.equal(await evaluate("document.querySelector('#wizard-primary').disabled"),true,"a failing check keeps the primary disabled");
    assert.equal(await evaluate("location.hash"),"#/new/check","a failing check never changes the step");
    assert.match(await textOf("#source-tiles"),/Problem.*Arm feed.*What is wrong.*No state sample received.*Likely cause.*What to do/s,"a problem tile explains itself");
    assert.match(await textOf("#source-tiles"),/Not checked yet.*Wrist camera/s);
    assert.match(await textOf("#sources-line"),/Last check just now/);
    assert.ok(!await evaluate("document.querySelector('#all-check-details').textContent.includes('Ready · state source')"));
    assert.equal(await evaluate("document.querySelector('#technical-details').open"),false,"technical detail is collapsed");
    await evaluate("[...document.querySelectorAll('#source-tiles button')].find(n=>n.textContent==='All the steps').click()");
    assert.match(await textOf("#help-content"),/ROS domain/);
    await click("#close-help");
    assert.equal(await evaluate("document.querySelector('[data-wizard-step=task]').disabled"),false,"a completed step can be revisited");
    await click("[data-wizard-step=task]");
    await until("location.hash==='#/new/task'");
    await click("[data-wizard-step=check]");
    await until("location.hash==='#/new/check'");
    assert.equal(await evaluate("document.querySelector('#scene-note').value"),"<script>throw Error(99)</script>","Back and forward keep inputs");
    if(process.env.SYNRIA_GUIDE_EXTRA_SHOTS)fs.writeFileSync(screenshot.replace(/\.png$/,"-failure.png"),Buffer.from((await send("Page.captureScreenshot",{format:"png"})).data,"base64"));
    scenario="camera";await click("#check-readiness");
    await until("document.querySelector('#wizard-reason').textContent.includes('fix the wrist camera')");
    assert.match(await textOf("#source-tiles"),/Problem.*Wrist camera.*Camera device busy/s);
    await evaluate("[...document.querySelectorAll('#source-tiles button')].find(n=>n.textContent==='All the steps').click()");
    assert.match(await textOf("#help-content"),/pgrep -a cheese/);
    await evaluate("document.querySelector('#close-help').click();document.querySelector('[name=fps]').value='30';document.querySelector('[name=fps]').dispatchEvent(new Event('change',{bubbles:true}));");
    assert.equal(await evaluate("[...document.querySelectorAll('#source-tiles .tile')].slice(0,3).every(n=>n.dataset.state==='unchecked')"),true,"changed settings make every tile Not checked yet");
    assert.ok(!(await reason()).includes("run the check"),"running the check is never left to the operator");
    scenario="recovery";
    await until("document.querySelector('#wizard-reason').textContent.includes('preserve the dataset for recovery')","after a setting changes the check runs again by itself");
    assert.match(await textOf("#source-tiles"),/journal requires review/);
    await evaluate("[...document.querySelectorAll('#source-tiles button')].find(n=>n.textContent==='All the steps').click()");
    assert.match(await textOf("#help-content"),/Do not remove the lock/);
    await click("#close-help");
    assert.equal(await evaluate("document.querySelector('#wizard-primary').disabled"),true,"missing identity cannot start a practice run");
    // Expiry never moves the operator: the failed check stays on this step, with the same reason.
    expireReports=true;
    await until("document.querySelector('#readiness-label').textContent.includes('expired')");
    assert.equal(await evaluate("location.hash"),"#/new/check");
    assert.match(await reason(),/preserve the dataset for recovery/);
    expireReports=false;
    // Back returns exactly one step with inputs intact.
    await evaluate("document.querySelector('[name=notes]').value='keep me';document.querySelector('[name=notes]').dispatchEvent(new Event('input',{bubbles:true}))");
    await click("#wizard-back");await until("location.hash==='#/new/task'");
    await pressPrimary();await until("location.hash==='#/new/check'");presses--;
    assert.equal(await evaluate("document.querySelector('[name=notes]').value"),"keep me");
    // Re-entering the step with an expired check reruns it by itself; wait for that run to finish.
    await until("!SynriaWizard.flags().checking && !readinessBusy && !document.querySelector('#check-readiness').disabled","the automatic rerun has finished");
    // Serial and power are one tap each, inline; filling them reruns the check by itself.
    await evaluate("const serial=document.querySelector('#setup-inline [name=follower_serial]');serial.value='__manual__';serial.dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('#follower-serial-custom').value='ARM-TEST';document.querySelector('#follower-serial-custom').dispatchEvent(new Event('input',{bubbles:true}))");
    await evaluate("document.querySelector('#setup-inline [name=power_choice][value=\"Arm powered and following the leader\"]').click()");
    assert.equal(await evaluate("formValues().power_state_start"),"Arm powered and following the leader");
    assert.equal(await evaluate("formValues().follower_serial"),"ARM-TEST");
    scenario="pass";
    await until("!document.querySelector('#wizard-reason').textContent.includes('preserve the dataset') && !SynriaWizard.flags().checking && !readinessBusy && SynriaWizard.tileRows().every(t=>t.state==='ok')","the check reruns by itself and passes");
    assert.equal(await reason(),"This task is not timed yet. A practice run records one 20 second take and throws it away.");
    assert.match(await textOf("#source-tiles"),/OK.*Arm feed.*Receiving 25 readings per second/s,"arm feed states its rate in words");
    assert.equal(await evaluate("document.querySelector('#wizard-primary').disabled"),false,"identity supplied lets the practice run start");
    assert.equal(await textOf("#wizard-primary"),"Start practice run");
    const checksSoFar=calls.filter(call=>call.route==="readiness").length;
    task.episode_window={min_episode_s:8,max_episode_s:25};
    await evaluate("tasks[0].episode_window={min_episode_s:8,max_episode_s:25};SynriaWizard.refresh()");
    assert.equal(await reason(),"Takes will not count toward the dataset until the joint limits are verified.");
    assert.equal(await textOf("#wizard-primary"),"Start practice run");
    state.demo=true;await until("document.body.classList.contains('demo')");
    await evaluate("SynriaWizard.refresh()");
    assert.equal(await evaluate("document.querySelector('#wizard-primary').disabled"),false,"synthetic demo needs no physical evidence");
    assert.equal(await textOf("#wizard-primary"),"Start session");
    assert.equal(await reason(),"Demo takes never count toward the dataset.");
    assert.equal(await textOf("#mode-help"),"Demo mode: fake cameras and arm. Nothing recorded here is real.");
    expireReports=true;await until("document.querySelector('#readiness-label').textContent.includes('expired')");
    assert.equal(await evaluate("location.hash"),"#/new/check","expiry never sends the operator back");
    await pressPrimary();
    expireReports=false;
    await until("location.hash==='#/session/s1/record'","Start session reruns the stale check in place, opens the session and lands on recording");
    assert.equal(calls.filter(call=>call.route==="readiness").length,checksSoFar+1,"a stale check is rerun exactly once before starting");
    assert.deepEqual(calls.slice(-2).map(call=>call.route),["sessions","session/s1/open"]);
    assert.match(calls.at(-2).payload.name,/^Die into cup, \d{1,2} [A-Z][a-z]{2}, 1$/);
    assert.ok(presses<=4,`Home to recording took ${presses} presses`);
    await until("document.querySelector('#record-title').textContent==='Take 1 of 10'","the heading counts takes; the breadcrumb names the session");
    assert.equal(await textOf("#record-move"),"The move: Synthetic fixture: put die into cup","the move is named under the take heading");
    assert.equal(await textOf("#breadcrumb"),`Home${calls.at(-2).payload.name}`);
    assert.equal(await evaluate("document.querySelector('#finish-session').classList.contains('primary')"),false,"Finish is neutral; the primary is Start recording");
    assert.equal(await evaluate("sessionStorage.getItem('synria-wizard-draft')"),null,"a started session clears the per-tab draft");
    state.demo=false;
    // Recording keyboard guards and save retries, unchanged from the earlier screen.
    state.capture={state:"stopped",pending_frame_count:2,controls:{success:true,failure:true,discard:true}};
    await evaluate("refreshState()");
    await evaluate("SynriaGuide.openHelp('save')");
    assert.equal(await evaluate("lastCapture"),"stopped");
    assert.equal(await textOf("#record-phase"),"DID IT WORK?","a stopped take asks whether it worked");
    assert.equal(await evaluate("[...document.querySelectorAll('#record-controls button')].filter(n=>!n.hidden).map(n=>n.textContent).join('|')"),"Yes, save as success (S)|No, save as failure (F)|Throw away, bad capture","only the actions that apply are shown");
    assert.equal(await evaluate("document.querySelector('[data-command=start]').hidden"),true,"Start recording is hidden while a take waits for its label");
    assert.equal(await evaluate("document.querySelector('#pause-session').hidden"),false,"Pause and Finish replace Exit while a session is open");
    assert.equal(await evaluate("document.querySelector('#exit-flow').hidden"),true);
    await click("#pause-session");
    await until("document.querySelector('#notice').textContent.includes('Save or throw away the stopped take first')");
    assert.equal(await evaluate("location.hash"),"#/session/s1/record","leaving with an unsaved take is refused");
    await evaluate("SynriaGuide.renderError('')");
    const protectedCount=calls.length;
    await evaluate("document.dispatchEvent(new KeyboardEvent('keydown',{code:'Space',key:' '}));document.dispatchEvent(new KeyboardEvent('keydown',{key:'s'}))");
    assert.equal(calls.length,protectedCount,"help-dialog keyboard use cannot label or start a recording");
    await evaluate("document.querySelector('#close-help').click();document.activeElement.blur();document.dispatchEvent(new KeyboardEvent('keydown',{key:'s',ctrlKey:true}))");
    assert.equal(calls.length,protectedCount,"browser shortcuts cannot label an episode");
    state.capture.error="Previous save failed; pending frames retained";
    state.capture.controls.retry=true;
    await evaluate("refreshState()");
    await click("[data-command=retry]");
    await until("document.querySelector('#notice').textContent.includes('disk full')");
    assert.equal(state.capture.pending_frame_count,2,"failed save retains pending state");
    allowSave=true;
    await click("[data-command=retry]");
    await until("document.querySelector('#notice').textContent==='' && state.capture.pending_frame_count===0");
    await until("document.querySelector('#record-message').textContent==='Take 1 saved as failure. Quality checks not run yet'","the saved message counts takes from 1");
    assert.equal(await evaluate("[...document.querySelectorAll('#record-controls button')].filter(n=>!n.hidden).map(n=>n.textContent).join('|')"),"Start recording (Space)");
    await until("document.querySelector('#record-message').textContent==='Set the scene, then press Start recording.'","after two seconds the banner returns to Ready");
    assert.equal(await evaluate("document.querySelector('#record-saved').hidden"),false,"the last result stays under Ready");
    assert.equal(await textOf("#record-saved"),"Take 1 saved as failure. Quality checks not run yet");
    assert.equal(await evaluate("document.querySelector('#reset-scene').hidden"),false,"the scene reset list shows between takes");
    assert.match(await textOf("#reset-scene-list"),/Fixed synthetic scene/);
    await click("#review-latest");
    await until("document.querySelector('#review-overlay').open && document.querySelector('#review-overlay #review')!==null","review opens as an overlay");
    await click("#close-review-overlay");
    assert.equal(await evaluate("document.querySelector('#review-overlay').open"),false);
    assert.equal(await evaluate("location.hash"),"#/session/s1/record","closing the overlay returns to the same recording state");
    assert.equal(await evaluate("document.querySelector('#record').hidden"),false);
    const finishAt=calls.length;
    await click("#finish-session");
    await until("!document.querySelector('#finish-confirm').hidden","finishing below target asks once, inline");
    assert.match(await textOf("#finish-confirm-text"),/^Finish with 1 of 10 takes\? You can reopen it later\.$/);
    assert.equal(await evaluate("document.querySelector('#pause-session').hidden && document.querySelector('#finish-session').hidden"),true,"only Finish and Keep recording while the confirm shows");
    await click("#finish-no");
    assert.equal(await evaluate("document.querySelector('#finish-confirm').hidden"),true);
    await click("#finish-session");await until("!document.querySelector('#finish-confirm').hidden");await click("#finish-yes");
    await until("location.hash==='#/session/s1/summary'","Finish marks the session finished and shows the summary");
    assert.equal(calls[finishAt].route,"session/s1/finish");
    assert.match(await textOf("#summary-list"),/TaskDie into cup.*Takes saved1 of a target of 10/s);
    assert.equal(await evaluate("[...document.querySelectorAll('#summary-actions button, #summary-actions a')].map(n=>n.textContent).join('|')"),"Record another session of this task|Review takes","one primary and one secondary on the summary");
    await evaluate("SynriaRouter.go('session/s1/record')");
    await until("!document.querySelector('#record').hidden");
    assert.equal(calls.filter(call=>call.route==='record/retry').length,2,"only two deliberate save retries occur");
    state.active_session=null;state.capture={state:"closed"};
    const mutations=calls.length;offline=true;
    await evaluate("refreshState().catch(error)");
    assert.match(await textOf("#notice"),/Reconnect without losing/);
    offline=false;await evaluate("refreshState()");
    assert.match(await textOf("#notice"),/Network connection lost/,"polling cannot erase an actionable error");
    assert.equal(calls.length,mutations,"reconnection never retries a mutation");
    await send("Page.reload");
    await until("window.synriaConsoleReady && document.querySelector('#environment-facts').textContent.includes('synthetic-test')");
    assert.equal(calls.length,mutations,"reload never restarts a diagnostic");
    assert.equal(await evaluate("location.hash"),"#/session/s1/record","refresh keeps the recording address");
    assert.equal(await evaluate("document.querySelector('#record-route-note').hidden"),false,"a closed session says so instead of pretending to record");
    await evaluate("SynriaRouter.go('sessions')");
    await until("!document.querySelector('#sessions').hidden");
    assert.match(await textOf("#session-list"),/Die into cup · Finished · 1 of 10 takes saved.*Reopen to add takes.*Review takes/s,"a finished session reads Finished and can be reopened");
    assert.equal(await evaluate("document.querySelector('#home-cards')?null:0"),null);
    await evaluate("SynriaRouter.go('home')");await until("!document.querySelector('#home').hidden");
    assert.equal(await evaluate("document.querySelector('#home-cards .home-card.primary-card h3').textContent"),"New session","after a finished session New session is the highlight again");
    assert.equal(await evaluate("document.querySelector('#sessions-more').open"),false,"trash and reindex sit behind More");
    await evaluate("SynriaRouter.go('new/task')");
    await until("!document.querySelector('#new').hidden");
    await send("Emulation.setDeviceMetricsOverride",{width:390,height:844,deviceScaleFactor:1,mobile:true});
    assert.equal(await evaluate("document.documentElement.scrollWidth<=window.innerWidth+2"),true,"mobile layout must not overflow");
    await send("Emulation.setDeviceMetricsOverride",{width:1360,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate("document.querySelector('#mode').textContent='SYNTHETIC UI TEST · NO HARDWARE';document.body.classList.add('demo')");
    fs.writeFileSync(screenshot,Buffer.from((await send("Page.captureScreenshot",{format:"png"})).data,"base64"));
    assert.deepEqual(exceptions,[],"no uncaught browser errors");
    console.log("Guided browser: routes, refresh, breadcrumb, labels, wizard steps, inline fields, tiles, footer reasons, bench checks, state timeout, camera fault, recovery, stale settings, expiry, Back keeps inputs, start reruns a stale check, keyboard, save retries, network, reload and mobile checks passed.");
  } finally {
    closing=true;
    try{await send("Browser.close",{},null);}catch{}
    child.kill("SIGTERM");
    for(const entry of pending.values())clearTimeout(entry.timer);
    server.closeAllConnections();await new Promise(resolve=>server.close(resolve));
    fs.rmSync(profile,{recursive:true,force:true});
  }
}
run().catch(error=>{console.error(error);if(exceptions.length)console.error("Browser exceptions:",JSON.stringify(exceptions.map(e=>e.exception?.description||e.text),null,1));server.closeAllConnections();server.close();process.exitCode=1;});
