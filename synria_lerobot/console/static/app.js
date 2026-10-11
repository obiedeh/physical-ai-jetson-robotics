"use strict";
// The page is a view of server-owned capture state, never its clock or owner.
const $ = id => document.getElementById(id);
const fragment = new URLSearchParams(location.hash.slice(1));
if (fragment.has("token")) {
  // The launch URL carries the token once; the route hash replaces it so refresh keeps the place.
  try { sessionStorage.setItem("console-token", fragment.get("token")); } catch (reason) { /* storage may be blocked; the page then shows the launch-key notice */ }
  history.replaceState(null, "", location.pathname);
}
let state = {}, sessions = [], tasks = [], selected = null, frame = 0, playing = false;
let rows = [], live = [], playGeneration = 0, frameGeneration = 0, lastCapture = "", lastCue = "", audioContext;
let recordView = {reasons:{}, counters:"", lastSave:""};
let reviewSessionId = "", route = null;
/** Read the per-launch token without failing when the browser blocks session storage. */
function storedToken() { try { return sessionStorage.getItem("console-token") || ""; } catch (reason) { return ""; } }
const token = storedToken();
/** Send same-origin JSON requests through the existing authenticated API boundary. */
async function api(path, payload) {
  const options = payload === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json", "X-Console-Token":token}, body:JSON.stringify(payload)};
  const response = await fetch(`/api/${path}`, options);
  const result = await response.json();
  if (!response.ok) throw Error(result.error || "Request refused");
  return result;
}
/** Show an operator-facing refusal without interpreting it as a successful check. */
function error(value) { if(window.SynriaGuide){window.SynriaGuide.renderError(value);return;} $("notice").textContent = value ? String(value.message || value) : ""; }
/** Surface asynchronous UI failures while leaving server-owned state intact. */
async function safe(operation) { try { return await operation(); } catch (reason) { error(reason); } }
/** Route helpers shared by every screen so no screen assembles a session address by hand. */
const routes = () => window.SynriaRouter.routes;
/** Navigate by screen name; kept for callers that predate hash routes. Never changes capture ownership. */
function screen(id) {
  const r = routes();
  const target = {home:r.home(), sessions:r.sessions(), help:r.help(), new:r.wizard("task"),
    record:r.record(state.active_session?.id ?? "none"), review:r.review(reviewSessionId || state.active_session?.id || "none")}[id];
  window.SynriaRouter.go(target || r.home());
}
/** Show exactly one screen section; routing decides which, this only toggles visibility. */
function showSection(id) { document.querySelectorAll(".screen").forEach(node => {node.hidden = node.id !== id;}); }
/** Build a labeled action control with shared refusal handling. */
function button(label, action, kind = "") { const node = document.createElement("button"); node.textContent = label; node.className = kind; node.onclick = () => safe(action); return node; }
/** Create text-only content so operator data cannot become executable markup. */
function text(tag, value) { const node = document.createElement(tag); node.textContent = value; return node; }
/** Create a same-page link to a route path, so navigation controls are real anchors. */
function routeLink(label, path, kind = "") { const node = text("a", label); node.className = `button ${kind}`.trim(); node.href = window.SynriaRouter.href(path); node.dataset.route = path; return node; }
/** Keep synthetic session identity visible wherever a session is displayed. */
function demoLabel(session) { return session.source_kind === "synthetic_demo" ? " · SYNTHETIC DEMO" : ""; }
/** Emit an optional local audio cue without controlling the recorder clock. */
function tone() {
  if (!$("audio").checked) return;
  audioContext ||= new AudioContext();
  const oscillator = audioContext.createOscillator(), gain = audioContext.createGain();
  oscillator.connect(gain); gain.connect(audioContext.destination); gain.gain.value = 0.06;
  oscillator.frequency.value = 660; oscillator.start(); oscillator.stop(audioContext.currentTime + 0.14);
}
/** Draw recorded state and action values without sending commands to hardware. */
function trace(canvas, data, cursor = -1) {
  const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h); const colors = ["#74d7ca","#edbe70","#bfa0fa","#6ab9fa","#ff9b98","#b9d875","#f4a8d0"];
  for (let j=0;j<7;j++) for (const key of ["state","action"]) {
    ctx.strokeStyle = colors[j]; ctx.setLineDash(key === "action" ? [5,4] : []); ctx.beginPath();
    data.forEach((row,i) => {const v=row[key]?.[j]; if(v === undefined)return; const x=i/Math.max(1,data.length-1)*w, y=h/2-v*(j===6?1000:25); i?ctx.lineTo(x,y):ctx.moveTo(x,y);}); ctx.stroke();
  }
  ctx.setLineDash([]); if(cursor>=0){ctx.strokeStyle="#fff";const x=cursor/Math.max(1,data.length-1)*w;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
}
/** Rebuild the session catalog view without editing authoritative dataset facts. */
async function refreshSessions() {
  sessions = await api(`sessions?search=${encodeURIComponent($("search").value)}`);
  $("session-list").replaceChildren(); $("trash-list").replaceChildren();
  const r = routes();
  for (const session of sessions) {
    const card = text("article", ""); card.className = "card";
    const open = state.active_session?.id === session.id;
    card.append(text("h3", session.name + demoLabel(session) + (open ? " · open now" : "")));
    card.append(text("p", `${session.task_id} · ${session.status} · target ${session.target_episodes}`));
    const counts = session.counts || session;
    card.append(text("p", `Saved ${counts.saved ?? counts.episode_count ?? 0} · success ${counts.success ?? counts.success_count ?? 0} · failure ${counts.failure ?? counts.failure_count ?? 0} · excluded ${counts.excluded ?? counts.excluded_count ?? 0}`));
    card.append(text("p", `${session.updated_utc || ""} · ${((session.disk_size_bytes || session.size_bytes || 0)/1e6).toFixed(1)} MB`));
    if(session.recovery_blocked)card.append(text("p",`RECOVERY BLOCKED: ${session.recovery_error}. Preserve the recording lock and journal; resolve through operator review before further writes.`));
    const actions = text("div", ""); actions.className = "actions";
    if (["trashed","purged"].includes(session.status)) {
      if(session.status === "trashed") {
        actions.append(button("Restore", async()=>{await api(`session/${session.id}/restore`,{});await refreshSessions();}));
        actions.append(button("Purge…",()=>deleteSession(session,"purge"),"danger"));
      }
      card.append(actions); $("trash-list").append(card);
    } else {
      if (open) actions.append(routeLink("Continue recording", r.record(session.id), "primary"));
      else actions.append(button("Open session",async()=>{await api(`session/${session.id}/open`,{});await refreshState();window.SynriaRouter.go(r.record(session.id));},"primary"));
      actions.append(routeLink("Review takes", r.review(session.id)));
      actions.append(button("Rename",async()=>{const name=prompt("New session name",session.name);if(name){await api(`session/${session.id}/rename`,{name});await refreshSessions();}}));
      actions.append(button("Delete…",()=>deleteSession(session,"trash"),"danger"));
      card.append(actions); $("session-list").append(card);
    }
    if(session.recovery_blocked)actions.querySelectorAll("button").forEach(node=>{node.disabled=true;});
  }
  if (!sessions.length) $("session-list").append(text("p", "No sessions yet. Start with New session."));
  window.dispatchEvent(new Event("console-sessions"));
}
/** Require typed identity and a reason before requesting an audited deletion. */
async function deleteSession(session, action) {
  const warning = session.evidence_warning || "If this session is cited in committed evidence, deletion needs a separate correction entry. The console never edits that evidence.";
  const confirmation=prompt(`${warning}\n${action === "purge" ? "PERMANENTLY PURGE" : "Move to trash"}: type the exact session name`,"");
  if(confirmation===null)return; const reason=prompt("Reason (required)","");if(reason===null)return;
  await api(`session/${session.id}/${action}`,{confirmation,reason}); await refreshSessions();
}
/** Serialize named form controls to the recorder configuration without changing units. */
function formValues() {
  const data=Object.fromEntries(new FormData($("new-form")));
  if (data.follower_serial === "__manual__") {
    data.follower_serial = $("follower-serial-custom").value.trim();
  }
  for(const key of ["fps","action_lookahead_steps","image_width","image_height","state_startup_timeout_s","target_episodes"])data[key]=Number(data[key]);
  data.state_has_velocity=$("new-form").elements.state_has_velocity.checked;
  data.guard_command_topic=data.guard_command_topic.split(",").map(s=>s.trim()).filter(Boolean);
  return data;
}
/** Display the immutable registry task and keep unset physical windows blocked. */
function taskDescription() {
  const task=tasks.find(t=>t.task_id===$("task").value);if(!task)return;
  const unset=task.episode_window.min_episode_s===null;
  $("task-description").replaceChildren(text("h3",task.task_text),text("p",`Success: ${task.success_rule}`),text("p",task.scene_requirements.join(" · ")),text("p",unset?"Real task window is unset. Qualifying recording is refused until the operator times this task and records the prospective window in DECISIONS.md. Only disposable physical smoke is available.":`Episode window: ${task.episode_window.min_episode_s}–${task.episode_window.max_episode_s} seconds.`));
  $("create").disabled=unset&&!state.demo;
  if(state.demo)$("task-description").append(text("p","Synthetic demo uses a separately marked non-qualifying window, never this physical task window."));
}
/** Show every registry window and make unset qualification decisions visible before task selection. */
function renderTaskWindows() {
  const container = $("task-window-list");
  container.replaceChildren();
  for (const task of tasks) {
    const window = task.episode_window;
    const registered = window.min_episode_s !== null && window.max_episode_s !== null;
    container.append(text("p", `${task.task_id}: ${registered
      ? `${window.min_episode_s}–${window.max_episode_s} seconds · registered`
      : "Not registered · qualifying recording unavailable"}`));
  }
}
/** Show one plain-language state banner with the single next action; never a recorder change. */
function renderRecordBanner(c, active) {
  recordView=window.SynriaRecord.describe(c, active);
  $("record-phase").textContent=recordView.phase.toUpperCase();$("record-message").textContent=recordView.message;
  $("record-banner").dataset.phase=recordView.phase.toLowerCase().replace(/[^a-z]+/g,"-");
  $("record-countdown").hidden=recordView.countdownS===null;
  if(recordView.countdownS!==null)$("record-countdown-value").textContent=recordView.countdownS.toFixed(0);
  const action=$("record-banner-action");action.hidden=!recordView.action;
  if(recordView.action){action.textContent=recordView.action.label;action.dataset.action=recordView.action.command;}
}
/** Render server-owned capture status without starting or advancing acquisition. */
async function refreshState() {
  state=await api("state"); const c=state.capture, active=state.active_session;
  document.body.classList.toggle("demo",state.demo);$("mode").className=`badge ${state.demo?"demo":""}`;
  $("mode").textContent=state.demo?"SYNTHETIC DEMO · NO ROBOT":"READ ONLY · PHYSICAL SOURCES";
  $("record-title").textContent=active?active.name+demoLabel(active):"No session open";
  const min=c.min_episode_s??0,max=c.max_episode_s??0,elapsed=c.elapsed_s||0;
  $("capture-header").textContent=active?`Episode ${c.episode_index??"–"} · ${c.state} · ${elapsed.toFixed(1)} s of ${min}–${max} s · ${c.pending_frame_count||0} frames`:"";
  renderRecordBanner(c, active);
  $("preflight").textContent=JSON.stringify({...state.preflight,free_disk_bytes:state.free_disk_bytes,required_free_bytes:state.min_free_bytes},null,2);
  if(c.catalog_error||c.error)error(c.catalog_error||c.error);
  const recording=c.state==="recording";
  const reasons=recordView.reasons;
  document.querySelectorAll("[data-command]").forEach(node=>{
    const cmd=node.dataset.command;
    node.disabled=!active||(cmd === "cancel_countdown" ? c.countdown_s === null || c.countdown_s === undefined : !c.controls?.[cmd]);
    node.classList.toggle("available",!node.disabled);node.title=node.disabled?(reasons[cmd]||"Not available now"):"";
  });
  $("discard").disabled=!active||!c.controls?.discard;
  $("discard").classList.toggle("available",!$("discard").disabled);$("discard").title=$("discard").disabled?(reasons.discard||"Not available now"):"";
  $("close-session").disabled=!active;$("last-save").hidden=!c.last_episode;
  $("save-result").textContent=recordView.lastSave;$("save-details").textContent=JSON.stringify(c.last_episode||{},null,2);
  $("counters").textContent=recordView.counters;
  const follower=c.follower||c.state_source||{}, values=follower.values||follower.positions||c.joint_state;
  if(Array.isArray(values)){live.push({state:values.slice(0,7)});live=live.slice(-100);$("joint-values").textContent=values.slice(0,7).map((v,i)=>`${i===6?"Gripper":"J"+(i+1)}: ${v.toFixed(4)}`).join("  ")+` · incoming ${follower.rate_hz??"synthetic"} Hz · age ${(follower.age_s||0).toFixed(3)} s${follower.stale?" · STALE":""}`;trace($("live-trace"),live);}else if(follower.error){$("joint-values").textContent=follower.error;}
  for(const camera of ["wrist","front"]){const info=c.cameras?.[camera]||{};$(camera+"-health").textContent=window.SynriaRecord.cameraBrief(info);$(camera+"-source").textContent=JSON.stringify(info,null,2);$(camera+"-preview").closest("figure").classList.toggle("stale",Boolean(info.error)||(info.age_s??0)>0.2);if(active&&!$("record").hidden)$(camera+"-preview").src=`/media/preview/${camera}?v=${Date.now()}`;}
  const cue=recording?(elapsed>=max-2?"cap":elapsed>=min?"minimum":"start"):c.last_episode?`saved-${c.last_episode.episode_index}`:"";
  if(cue&&cue!==lastCue){tone();lastCue=cue;}lastCapture=c.state;
  renderFlowNav();
  window.dispatchEvent(new Event("console-state"));
}
/** Request one existing recorder transition and refresh the resulting state. */
async function command(name) {
  if(name==="start"&&Number($("countdown").value)>0)await api("record/countdown",{seconds:Number($("countdown").value)});
  else await api(`record/${name}`,{});
  await refreshState();if(["success","failure","retry"].includes(name))await refreshSessions();
  if(!state.capture?.error&&!state.capture?.catalog_error)error("");
}
/** Display saved labels and gates without changing episode evidence. */
async function loadEpisodes() {
  playing=false;selected=null;$("player").hidden=true;const id=reviewSessionId;if(!id)return;
  const episodes=await api(`episodes/${id}`);$("episodes").replaceChildren();
  for(const ep of episodes){const tr=text("tr","");for(const value of [ep.episode_index,ep.operator_label,ep.duration_s,ep.frame_count,ep.achieved_sample_rate_hz,ep.gate_passed?"pass":JSON.stringify(ep.failed_gates||ep.failed_gates_json),ep.excluded?"yes":"no"]){tr.append(text("td",typeof value==="number"?String(Math.round(value*100)/100):String(value??"unknown")));}
    const actions=text("td",ep.note||"");actions.append(button(`Play take ${ep.episode_index}`,()=>selectEpisode(id,ep)));
    if(id!=="smoke"){actions.append(button(ep.excluded?"Restore":"Exclude…",()=>curate(id,ep)));actions.append(button("Note",async()=>{const note=prompt("Append episode note","");if(note!==null){await api(`episode/${id}/${ep.episode_index}/note`,{note});await loadEpisodes();}}));}if(sessions.find(session=>session.id===id)?.recovery_blocked)actions.querySelectorAll("button").forEach(node=>{if(!node.textContent.startsWith("Play take"))node.disabled=true;});tr.append(actions);$("episodes").append(tr);}
  if(!episodes.length)$("episodes").append(text("tr","")).append(text("td","No takes saved yet."));
}
/** Request an audited exclusion or restoration without rewriting raw episodes. */
async function curate(id, ep) {
  const session=sessions.find(s=>s.id===id), operator=prompt("Operator",session?.operator||"");if(!operator)return;
  let reason_code="capture_fault",note="";
  if(!ep.excluded){reason_code=prompt("Quality-valid demonstrations that failed the task stay in the dataset with their failure label.\nReason: capture_fault, scene_setup_error, operator_interruption, other","");if(!reason_code)return;note=prompt("Exclusion note (required for other)","");if(note===null)return;}
  await api(`episode/${id}/${ep.episode_index}/${ep.excluded?"restore":"exclude"}`,{operator,reason_code,note});await loadEpisodes();await refreshSessions();
}
/** Load one episode while rejecting late responses from earlier selections. */
async function selectEpisode(id, ep) {
  const selection={id,ep};selected=selection;playing=false;frame=0;rows=[];const generation=++playGeneration;$("player").hidden=false;
  const reviewedSession=sessions.find(session=>session.id===id)||state.active_session;
  $("player-title").textContent=`Take ${ep.episode_index} · ${ep.operator_label}${reviewedSession?demoLabel(reviewedSession):""}`;
  $("still").src=`/media/still/${id}/${ep.episode_index}`;
  const provenance=await api(`provenance/${id}/${ep.episode_index}`);
  if(generation!==playGeneration||selected!==selection)return;
  $("episode-provenance").textContent=JSON.stringify(provenance,null,2);
  await showFrame();if(generation!==playGeneration||selected!==selection)return;
  const count=Number($("scrub").max)+1,loaded=[];
  for(let i=0;i<count;i++){const row=await api(`frame/${id}/${ep.episode_index}/${i}`);if(generation!==playGeneration||selected!==selection)return;loaded.push({state:row["observation.state"],action:row.action});}
  if(generation!==playGeneration||selected!==selection)return;rows=loaded;trace($("review-trace"),rows,frame);
}
/** Display a matched camera pair and exact row only for the current frame request. */
async function showFrame() {
  if(!selected)return;
  const selection=selected,index=frame,generation=++frameGeneration,{id,ep}=selection;
  const row=await api(`frame/${id}/${ep.episode_index}/${index}`);
  if(generation!==frameGeneration||selected!==selection||frame!==index)return;
  const images=await Promise.all(["wrist","front"].map(async name=>{
    const image=new Image();image.id="play-"+name;image.alt=`Recorded ${name} frame ${index}`;
    image.src=`/media/frame/${id}/${ep.episode_index}/${index}/${name}`;await image.decode();return image;
  }));
  // Commit the already decoded pair and matching row in one task. Stale requests never paint.
  if(generation!==frameGeneration||selected!==selection||frame!==index)return;
  for(const image of images)$(image.id).replaceWith(image);
  $("scrub").max=row.frame_count-1;$("scrub").value=index;$("frame-label").textContent=`Frame ${index+1} / ${row.frame_count}`;
  const names=["state_monotonic_s","wrist_monotonic_s","front_monotonic_s","action_monotonic_s"];
  const times=names.map(key=>Number(row[key]?.[0]??row[key]));
  $("frame-data").textContent=JSON.stringify({...row,state_camera_skew_s:Math.max(...times.slice(0,3))-Math.min(...times.slice(0,3)),all_source_skew_s:Math.max(...times)-Math.min(...times),action_minus_state_s:times[3]-times[0],timing_note:"Derived next-state actions intentionally reference their lookahead target; all-source skew is not the contemporaneous camera gate."},null,2);
  trace($("review-trace"),rows,index);
}
/** Advance screen-only playback using the recorded rate and selected review speed. */
async function advance() {if(!playing||!selected)return;if(frame>=Number($("scrub").max)){playing=false;$("play").textContent="Play";return;}frame++;await safe(showFrame);const session=sessions.find(s=>s.id===selected.id);setTimeout(advance,1000/((session?.settings?.fps||15)*Number($("speed").value)));}
/** Name the session a route points at, from the catalog or the open session, for headings. */
function sessionName(id) { const found = sessions.find(s => s.id === id) || (state.active_session?.id === id ? state.active_session : null); return found ? found.name + demoLabel(found) : (id === "smoke" ? "Practice run" : "Session"); }
/** Show the global nav outside a flow and the breadcrumb with one exit inside a flow. */
function renderFlowNav() {
  if (!route) return;
  const inFlow = ["new", "record", "review"].includes(route.name);
  $("global-nav").hidden = inFlow; $("flow-nav").hidden = !inFlow;
  document.querySelectorAll("#global-nav a").forEach(node => { if (node.dataset.route === route.name) node.setAttribute("aria-current", "page"); else node.removeAttribute("aria-current"); });
  if (!inFlow) return;
  const crumbs = [["Home", routes().home()]];
  if (route.name === "new") crumbs.push(["New session", null]);
  else {
    const c = state.capture || {};
    crumbs.push([sessionName(route.sessionId), null]);
    if (route.name === "record") crumbs.push([state.active_session?.id === route.sessionId ? `Recording take ${(c.episode_index ?? 0) + 1} of ${state.active_session.target_episodes ?? "?"}` : "Recording", null]);
    else crumbs.push(["Review takes", null]);
  }
  const list = $("breadcrumb");
  const signature = JSON.stringify(crumbs); if (list.dataset.signature === signature) return; list.dataset.signature = signature;
  list.replaceChildren();
  for (const [label, path] of crumbs) { const item = document.createElement("li"); if (path) { const link = text("a", label); link.href = window.SynriaRouter.href(path); link.dataset.route = path; item.append(link); } else { item.append(text("span", label)); item.setAttribute("aria-current", "location"); } list.append(item); }
}
/** Move focus to the new screen heading, scroll up and announce the place for screen readers. */
function arrive(sectionId, announcement) {
  const heading = document.querySelector(`#${sectionId} .screen-title`);
  window.scrollTo({top:0, behavior:"auto"});
  if (heading) heading.focus({preventScroll:true});
  $("route-announcer").textContent = announcement;
}
/** Render one route: pick the section, set the nav mode and load what that screen shows. */
function render(next) {
  const previous = route; route = next;
  const sections = {home:"home", sessions:"sessions", help:"help", intro:"home", new:"new", record:"record", review:"review"};
  showSection(sections[next.name]);
  $("record-route-note").hidden = true;
  if (next.name === "record" && state.active_session && state.active_session.id !== next.sessionId) {
    $("record-route-note").hidden = false;
    $("record-route-note").textContent = `Another session is open: ${state.active_session.name}. Close its sources before opening this one.`;
  } else if (next.name === "record" && !state.active_session) {
    $("record-route-note").hidden = false;
    $("record-route-note").textContent = "This session is not open. Open it from Sessions to record.";
  }
  if (next.name === "review") {
    reviewSessionId = next.sessionId;
    $("review-session-name").textContent = `${sessionName(next.sessionId)}. Playback reviews recorded data on screen. It never moves the robot.`;
    safe(async () => { await loadEpisodes(); if (next.episode !== null) { const episodes = await api(`episodes/${next.sessionId}`); const ep = episodes.find(e => e.episode_index === next.episode); if (ep) await selectEpisode(next.sessionId, ep); } });
  }
  if (next.name === "sessions") safe(refreshSessions);
  renderFlowNav();
  const titles = {home:"Home", sessions:"Sessions", help:"Help", intro:"Welcome", new:"New session", record:"Recording", review:"Review takes"};
  if (!previous || previous.path !== next.path) arrive(sections[next.name], titles[next.name]);
  window.dispatchEvent(new Event("console-route"));
}
$("exit-flow").onclick = () => window.SynriaRouter.go(routes().home());
document.querySelectorAll("[data-command]").forEach(node=>node.onclick=()=>safe(()=>command(node.dataset.command)));
$("new-form").onsubmit=event=>{event.preventDefault();safe(async()=>{const session=await api("sessions",formValues());await refreshSessions();window.SynriaRouter.go(routes().sessions());if(session?.id)error("");});};
$("smoke").onclick=()=>safe(async()=>{if(!$("new-form").reportValidity())return;await api("smoke",formValues());await refreshState();window.SynriaRouter.go(routes().record("smoke"));await refreshSessions();});
$("discard").onclick=()=>safe(async()=>{if(!confirm("Discard all pending unsaved frames? This cannot be undone."))return;const reason=prompt("Capture/setup fault reason (required)","");if(reason){await api("record/discard",{reason});await refreshState();}});
$("close-session").onclick=()=>safe(async()=>{await api(`session/${state.active_session.id}/close`,{});await refreshState();await refreshSessions();});
$("record-banner-action").onclick=()=>{if($("record-banner-action").dataset.action==="close")$("close-session").click();};
$("quit").onclick=()=>safe(async()=>{if(confirm("Close sources and quit the console? Pending captures must be resolved first.")){await api("shutdown",{});error("Console closed. You may close this tab.");}});
$("reindex").onclick=()=>safe(async()=>{const result=await api("reindex",{});await refreshSessions();error(`Reindex corrections: ${JSON.stringify(result)}`);});
$("review-latest").onclick=()=>{const ep=state.capture.last_episode;window.SynriaRouter.go(routes().review(state.active_session.id, ep?ep.episode_index:null));};
$("task").onchange=taskDescription;$("search").oninput=()=>safe(refreshSessions);
$("previous").onclick=()=>safe(async()=>{frame=Math.max(0,frame-1);await showFrame();});$("next").onclick=()=>safe(async()=>{frame=Math.min(Number($("scrub").max),frame+1);await showFrame();});
$("scrub").oninput=()=>safe(async()=>{frame=Number($("scrub").value);await showFrame();});$("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play";if(playing)advance();};
/** Accept hands-busy shortcuts only on the recording canvas, never within dialogs or focused controls. */
function recordingShortcut(event) {
  if(event.repeat||event.isComposing||event.ctrlKey||event.metaKey||event.altKey||$("record").hidden||
    document.querySelector("dialog[open]")||document.activeElement.closest("input,select,textarea,button,a,summary,[contenteditable=true]"))return;
  let cmd;
  if(event.code==="Space")cmd=lastCapture==="recording"?"stop":"start";
  else if(lastCapture==="stopped"&&event.key.toLowerCase()==="s")cmd="success";
  else if(lastCapture==="stopped"&&event.key.toLowerCase()==="f")cmd="failure";
  if(cmd&&state.capture?.controls?.[cmd]){event.preventDefault();safe(()=>command(cmd));}
}
document.addEventListener("keydown",recordingShortcut);
/** Load server state and form choices, then show the place the hash names or the natural start. */
async function boot(){await refreshState();tasks=await api("tasks");for(const task of tasks){const option=text("option",task.task_id);option.value=task.task_id;$("task").append(option);}renderTaskWindows();const cameras=await api("cameras");$("camera-note").textContent=cameras.message;for(const id of ["wrist-choice","front-choice"]){$(id).append(text("option",""));for(const camera of cameras.cameras){const option=text("option",camera);option.value=camera;$(id).append(option);}}$("mode-help").textContent=state.demo?"Synthetic demo only. No ROS, serial ports or camera devices are opened. These datasets are refused by physical summaries, training and evaluation.":"All runbook safety, wiring and preflight rules apply unchanged. This app never starts a driver or teleoperation.";$("source").disabled=state.demo;$("smoke").hidden=state.demo;taskDescription();await refreshSessions();
  const start=state.active_session?routes().record(state.active_session.id):routes().home();
  window.SynriaRouter.start(render,start);
  window.synriaConsoleReady=true;window.dispatchEvent(new Event("console-ready"));}
safe(boot);setInterval(()=>safe(refreshState),700);
