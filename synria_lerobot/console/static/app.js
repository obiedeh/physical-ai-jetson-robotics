"use strict";
// The page is a view of server-owned capture state, never its clock or owner.
const $ = id => document.getElementById(id);
const fragment = new URLSearchParams(location.hash.slice(1));
if (fragment.has("token")) sessionStorage.setItem("console-token", fragment.get("token"));
history.replaceState(null, "", location.pathname);
let state = {}, sessions = [], tasks = [], selected = null, frame = 0, playing = false;
let rows = [], live = [], playGeneration = 0, frameGeneration = 0, lastCapture = "", lastCue = "", audioContext;
const token = sessionStorage.getItem("console-token") || "";
/** Send same-origin JSON requests through the existing authenticated API boundary. */
async function api(path, payload) {
  const options = payload === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json", "X-Console-Token":token}, body:JSON.stringify(payload)};
  const response = await fetch(`/api/${path}`, options);
  const result = await response.json();
  if (!response.ok) throw Error(result.error || "Request refused");
  return result;
}
/** Show an operator-facing refusal without interpreting it as a successful check. */
function error(value) { $("notice").textContent = value ? String(value.message || value) : ""; }
/** Surface asynchronous UI failures while leaving server-owned state intact. */
async function safe(operation) { try { error(""); return await operation(); } catch (reason) { error(reason); } }
/** Select a view without changing capture ownership or timing. */
function screen(id) { document.querySelectorAll(".screen").forEach(node => {node.hidden = node.id !== id;}); }
/** Build a labeled action control with shared refusal handling. */
function button(label, action, kind = "") { const node = document.createElement("button"); node.textContent = label; node.className = kind; node.onclick = () => safe(action); return node; }
/** Create text-only content so operator data cannot become executable markup. */
function text(tag, value) { const node = document.createElement(tag); node.textContent = value; return node; }
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
  const oldSelection = $("review-session").value; $("review-session").replaceChildren();
  for (const session of sessions) {
    const card = text("article", ""); card.className = "card";
    card.append(text("h3", session.name + demoLabel(session)));
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
      actions.append(button("Open",async()=>{await api(`session/${session.id}/open`,{});screen("record");await refreshState();}));
      actions.append(button("Review",async()=>{$("review-session").value=session.id;screen("review");await loadEpisodes();}));
      actions.append(button("Rename",async()=>{const name=prompt("New session name",session.name);if(name){await api(`session/${session.id}/rename`,{name});await refreshSessions();}}));
      actions.append(button("Close",async()=>{await api(`session/${session.id}/close`,{});await refreshSessions();await refreshState();}));
      actions.append(button("Delete…",()=>deleteSession(session,"trash"),"danger"));
      card.append(actions); $("session-list").append(card);
      const option=text("option",session.name+demoLabel(session));option.value=session.id;$("review-session").append(option);
    }
    if(session.recovery_blocked)actions.querySelectorAll("button").forEach(node=>{if(node.textContent!=="Review")node.disabled=true;});
  }
  if(state.active_session?.id === "smoke"){const option=text("option","Disposable smoke · never qualifying");option.value="smoke";$("review-session").append(option);}
  if([...$("review-session").options].some(o=>o.value===oldSelection))$("review-session").value=oldSelection;
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
/** Render server-owned capture status without starting or advancing acquisition. */
async function refreshState() {
  state=await api("state"); const c=state.capture, active=state.active_session;
  document.body.classList.toggle("demo",state.demo);$("mode").className=`badge ${state.demo?"demo":""}`;
  $("mode").textContent=state.demo?"SYNTHETIC DEMO · NO ROBOT":"READ ONLY · PHYSICAL SOURCES";
  $("record-title").textContent=active?active.name+demoLabel(active):"No session open";
  const min=c.min_episode_s??0,max=c.max_episode_s??0,elapsed=c.elapsed_s||0;
  $("capture-header").textContent=active?`Episode ${c.episode_index??"–"} · ${c.state} · ${elapsed.toFixed(1)} s / ${min}–${max} s · cap in ${Math.max(0,max-elapsed).toFixed(1)} s · ${c.pending_frame_count||0} frames${c.countdown_s !== null && c.countdown_s !== undefined ? ` · starts in ${c.countdown_s.toFixed(1)} s` : ""}`:"Open a named session, or run a disposable smoke check.";
  $("preflight").textContent=JSON.stringify({...state.preflight,free_disk_bytes:state.free_disk_bytes,required_free_bytes:state.min_free_bytes},null,2);
  if(c.catalog_error||c.error)error(c.catalog_error||c.error);
  const recording=c.state==="recording", stopped=c.state==="stopped", idle=c.state==="idle";
  document.querySelectorAll("[data-command]").forEach(node=>{
    const cmd=node.dataset.command;
    node.disabled=!active||(cmd === "cancel_countdown" ? c.countdown_s === null || c.countdown_s === undefined : !c.controls?.[cmd]);
  });
  $("discard").disabled=!active||!c.controls?.discard;
  $("close-session").disabled=!active;$("last-save").hidden=!c.last_episode;
  $("save-result").textContent=JSON.stringify(c.last_episode||{},null,2);
  $("counters").textContent=JSON.stringify(active?.counts||c.counters||{})+(active?` · target ${active.target_episodes||"smoke only"}`:"");
  const follower=c.follower||c.state_source||{}, values=follower.values||follower.positions||c.joint_state;
  if(Array.isArray(values)){live.push({state:values.slice(0,7)});live=live.slice(-100);$("joint-values").textContent=values.slice(0,7).map((v,i)=>`${i===6?"Gripper":"J"+(i+1)}: ${v.toFixed(4)}`).join("  ")+` · incoming ${follower.rate_hz??"synthetic"} Hz · age ${(follower.age_s||0).toFixed(3)} s${follower.stale?" · STALE":""}`;trace($("live-trace"),live);}else if(follower.error){$("joint-values").textContent=follower.error;}
  for(const camera of ["wrist","front"]){const info=c.cameras?.[camera]||{};$(camera+"-health").textContent=JSON.stringify(info);$(camera+"-preview").closest("figure").classList.toggle("stale",Boolean(info.error)||(info.age_s??0)>0.2);if(active)$(camera+"-preview").src=`/media/preview/${camera}?v=${Date.now()}`;}
  const cue=recording?(elapsed>=max-2?"cap":elapsed>=min?"minimum":"start"):c.last_episode?`saved-${c.last_episode.episode_index}`:"";
  if(cue&&cue!==lastCue){tone();lastCue=cue;}lastCapture=c.state;
}
/** Request one existing recorder transition and refresh the resulting state. */
async function command(name) {
  if(name==="start"&&Number($("countdown").value)>0)await api("record/countdown",{seconds:Number($("countdown").value)});
  else await api(`record/${name}`,{});
  await refreshState();if(["success","failure","retry"].includes(name))await refreshSessions();
}
/** Display saved labels and gates without changing episode evidence. */
async function loadEpisodes() {
  playing=false;selected=null;$("player").hidden=true;const id=$("review-session").value;if(!id)return;
  const episodes=await api(`episodes/${id}`);$("episodes").replaceChildren();
  for(const ep of episodes){const tr=text("tr","");for(const value of [ep.episode_index,ep.operator_label,ep.duration_s,ep.frame_count,ep.achieved_sample_rate_hz,ep.gate_passed?"pass":JSON.stringify(ep.failed_gates||ep.failed_gates_json),ep.excluded?"yes":"no"]){tr.append(text("td",typeof value==="number"?String(Math.round(value*100)/100):String(value??"unknown")));}
    const actions=text("td",ep.note||"");actions.append(button("Review",()=>selectEpisode(id,ep)));
    if(id!=="smoke"){actions.append(button(ep.excluded?"Restore":"Exclude…",()=>curate(id,ep)));actions.append(button("Note",async()=>{const note=prompt("Append episode note","");if(note!==null){await api(`episode/${id}/${ep.episode_index}/note`,{note});await loadEpisodes();}}));}if(sessions.find(session=>session.id===id)?.recovery_blocked)actions.querySelectorAll("button").forEach(node=>{if(node.textContent!=="Review")node.disabled=true;});tr.append(actions);$("episodes").append(tr);}
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
  $("player-title").textContent=`Episode ${ep.episode_index} · ${ep.operator_label}${reviewedSession?demoLabel(reviewedSession):""}`;
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
document.querySelectorAll("[data-screen]").forEach(node=>node.onclick=()=>safe(async()=>{screen(node.dataset.screen);if(node.dataset.screen==="review")await loadEpisodes();if(["sessions","trash"].includes(node.dataset.screen))await refreshSessions();}));
document.querySelectorAll("[data-command]").forEach(node=>node.onclick=()=>safe(()=>command(node.dataset.command)));
$("new-form").onsubmit=event=>{event.preventDefault();safe(async()=>{await api("sessions",formValues());await refreshSessions();screen("sessions");});};
$("smoke").onclick=()=>safe(async()=>{if(!$("new-form").reportValidity())return;await api("smoke",formValues());screen("record");await refreshState();await refreshSessions();});
$("discard").onclick=()=>safe(async()=>{if(!confirm("Discard all pending unsaved frames? This cannot be undone."))return;const reason=prompt("Capture/setup fault reason (required)","");if(reason){await api("record/discard",{reason});await refreshState();}});
$("close-session").onclick=()=>safe(async()=>{await api(`session/${state.active_session.id}/close`,{});await refreshState();await refreshSessions();});
$("quit").onclick=()=>safe(async()=>{if(confirm("Close sources and quit the console? Pending captures must be resolved first.")){await api("shutdown",{});error("Console closed. You may close this tab.");}});
$("reindex").onclick=()=>safe(async()=>{const result=await api("reindex",{});await refreshSessions();error(`Reindex corrections: ${JSON.stringify(result)}`);});
$("review-latest").onclick=()=>safe(async()=>{await refreshSessions();$("review-session").value=state.active_session.id;screen("review");await loadEpisodes();const ep=state.capture.last_episode;if(ep)await selectEpisode(state.active_session.id,ep);});
$("task").onchange=taskDescription;$("search").oninput=()=>safe(refreshSessions);$("review-session").onchange=()=>safe(loadEpisodes);
$("previous").onclick=()=>safe(async()=>{frame=Math.max(0,frame-1);await showFrame();});$("next").onclick=()=>safe(async()=>{frame=Math.min(Number($("scrub").max),frame+1);await showFrame();});
$("scrub").oninput=()=>safe(async()=>{frame=Number($("scrub").value);await showFrame();});$("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play";if(playing)advance();};
document.addEventListener("keydown",event=>{if(event.repeat||$("record").hidden||/INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName))return;let cmd;if(event.code==="Space")cmd=lastCapture==="recording"?"stop":"start";else if(lastCapture==="stopped"&&event.key.toLowerCase()==="s")cmd="success";else if(lastCapture==="stopped"&&event.key.toLowerCase()==="f")cmd="failure";if(cmd&&state.capture?.controls?.[cmd]){event.preventDefault();safe(()=>command(cmd));}});
/** Load server state and form choices before announcing that setup suggestions may apply. */
async function boot(){await refreshState();tasks=await api("tasks");for(const task of tasks){const option=text("option",task.task_id);option.value=task.task_id;$("task").append(option);}const cameras=await api("cameras");$("camera-note").textContent=cameras.message;for(const id of ["wrist-choice","front-choice"]){$(id).append(text("option",""));for(const camera of cameras.cameras){const option=text("option",camera);option.value=camera;$(id).append(option);}}$("mode-help").textContent=state.demo?"Synthetic demo only. No ROS, serial ports or camera devices are opened. These datasets are refused by physical summaries, training and evaluation.":"All runbook safety, wiring and preflight rules apply unchanged. This app never starts a driver or teleoperation.";$("source").disabled=state.demo;$("smoke").hidden=state.demo;taskDescription();await refreshSessions();if(state.active_session)screen("record");window.synriaConsoleReady=true;window.dispatchEvent(new Event("console-ready"));}
safe(boot);setInterval(()=>safe(refreshState),700);
