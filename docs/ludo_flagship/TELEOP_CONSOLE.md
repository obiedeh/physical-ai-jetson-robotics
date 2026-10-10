# Synria Teleop Console

**Implemented, unmeasured.** A local recording and review surface for the
Synria/Alicia-D leader-follower operator. It observes the existing recorder;
it does not move the arm or start teleoperation. D1–D5 remain planned.

## Launch

Use the existing Python 3.12 recording environment with the `robot-learning`
extra and ffmpeg. Demo also needs that extra: it writes real upstream datasets,
not a substitute format. Use an editable repository install so the canonical
task registry and limits remain beside the installed sources. From the repository root:

```bash
source "$RECORDING_VENV/bin/activate"
python -m pip install -e '.[robot-learning,vision]'
synria-teleop-console --demo --workspace "$HOME/synria-console-demo"
```

The browser opens automatically. For a future authorized physical session,
complete [the D1 safety and wiring preflight](D1_OPERATOR_RUNBOOK.md#1-safety-wiring-and-recording-environment),
retain its read-only state-source configuration, then launch:

```bash
synria-teleop-console --workspace "$HOME/synria-console"
```

The [RTX launcher](../../scripts/linux_rtx/synria_teleop_console.sh) activates
`RECORDING_VENV` (default `/srv/venvs/synria-d1-py312`) without starting ROS or
any robot process. Copy the [desktop entry](../../scripts/linux_rtx/synria-teleop-console.desktop)
to `~/.local/share/applications/`, adjusting its checkout, workspace and
environment paths for this host; its terminal must retain the authorized ROS
environment for physical recording. Neither launch configures a driver.

## Use

![Synthetic demo session; no physical data or hardware run.](../assets/synria-teleop-console-demo.png)

Home lists named sessions and saved success/failure, excluded and target counts.
Create selects the registry task, explicit gripper/source kind, cameras, rate,
image size and operator metadata. Real task windows remain unset: qualifying
collection is refused until prospective operator timing is recorded in
[DECISIONS.md](DECISIONS.md) and the registry is configured outside the console.
Only disposable smoke is available before then. Its button runs the shared
preflight, then automatically starts the existing 20-second temporary capture;
this flow is fake-tested, not hardware-verified. Review remains available until
closing its context, when temporary data is disposed. It never qualifies.

The new-session form offers documented Synria presets: explicit 15/30 FPS
choices, 224-pixel stored dimensions, one-frame lookahead, a 10-second first-state
wait and 10/50/100-episode targets. Each has a custom-value option; existing
validation remains authoritative. Follower-state actions remain the default
for hardware-sync wiring; USB leader actions remain optional. Gripper and state
source have no assumed selection. Choose actual camera IDs, identity and power
state yourself. Presets are candidates or software defaults, not hardware
measurements, verified limits or qualifying counts. The native final still and
operator-owned task windows are unchanged. **Implemented, unmeasured.**

Record shows both latest camera views, source age/resolution, follower state,
preflight results, elapsed time and the hard cap. Space starts/stops; S/F label
a stopped episode. Optional countdown and audio cues support hands-busy use.
Retry keeps failed-save frames; Discard needs confirmation and a reason. Tab
closure does not stop capture; the server enforces the cap. Page Quit refuses
pending frames. Process signals close sources and audit any lost pending frames.

Review locks both cameras to the same stored row, with frame stepping, scrub,
speed, state/action traces, timestamps/skew, provenance and native final still.
Server-side decoding and a bounded JPEG cache avoid browser video-codec needs.
Playback only displays data. Close the capture run before curation or reindex.

## Storage and curation

The workspace must be outside git. It contains `console.sqlite3`, five rotating
consistent backups in `backups/`, and `sessions/<slug>/session.json` beside
`dataset/` and `dataset.exclusions.jsonl`. Synthetic sessions instead occupy
`sessions-demo/` and show a demo banner. Trash is `trash/<slug>-<utc>/`.
No console file is placed inside the recorder-owned dataset root. The dataset
and sidecars are authoritative; SQLite is a rebuildable catalog and append-only
audit log. Saves commit the dataset before catalog insertion. Reindex repairs
missing catalog facts from session mirrors and sidecars and records differences:

```bash
synria-teleop-console --workspace "$HOME/synria-console" --reindex
```

Discard abandons unsaved memory. Exclude keeps saved bytes and labels, appending
a reason/operator/time to the sibling log; Restore appends the inverse decision.
Quality-valid demonstrations that fail the task retain their failure label.
Summaries separate exclusions and reject stale curation hashes. Training reads
the same log but **refuses active exclusions** to preserve its frozen hash-bound
partition, with a specific refusal for excluded held-out probes; supported
subset training remains follow-up. Accepted runs record the log hash and IDs.

Delete needs an idle, unlocked session, exact typed name and reason; it moves
the directory to trash. Restore moves it back. Purge permanently removes trash
and retains an audited tombstone with counts, size, time and reason. Interrupted
purges retain their intent and can be retried; a removed directory is reconciled
before declaring completion. A cited session warns that committed evidence
needs a correction; the console never edits that evidence or saved labels.
Automatic evidence lookup uses the session slug. Before deleting, the operator
must also check differently named or manually created artifacts and record any
required correction; those links cannot be discovered from the slug alone.

## Safety and limits

The console adds no publisher, service/action client, serial access or motion
control. Existing contract, velocity, measured-rate, command-publisher, recovery,
freshness and quality checks remain in force. Preview reuses source samples;
it never opens a second camera. A free-space floor blocks Start. One instance
owns each workspace. Loopback is the default; mutations require POST plus the
per-launch token and valid Host/Origin. No external assets or uploads are used.
Non-loopback serving needs `--allow-remote` and is not recommended for recordings.
Demo contracts are refused by D1 summaries, training, serving and D2.

No hardware, camera device, serial port, ROS runtime, driver, bridge, launch
file or teleoperation was run. Physical discovery, timing, visibility and safe
site integration still need operator validation and a hardware safety review.
Object success is the operator's label plus a camera still; no independent sensor confirms it.

## Related operator surfaces — as documented on 2026-10-10

The linked pages were rechecked on that date; silence is recorded as “not
documented,” not absence of a capability. This is a scope comparison, not a ranking.

| Surface | Documented collection/review | Session evidence on the linked pages |
|---|---|---|
| [LeRobot recording](https://huggingface.co/docs/lerobot/main/en/il_robots) and [dataset tools](https://huggingface.co/docs/lerobot/en/using_dataset_tools) | Terminal right/n ends an episode, left/r repeats, Escape/q stops; optional Rerun display through `--display_data=true`; `repo_id`, `--resume=true`, hosted dataset visualizer after upload; deletion with `lerobot-edit-dataset`. | Per-episode success/failure labels, named sessions and a database are not documented there. |
| [LeLab](https://huggingface.co/docs/lerobot/en/lelab) | LeRobot graphical interface; currently documented for SO-ARM101 only; task text, episode count/durations and spacebar episode advance. | Synria read-only graph guards and this session-audit contract are not documented there. |
| [phosphobot](https://docs.phospho.ai/basic-usage/dataset-recording) | Dashboard dataset browser; recording through the Meta Quest app or recording API; LeRobot v2 output. | This Synria contract and append-only exclusion protocol are not documented there. |
| This console | **Implemented, unmeasured:** read-only collection with visible graph guards; named exact-contract resume; operator labels and per-save gates; offline frame-exact two-camera/state/action review. | **Implemented, unmeasured:** audited curation and downstream exclusion checks, rebuildable local catalog, hands-busy controls, and unmistakably synthetic no-robot demo mode. |
