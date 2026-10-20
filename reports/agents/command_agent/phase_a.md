# Command-agent Phase A

Status: **implemented, unmeasured**. The committed sample plan and gate report
exercise all five gates using an explicitly labelled hermetic simulation fixture.
They do not establish Isaac task completion, live MoveIt operation, physical
motion, real-camera perception or planner performance.

Implementation lives in schema.py, planners.py, harness.py and cli.py under
agents/command_agent/. The planner follows the existing copilot protocol shape;
runtime skill IDs and text/step limits come from the existing registries. Provider
clients are injected in tests. Local models use the launch-owned external-command
pattern with no silent fallback.

The harness validates the entire proposal before any dry-run check. G2 requires
complete finite joint states, conservative workspace reachability, the existing
ArmSafetyGate limits and configured keep-outs, including TCP segment crossings.
The optional MoveIt path requests plans/FK/collision validity only, installs
keep-out collision objects in a dedicated simulation domain, and never publishes
controller commands or invokes trajectory execution. Every step must resolve
from a trusted fixed-scene skill library; missing bindings refuse the request.
Invalid proposals and first-failure decisions remain in audit.jsonl. Audit-write
failure blocks the returned decision.

G3 always checks whole-request intent and instruction-like context data. Optional
provider monitoring independently proposes the requested skill sequence and must
agree exactly; it cannot use the planner's provider. The deterministic vocabulary
is intentionally conservative and can false-block unsupported phrasing.

Hardware remains parked. G4 requires mandatory simulation success, unchanged
Phase 0B evidence committed at HEAD and the explicit flag, then still refuses
physical eligibility because Phase C protections and review are not implemented.
There is no physical executor. This work supplies no D1–D5 evidence.

## Inspect and use

On Thor, checkout: /home/oedeh/github/physical-ai-command-agent, branch command-agent.
The first work-order commit records the operator's Focus Rule exception.

Run the standalone package CLI or the registered physical-ai-lab command:

    physical-ai-lab command-agent "put the die in the cup and roll it" --dry-run

On the present Thor setup this returns JSON and exit code 2 at G2: no trusted
skill library or isolated MoveIt runtime has been configured. This real result is
retained in thor_cli_refusal.json; it is a successful fail-closed check.

To use an already running isolated MoveIt simulation, supply --simulation-library
and --simulation-domain-id. The library JSON has exactly schema_version
(command_sim_skills_v1), start_positions (all six joint_1..joint_6 names), and
skills (skill ID to ordered joint-target dictionaries). It must be an audited
export of existing skill paths, never planner output. The current staged expert
supports cup tasks; do not infer bindings for unimplemented roll skills.
No site-specific library is fabricated by this change. Keep-out configuration
uses --keepout-config with a JSON list of minimum/maximum XYZ boxes in base_link.
The service path uses synria_arm, tool0 and the committed wrapper joint convention;
a vendor-name or calibration mapping requires separately validated evidence.

Select --planner anthropic or openai and set the matching launch-owned
COMMAND_AGENT_ANTHROPIC_MODEL or COMMAND_AGENT_OPENAI_MODEL. Install the existing
llm extra for provider SDKs. No model or API key is silently substituted.
--monitor anthropic|openai adds provider-independent checking; deterministic
checks always run. --planner external uses SYNRIA_TASK_PLANNER_COMMAND and passes
the runtime command schema on JSON stdin. --context accepts only labelled
camera/retrieved_note/other JSON records. Do not pass untrusted data as launch
configuration, a skill library, a command or model-selection authority.

The provider request shapes follow the official
[OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling)
and [Anthropic tool definitions](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use).
No live provider calls were made.

## Evidence and remaining gates

sample_plan.json and sample_gate_report.json contain the explicitly hermetic
all-gates-pass example. audit.jsonl contains that run and the actual Thor refusal.
thor_provenance.json records Python, JetPack, L4T and queried power mode alongside
the sample diagnostic timings. Those timings are not the pre-registered Phase B
latency experiment.

validation.json, ci_checks.json and regression_failures.json retain test results and the baseline
comparison. Phase B's fixed command suite and protocol are committed separately
before any evaluation. Isaac task completion and monitor benchmark measurements
remain **planned**. The RTX 5090 was not used or interrupted. No ROS service,
camera, serial device, arm driver or hardware session was opened.

Change Report:
1. Added the command package, gate tests, core-only CI job and committed sample
   evidence; shared CLI/dependency/AGENTS edits are separate small commits.
2. Ran hermetic gate/adapter/client tests, repository lint, repository type checks
   and the full regression suite on Thor; see validation.json for exact outcomes.
3. Live SDK, live MoveIt, Isaac, 5090 and physical tests not run because their
   configured/authorized runtimes are unavailable; Phase C remains parked.
4. Preserve the baseline regression failures; a qualified skill library and twin
   binding plus prospective run profile are required before Phase B measurements.
   Rebase the three shared-file commits onto the flagship before merging.
5. A separate hardware safety review, accepted Phase 0B and explicit operator
   authorization are required before any future physical execution.

Hosted command-agent CI passed all 93 tests on Python 3.10 and 3.12, including
lint and typing, at source commit c4db228. The broader repository workflow has
completed failures in unchanged recorder/console typing checks; other jobs were
still running at capture. This is not a green full-repository CI claim. The
small CI record preserves job links and the inspected causes.

Change Report (hosted validation): updated CI evidence, validation and the evidence
manifest to retain verified gate-job results. No source changed; source hashes
remain identical to the passing jobs. Broader CI, real simulation and physical
work remain open as documented. Hardware safety review is still required before
robot execution.

## Final results and operator merge instruction

The final source commit cb29a0a passed both dedicated gate jobs (93 tests each,
Python 3.10 and 3.12). The broader workflow is now complete: the Synria CPU
acceptance and ROS 2 description build passed; six general matrix jobs failed
in the previously recorded typing/dependency/console issues. The Ubuntu 3.11
general suite ran the extra CLI test: 2027 passed, 83 skipped and the same
20 existing failures remained. Completed job evidence is in ci_checks.json.

The operator explicitly instructed documenting, pushing and merging this work
on 2026-10-20. This records merge authorization, not a change to status labels
or a waiver of any robot or simulator gate. Phase B preparation proceeds on
Thor; the pre-registered command suite is unchanged, the 5090 still requires
separate release, and Phase C remains parked.

Change Report:
1. Updated final CI results, validation, this report and the evidence manifest.
2. Inspected completed hosted jobs; verified source hashes against the passing
   source and checked documentation integrity/hygiene.
3. No code tests repeated: source did not change. Live simulation and physical
   work remain unrun for the recorded binding/authorization reasons.
4. Existing broad-CI failures remain visible; proceed to Phase B preflight and
   suite execution preparation without fabricating simulator success.
5. Hardware safety review is required before any robot execution.

Merge confirmed: PR #6 was merged on the operator's instruction at
d13216a91e194050898e9cc6c9a10fa2723fb76f. The recorded baseline CI failures and
implemented-unmeasured status are retained. Phase B preparation continues in the
isolated Thor checkout; original flagship working trees are not updated here.
