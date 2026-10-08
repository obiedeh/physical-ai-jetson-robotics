# D2 evaluation pre-registration template

Software status: **implemented, unmeasured**. D2: **planned**.
Copy this template for an evaluation campaign, fill operator fields, run the
protocol registration command, and commit it before collecting any trials.
The SHA-256 covers this entire UTF-8 document with the protocol_sha256 value
cleared to an empty string. The EvalLog must repeat that hash. Registration
does not constitute operator approval or hardware evidence.

```json
{
  "trials": 20,
  "success_threshold": 14,
  "protocol_sha256": ""
}
```

- Operator:
- UTC date:
- Policy/checkpoint hash:
- Dataset id/hash:
- Device, controller, serving host and power:
- Accepted safety records and verified limits:
- Scene schedule and randomisation seed:
- Camera identities, settings and calibration:

Pre-register 20 trials with varied reachable source/target squares, token
colours and lighting; randomise their order before recording. Freeze the
schedule with the protocol. Keep the board and camera mounts fixed.

Success requires the intended token upright and entirely inside its intended
target square, gripper released and arm retracted. The operator labels every
trial success or failure and links a timestamped final front-camera still.
Record cumulative reached, grasped, lifted, placed and released observations.
Uncertain outcomes are failures. All failures remain in the EvalLog. Retries
are individually recorded; first-try rate uses attempt one only. The trial
result is the final attempt; the threshold applies to all 20 trials.

The threshold is read from the committed configuration above. Change it only
prospectively with a new protocol and documented decision. An incomplete
evaluation cannot pass. Synthetic evaluations never advance D2.

Object success is the operator's label plus a camera still; no independent sensor confirms it.
