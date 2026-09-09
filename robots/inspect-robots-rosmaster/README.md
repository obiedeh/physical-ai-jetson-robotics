# inspect-robots-rosmaster

An [Inspect Robots](https://github.com/robocurve/inspect-robots) embodiment
adapter for the **Yahboom ROSMASTER M3 Pro** 6-DOF arm (Jetson Orin NX) — as
far as we know the first mobile-manipulator-class arm in the open physical-AI
eval framework. It lets any LLM/VLA policy Inspect Robots supports (and its
`inspect-robots-agent`) be scored on the real M3 Pro arm, with immutable
EvalLogs (config, git rev, transcript, frames).

The embodiment wraps our `lerobot_robot_rosmaster_m3pro` ZMQ client, so the
robot is served on the Orin (`m3pro_host`, ACTIVE mode — the policy commands
the arm) and evaluated from the 5090.

v1 is arm-only (`joint_pos`, 6 dims, absolute degrees) for tabletop
pick-place — the same scoring convention as the Ludo/Synria campaign. Mobile
fetch (base velocity) is a planned extension.

    inspect-robots doctor --embodiment rosmaster_m3pro          # offline conformance
    inspect-robots run --embodiment rosmaster_m3pro \
        --instruction "pick up the cube and place it on the plate" \
        --policy <vla-or-agent> --grader operator -E remote_ip=192.168.1.251

Real-robot contract (Inspect Robots discipline): human-in-the-loop reset
(ConsolePoll), no privileged success oracle, wall-clock control rate.
