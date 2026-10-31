# Validate the current simulation and CPU contracts

The CPU CI checks must exercise the contracts of the shipped source without requiring vendor assets, hardware, or a simulation runtime.

## Decision

- Validate the committed owner-authored primitive arm description, including its joint chain, finite limits and positive collision dimensions. Require that first-party description to be present. Optional vendor mesh checks retain their separate boundary. A structural test establishes neither physical collision accuracy nor hardware readiness.
- Assert that action clamping preserves the gripper tensor exactly and leaves the caller's tensor unchanged. Comparing a float32 value converted to a Python float against a decimal literal tests representation rather than preservation; no numeric tolerance is relaxed.
- Normalize the OpenCV range-mask return to its documented unsigned-byte type before combining it with a NumPy mask. This resolves the wider return type in the installed bindings without changing the classification algorithm. Synthetic colour regressions inspect the actual component masks and retain their declared labels; they are not field-accuracy evidence.
- Observe the owned browser's process-close event before removing its isolated profile. A process-exit event alone does not guarantee that standard streams have closed. Use bounded termination signals only on the spawned child, retain its profile if closure cannot be confirmed, and retry only transient filesystem-removal errors after closure.
- Keep the browser's GPU-disabled launch and synthetic loopback API. Lifecycle regressions also run against inert child events, covering delayed close, exit preceding close, escalation and a process that does not close.

The existing workflow commands and required checks remain in force. The change does not modify a recorded policy result, robot description, runtime limit, or physical control path.

## API references

- [Process-close event contract](https://nodejs.org/api/child_process.html#event-close)
- [Bounded recursive removal retries](https://nodejs.org/api/fs.html#fsrmsyncpath-options)
