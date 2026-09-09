# RoboCar SLAM and Navigation

This track targets the Yahboom Orin NX 8GB ROS 2 robot car with mecanum wheels,
display, multimodal AI features, vision recognition, and an onboard 6DOF arm.

Current CLI entry point:

```bash
physical-ai-lab mecanum-calibration
physical-ai-lab mecanum-calibration --json
```

The initial helper summarizes commanded-vs-measured mecanum calibration trials
and flags whether the base is ready for SLAM bring-up or still needs tuning.
Once the Yahboom robot is online, replace the demo trials with captured motion
evidence from the real base.

Planned work:

- Yahboom ROS 2 bring-up
- mecanum base calibration
- sensor calibration notes
- SLAM launch files
- map storage and validation
- Nav2 integration
- Jetson runtime benchmarks
- mobile manipulation navigation tasks
