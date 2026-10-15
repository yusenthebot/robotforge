# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Plan for teaching + playability: docs/PLAY_PLAN.md (user 2026-10-15: teaching mode to learn principles while playing; much better asset quality)
- Mode: evolving, /loop running; trunk-based on main (only `main` exists)
- Round: 10 done (RL lesson 2ccd6fb: a cart-pole trains live with ARS); round 11 next
- Teaching mode: Learn button -> 4 lessons (PID, IK, sensors, reinforcement learning) with live plots, 3D overlays, quizzes and physics-checked steps; 3 challenges with stars (maze escape vs par, shove survival by controller, obstacle course distance)
  - PID control; inverse kinematics (drag the target, Jacobian steps); sensors (WASD rover: encoders, IMU, lidar rays, bumper, robot camera view)
- Assets: every template robot, scene prop and parts-built robot has detailed visual meshes (rf_* materials, group 2, no mass, no contacts); physics bit-for-bit unchanged (tests/test_visuals.py); viewer renders them with a studio environment map and PBR presets
- Learned walk (RL) runs without onnxruntime (numpy ONNX evaluator, matches to 1e-4)
- Verified for real: three lessons walked end to end in the browser; new assets checked in the live viewer (park: dog, rover, arm, rocks, benches); 500 passed / 18 skipped (default install, no onnxruntime) + smoke green; challenges played in the browser (maze timer + follow camera, MPC survived 6 shoves for 2 stars)
- Floor status: A1 [x GPT via codex exec] A2 [x] A3 [x] A4 [x] A5 [~ MCP App embed untested] A6 [~ Claude Code blocked by expired CLI login] A7 [x, recompile now 70-140 ms with detailed meshes] A8 [x] A9 [x]
- Open review findings: builder MED 6-9 (snake reverse sign, bare errors, cascade on copies, shorthand merge); unilab 1, 2, 4 + naming A-E
- New finding: a rover-mounted arm that drove a turn first can fling the cube at release (gripper vs bin wall when the rover sits at an unplanned heading); see progress.md
- Pending for user: the 5 decisions in docs/PLAY_PLAN.md (new MCP lesson tools need a yes); license + publish timing; `claude auth login`
- Updated: 2026-10-15
