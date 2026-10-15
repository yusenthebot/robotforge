# STATUS

- Goal: docs/GOAL.md (evolving; Floor = A1-A9; vision 2026-10-14: any robot, scenes, terrains, algorithm adapters)
- Plan for teaching + playability: docs/PLAY_PLAN.md (user asks 2026-10-15: a teaching mode where you learn the principles while playing; much better asset quality)
- Mode: evolving, /loop running; trunk-based on main (one short-lived branch at most, merged the same day)
- Round: 8 in progress; done: sensors lesson (7f94573), numpy walker runtime (3ba5efd), MPPI walk then stand (e803b00), skill update (e374d92); left: merge the asset upgrade
- Teaching mode (new): "学习" button -> course picker -> lesson panel (steps, quizzes, controls), live plots, 3D overlays; checks run on the physics; progress + stars per browser
  - Lesson 1 PID control: P spring, D damper, steady-state error quiz, gravity feedforward / I, graded challenge (3 stars needs the error gone)
  - Lesson 2 inverse kinematics: FK sliders, analytic IK (drag the target), workspace, two elbows, iterative DLS IK step by step, quiz
  - Lesson 3 sensors: drive a rover (WASD); encoders/odometry, IMU, 24-ray lidar drawn in 3D, bumper, robot camera picture-in-picture, drift quiz
- Rendering: procedural studio environment map + PBR presets for rf_* materials (look.js)
- In flight: asset agent on branch feat/assets (~/worktrees/robotforge-assets): detailed visual meshes for templates and props, physics unchanged
- Verified for real: both lessons walked end to end in the browser with real pointer drags and sliders (PID 3 stars at 0.36 s; IK error 55 -> 0.2 cm in 8 steps); leaving a lesson restores the world; 451 tests + smoke green
- Floor status: A1 [x GPT via codex exec] A2 [x] A3 [x] A4 [x] A5 [~ MCP App embed untested] A6 [~ Claude Code blocked by expired CLI login] A7 [x] A8 [x] A9 [x]
- Open review findings: builder HIGH 1-4, MED 6-9; unilab 1, 2, 4 + naming A-E (unilab 3 fixed: numpy runtime)
- Known gaps: builder HIGH 2 (rangefinder intprm) still open in the builder (fixed in the lesson rig); wait for the asset merge before touching builder files
- Pending for user: the 5 decisions in docs/PLAY_PLAN.md (new MCP lesson tools need a yes); license + publish timing; `claude auth login`
- Updated: 2026-10-15
