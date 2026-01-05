\# realtime locomotion mvp



\- target: 60 fps runtime, policy @ 30 hz

\- biped: physics-based, pd tracking

\- behaviors: idle, walk, run, turn-in-place, turning while moving

\- terrain: flat + ramp (±10–15°)

\- stack: pytorch training → onnx export → c++ inference

\- polish: foot-lock ik + slope-aligned feet

\- deliverables: demo video, github repo, short writeup

