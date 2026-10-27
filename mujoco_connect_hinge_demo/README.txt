MuJoCo connect-as-hinge minimal demo

Files:
- connect_hinge_demo.xml: free rigid body + two site-to-site connect equalities.
- run_demo.py: applies a constant torque about world X; Space pauses/runs, N steps once.

Run from this directory:
    python run_demo.py

Requires:
    pip install mujoco

Expected behavior:
- The long bar's axis is X.
- Two pairs of sites are initially coincident at x=-0.1 and x=+0.1.
- The two connect constraints are collinear, so the intended remaining motion is rotation about X.
- The orange off-axis marker should rotate around X when the simulation runs.

This is a minimal demonstration of the documented modeling method, not a guarantee that every closed-chain model will be numerically stable.
