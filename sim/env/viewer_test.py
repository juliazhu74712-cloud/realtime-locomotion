import traceback
import mujoco
import mujoco.viewer

print("starting viewer_test...")

XML = """
<mujoco model="box">
  <option timestep="0.01"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>
    <body pos="0 0 1">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    print("model loaded. launching viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("viewer launched (window should be visible). stepping...")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

    print("viewer closed normally.")

except Exception as e:
    print("viewer failed with exception:")
    traceback.print_exc()