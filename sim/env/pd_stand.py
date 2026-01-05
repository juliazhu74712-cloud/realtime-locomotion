import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "sim/assets/biped.xml"

JOINTS = [
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
]

# less aggressive crouch = more stable
Q_STAND = np.array([0.02, 0.70, -0.35, 0.02, 0.70, -0.35], dtype=np.float64)


def _as_int(x):
    """Robustly convert MuJoCo address fields to Python int.
    Works if x is int, np.int32, or a 1-element ndarray like array([6]).
    """
    a = np.asarray(x)
    return int(a.reshape(-1)[0])


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Robust 1D integer arrays (prevents [[6]] / (6,1) shape issues)
    qpos_adr = np.array([_as_int(model.joint(j).qposadr) for j in JOINTS], dtype=np.int32)
    dof_adr  = np.array([_as_int(model.joint(j).dofadr)  for j in JOINTS], dtype=np.int32)

    print("qpos_adr:", qpos_adr.tolist(), "shape", qpos_adr.shape)
    print("dof_adr :", dof_adr.tolist(),  "shape", dof_adr.shape)
    print("nu (actuators):", model.nu)

    # PD gains (outer-loop -> motor ctrl), clipped to [-1, 1]
    Kp = 2.0
    Kd = 0.6

    warmup_steps = 40
    step = 0

    root_roll_dof  = _as_int(model.joint("rootroll").dofadr)
    root_pitch_dof = _as_int(model.joint("rootpitch").dofadr)
    root_yaw_dof   = _as_int(model.joint("rootyaw").dofadr)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            if step < warmup_steps:
                data.ctrl[:] = 0.0
            else:
                # Force 1D vectors
                q  = np.asarray(data.qpos[qpos_adr]).reshape(-1)
                qd = np.asarray(data.qvel[dof_adr]).reshape(-1)

                # Root angular damping to reduce spin
                data.qvel[root_roll_dof]  *= 0.90
                data.qvel[root_pitch_dof] *= 0.90
                data.qvel[root_yaw_dof]   *= 0.90

                # PD control
                u = Kp * (Q_STAND - q) - Kd * qd
                u = np.clip(u, -1.0, 1.0).reshape(-1)

                data.ctrl[:] = u

            mujoco.mj_step(model, data)

            # Debug print every ~1 second (200 * 0.005 = 1.0s)
            if step % 200 == 0:
                print(
                    "t =", round(data.time, 2),
                    "rootz =", float(data.qpos[2]),
                    "|ctrl| =", float(np.linalg.norm(data.ctrl))
                )

            viewer.sync()
            step += 1


if __name__ == "__main__":
    main()