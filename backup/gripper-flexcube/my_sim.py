import os
import time
import csv
import numpy as np

import mujoco
from mujoco import viewer


def main():
    # Resolve XML path relative to this file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(this_dir, "my_gripper.xml")
    log_path = os.path.join(this_dir, "cube_contact.csv")

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"Model XML not found: {xml_path}")

    # Load model and data
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Optional: try to find the contact sensor for on-screen overlay; logging will use raw contacts
    sensor_name = "cube_contact"
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id != -1:
        sensor_adr = m.sensor_adr[sensor_id]
        sensor_dim = m.sensor_dim[sensor_id]

    # Prepare CSV logging
    log_f = open(log_path, "w", newline="")
    log_writer = csv.writer(log_f)
    # We log ALL raw contacts present at the logging instant, one row per contact
    header = [
        "time",
        "contact_index",
        "geom1",
        "geom2",
        "pos_x", "pos_y", "pos_z",
        "force_cx", "force_cy", "force_cz",  # force in contact frame
        "torque_cx", "torque_cy", "torque_cz",  # torque in contact frame
    ]
    log_writer.writerow(header)

    # Launch viewer in passive mode; we will step physics and sync the GUI
    handle = viewer.launch_passive(m, d)
    last_log_time = -5.0  # ensure we log at t=0, then every 5s of sim time

    try:
        while handle.is_running():
            step_start = time.time()

            # Apply GUI inputs (e.g., control sliders, perturbations) before stepping
            handle.sync()

            # Step physics once
            mujoco.mj_step(m, d)

            # Optional overlay: show number of contacts
            with handle.lock():
                handle.set_texts((None, None, f"contacts: {d.ncon}", None))

            # Log and print every 5 seconds of simulation time: dump all contacts
            if d.time - last_log_time >= 5.0:
                n_logged = 0
                for ci in range(d.ncon):
                    out6 = np.zeros(6, dtype=float)
                    mujoco.mj_contactForce(m, d, ci, out6)
                    fx, fy, fz = float(out6[0]), float(out6[1]), float(out6[2])
                    tqx, tqy, tqz = float(out6[3]), float(out6[4]), float(out6[5])
                    px, py, pz = (
                        float(d.contact[ci].pos[0]),
                        float(d.contact[ci].pos[1]),
                        float(d.contact[ci].pos[2]),
                    )
                    g1 = int(d.contact[ci].geom1)
                    g2 = int(d.contact[ci].geom2)
                    row = [
                        d.time,
                        ci,
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1),
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2),
                        px, py, pz,
                        fx, fy, fz,
                        tqx, tqy, tqz,
                    ]
                    log_writer.writerow(row)
                    n_logged += 1
                log_f.flush()
                print(f"Logged {n_logged} contacts at t={d.time:.3f}")
                last_log_time = d.time

            # Fast state-only sync to update rendering
            handle.sync(state_only=True)

            # Keep approximate realtime
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    finally:
        handle.close()
        try:
            log_f.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
