import os
import time
import csv
import numpy as np

import mujoco
from mujoco import viewer


def main():
    # Resolve project paths
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    xml_path = os.path.join(project_root, "models", "parallel-jaw_gripper.xml")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(data_dir, "cube_contact.csv")

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
        "normal_x", "normal_y", "normal_z",
        "tangent1_x", "tangent1_y", "tangent1_z",
        "tangent2_x", "tangent2_y", "tangent2_z",
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
                    # Contact frame rows: normal in frame[0:3], tangent1 in [3:6], tangent2 in [6:9]
                    nx, ny, nz = (
                        float(d.contact[ci].frame[0]),
                        float(d.contact[ci].frame[1]),
                        float(d.contact[ci].frame[2]),
                    )
                    t1x, t1y, t1z = (
                        float(d.contact[ci].frame[3]),
                        float(d.contact[ci].frame[4]),
                        float(d.contact[ci].frame[5]),
                    )
                    t2x, t2y, t2z = (
                        float(d.contact[ci].frame[6]),
                        float(d.contact[ci].frame[7]),
                        float(d.contact[ci].frame[8]),
                    )

                    row = [
                        d.time,
                        ci,
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1),
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2),
                        px, py, pz,
                        fx, fy, fz,
                        tqx, tqy, tqz,
                        nx, ny, nz,
                        t1x, t1y, t1z,
                        t2x, t2y, t2z,
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
