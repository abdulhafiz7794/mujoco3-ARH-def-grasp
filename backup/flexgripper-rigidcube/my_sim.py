import os
import time
import csv

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

    # Find sensor id for cube_contact
    sensor_name = "cube_contact"
    sensor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sensor_id == -1:
        raise RuntimeError(f"Sensor '{sensor_name}' not found in model")
    sensor_adr = m.sensor_adr[sensor_id]
    sensor_dim = m.sensor_dim[sensor_id]

    # Prepare CSV logging
    log_f = open(log_path, "w", newline="")
    log_writer = csv.writer(log_f)
    # Contact sensor fields for data="found force torque dist pos normal tangent" (num=1)
    header = [
        "time",
        "found",
        "force_x", "force_y", "force_z",
        "torque_x", "torque_y", "torque_z",
        "dist",
        "pos_x", "pos_y", "pos_z",
        "normal_x", "normal_y", "normal_z",
        "tangent_x", "tangent_y", "tangent_z",
    ]
    log_writer.writerow(header)

    # Launch viewer in passive mode; we will step physics and sync the GUI
    handle = viewer.launch_passive(m, d)

    try:
        while handle.is_running():
            step_start = time.time()

            # Apply GUI inputs (e.g., control sliders, perturbations) before stepping
            handle.sync()

            # Step physics once
            mujoco.mj_step(m, d)

            # Read and parse contact sensor block
            # Layout: [found, force(3), torque(3), dist, pos(3), normal(3), tangent(3)]
            if sensor_adr + sensor_dim <= m.nsensordata:
                base = sensor_adr
                found = float(d.sensordata[base + 0])
                fx, fy, fz = (
                    float(d.sensordata[base + 1]),
                    float(d.sensordata[base + 2]),
                    float(d.sensordata[base + 3]),
                )
                tqx, tqy, tqz = (
                    float(d.sensordata[base + 4]),
                    float(d.sensordata[base + 5]),
                    float(d.sensordata[base + 6]),
                )
                dist = float(d.sensordata[base + 7])
                px, py, pz = (
                    float(d.sensordata[base + 8]),
                    float(d.sensordata[base + 9]),
                    float(d.sensordata[base + 10]),
                )
                nx, ny, nz = (
                    float(d.sensordata[base + 11]),
                    float(d.sensordata[base + 12]),
                    float(d.sensordata[base + 13]),
                )
                tx, ty, tz = (
                    float(d.sensordata[base + 14]),
                    float(d.sensordata[base + 15]),
                    float(d.sensordata[base + 16]),
                )
            else:
                found = 0.0
                fx = fy = fz = 0.0
                tqx = tqy = tqz = 0.0
                dist = 0.0
                px = py = pz = 0.0
                nx = ny = nz = 0.0
                tx = ty = tz = 0.0

            # Update overlay with found and dist
            with handle.lock():
                handle.set_texts((None, None, f"cube_contact found={found:.0f} dist={dist:.4f}", None))

            # Write CSV row and flush
            row = [
                d.time,
                found,
                fx, fy, fz,
                tqx, tqy, tqz,
                dist,
                px, py, pz,
                nx, ny, nz,
                tx, ty, tz,
            ]
            log_writer.writerow(row)
            log_f.flush()
            # Print a confirmation for each update
            force_norm = (fx * fx + fy * fy + fz * fz) ** 0.5
            print(f"Logged contact: t={d.time:.3f}, found={found:.0f}, dist={dist:.5f}, |F|={force_norm:.5f}")

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
