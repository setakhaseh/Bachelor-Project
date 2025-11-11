import carla
import numpy as np
import pandas as pd
import time
from data import Data  # Your Data class
import utils as u  # Utility functions


def get_vehicle_state(vehicle):
    """Return position, rotation (Euler), velocity, and acceleration arrays."""
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    accel = vehicle.get_acceleration()

    # Position in meters
    p = np.array([transform.location.x, transform.location.y, transform.location.z])

    # Rotation (convert degrees to radians)
    r = np.deg2rad(np.array([transform.rotation.roll,
                             transform.rotation.pitch,
                             transform.rotation.yaw]))

    # Velocity and acceleration vectors
    v = np.array([velocity.x, velocity.y, velocity.z])
    a = np.array([accel.x, accel.y, accel.z])

    return p, r, v, a


def record_autopilot_data(
    duration=30.0,
    save_path="autopilot_data.csv",
    town="Town03"
):
    """
    Record ground-truth data of a CARLA autopilot vehicle and save to CSV.
    """
    # --- Connect to CARLA ---
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.load_world(town)
    blueprint_library = world.get_blueprint_library()

    # --- Spawn vehicle ---
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_point = np.random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Enable autopilot
    vehicle.set_autopilot(True)
    print("Vehicle spawned and autopilot enabled.")

    # --- Initialize lists ---
    t_list, p_list, r_list, v_list, a_list = [], [], [], [], []
    start_time = time.time()

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > duration:
                break

            # Get data from vehicle
            p, r, v, a = get_vehicle_state(vehicle)

            # Append to lists
            t_list.append(elapsed)
            p_list.append(p)
            r_list.append(r)
            v_list.append(v)
            a_list.append(a)

            # Tick rate (10 Hz)
            time.sleep(0.1)

    finally:
        print("Stopping vehicle and cleaning up...")
        vehicle.destroy()

    # --- Convert to numpy arrays ---
    t = np.array(t_list)
    p = np.vstack(p_list)
    r = np.vstack(r_list)
    v = np.vstack(v_list)
    a = np.vstack(a_list)

    # --- Store in Data class ---
    car_data = Data(t=t, p=p, r=r, v=v, a=a, do_diff=True)

    # --- Save to CSV ---
    df = pd.DataFrame({
        "time": t,
        "x": p[:, 0],
        "y": p[:, 1],
        "z": p[:, 2],
        "roll": r[:, 0],
        "pitch": r[:, 1],
        "yaw": r[:, 2],
        "vx": v[:, 0],
        "vy": v[:, 1],
        "vz": v[:, 2],
        "ax": a[:, 0],
        "ay": a[:, 1],
        "az": a[:, 2]
    })
    df.to_csv(save_path, index=False)
    print(f"✅ Data saved to {save_path}")

    return car_data


if __name__ == "__main__":
    # Record 30 seconds of autopilot data in Town03
    record_autopilot_data(duration=30.0, save_path="D:/Adaptive Control/autopilot_data.csv")
