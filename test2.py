import carla
import numpy as np
import math
import csv
import time
from rotations import Quaternion
from utils import StampedData, from_mat, to_mat
# --- Temporary patch for NumPy 2.0 removal of np.mat ---
if not hasattr(np, "mat"):
    np.mat = np.asmatrix
# --------------------------------------------------------

# ==========================================================
# Initialize StampedData Containers
# ==========================================================
imu_f = StampedData()   # IMU specific force (acceleration)
imu_w = StampedData()   # IMU angular velocity
gnss  = StampedData()   # GNSS position in world frame
lidar = StampedData()   # LiDAR point cloud (positions only)

# ==========================================================
# Connect to CARLA
# ==========================================================
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town10HD_Opt')

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
world.tick()

# ==========================================================
# Sensor Setup
# ==========================================================
# IMU
imu_bp = blueprint_library.find('sensor.other.imu')
imu_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
imu_sensor = world.spawn_actor(imu_bp, imu_tf, attach_to=vehicle)

# GNSS
gnss_bp = blueprint_library.find('sensor.other.gnss')
gnss_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
gnss_sensor = world.spawn_actor(gnss_bp, gnss_tf, attach_to=vehicle)

# LiDAR
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('range', '50.0')
lidar_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
lidar_sensor = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)

# ==========================================================
# Helper: CARLA Transform → 4x4 Matrix
# ==========================================================
def carla_tf_to_numpy(tf: carla.Transform):
    p = np.array([tf.location.x, tf.location.y, tf.location.z])
    r = np.radians([tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw])
    return to_mat(p, r)

# Compute transformations (sensor -> vehicle)
imu_T_vehicle   = carla_tf_to_numpy(imu_tf)
lidar_T_vehicle = carla_tf_to_numpy(lidar_tf)
gnss_T_vehicle  = carla_tf_to_numpy(gnss_tf)

calibrations = {
    'imu_T_vehicle': imu_T_vehicle,
    'lidar_T_vehicle': lidar_T_vehicle,
    'gnss_T_vehicle': gnss_T_vehicle
}

# ==========================================================
# Sensor Callbacks
# ==========================================================
def imu_callback(sensor_data):
    """Store IMU data in vehicle frame"""
    t_ms = sensor_data.timestamp * 1000.0
    accel = np.array([sensor_data.accelerometer.x,
                      sensor_data.accelerometer.y,
                      sensor_data.accelerometer.z])
    gyro = np.array([sensor_data.gyroscope.x,
                     sensor_data.gyroscope.y,
                     sensor_data.gyroscope.z])

    imu_f.data.append(accel)
    imu_f.t.append(t_ms)
    imu_w.data.append(gyro)
    imu_w.t.append(t_ms)

def gnss_callback(sensor_data):
    """Store GNSS position in world frame"""
    t_ms = sensor_data.timestamp * 1000.0
    p_world = np.array([sensor_data.latitude,
                        sensor_data.longitude,
                        sensor_data.altitude])
    gnss.data.append(p_world)
    gnss.t.append(t_ms)

def lidar_callback(point_cloud):
    """Store LiDAR point positions in vehicle frame"""
    t_ms = point_cloud.timestamp * 1000.0
    points = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.float32))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))[:, :3]  # x, y, z
    lidar.data.append(points)
    lidar.t.append(t_ms)

# ==========================================================
# Attach Callbacks
# ==========================================================
imu_sensor.listen(imu_callback)
gnss_sensor.listen(gnss_callback)
lidar_sensor.listen(lidar_callback)

# ==========================================================
# Ground Truth Data Collection
# ==========================================================
gt_data = []

def tick_callback():
    t = world.get_snapshot().timestamp.elapsed_seconds * 1000.0
    tf = vehicle.get_transform()
    vel = vehicle.get_velocity()
    ang_vel = vehicle.get_angular_velocity()
    gt_data.append([
        t,
        tf.location.x, tf.location.y, tf.location.z,
        math.radians(tf.rotation.roll),
        math.radians(tf.rotation.pitch),
        math.radians(tf.rotation.yaw),
        vel.x, vel.y, vel.z,
        ang_vel.x, ang_vel.y, ang_vel.z
    ])

world.on_tick(lambda _: tick_callback())

# ==========================================================
# Simulation Loop
# ==========================================================
print("Recording... Press Ctrl+C to stop.")
try:
    for _ in range(600):  # record ~60s (10 Hz)
        world.tick()
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Stopping recording...")
finally:
    imu_sensor.stop()
    gnss_sensor.stop()
    lidar_sensor.stop()
    vehicle.destroy()

# ==========================================================
# Save Data
# ==========================================================
print("Saving data...")

# Ground truth
with open('ground_truth.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw',
                     'vx', 'vy', 'vz', 'wx', 'wy', 'wz'])
    writer.writerows(gt_data)

# IMU data
with open('imu.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz'])
    for i in range(len(imu_f.t)):
        writer.writerow([
            imu_f.t[i],
            *imu_f.data[i],
            *imu_w.data[i]
        ])

# GNSS data
with open('gnss.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'latitude', 'longitude', 'altitude'])
    for i in range(len(gnss.t)):
        writer.writerow([gnss.t[i], *gnss.data[i]])

# Save sensor calibration
np.savez('sensor_calibration.npz', **calibrations)

print("✅ Data saved successfully: ground_truth.csv, imu.csv, gnss.csv, sensor_calibration.npz")
