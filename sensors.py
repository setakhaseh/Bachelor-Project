# sensor.py
import carla
import numpy as np
from utils import StampedData, to_mat

# --- Temporary patch for NumPy 2.0 removal of np.mat ---
if not hasattr(np, "mat"):
    np.mat = np.asmatrix
# --------------------------------------------------------


class SensorManager:
    def __init__(self, world, vehicle):
        """
        Initialize and attach IMU, GNSS, and LiDAR sensors to a vehicle.
        Args:
            world (carla.World): active CARLA world.
            vehicle (carla.Actor): vehicle actor to attach sensors to.
        """
        self.world = world
        self.vehicle = vehicle
        self.blueprint_library = world.get_blueprint_library()

        # Sensor actors
        self.imu_sensor = None
        self.gnss_sensor = None
        self.lidar_sensor = None

        # Data storage
        self.imu_f = StampedData()   # linear acceleration
        self.imu_w = StampedData()   # angular velocity
        self.gnss = StampedData()    # GNSS position
        self.lidar = StampedData()   # point clouds

        # Transformations (sensor -> vehicle)
        self.calibrations = {}

    # ==========================================================
    # --- Sensor Spawning ---
    # ==========================================================
    def setup_sensors(self):
        """Spawn and attach IMU, GNSS, and LiDAR sensors to the vehicle."""
        # --- IMU ---
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        imu_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0))
        self.imu_sensor = self.world.spawn_actor(imu_bp, imu_tf, attach_to=self.vehicle)
        self.imu_sensor.listen(self._imu_callback)

        # --- GNSS ---
        gnss_bp = self.blueprint_library.find('sensor.other.gnss')
        gnss_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
        self.gnss_sensor = self.world.spawn_actor(gnss_bp, gnss_tf, attach_to=self.vehicle)
        self.gnss_sensor.listen(self._gnss_callback)

        # --- LiDAR ---
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50.0')
        lidar_tf = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
        self.lidar_sensor = self.world.spawn_actor(lidar_bp, lidar_tf, attach_to=self.vehicle)
        self.lidar_sensor.listen(self._lidar_callback)

        # --- Save calibration transforms ---
        self.calibrations = {
            'imu_T_vehicle': self._carla_tf_to_numpy(imu_tf),
            'lidar_T_vehicle': self._carla_tf_to_numpy(lidar_tf),
            'gnss_T_vehicle': self._carla_tf_to_numpy(gnss_tf)
        }

        print("✅ Sensors initialized and listening.")

    # ==========================================================
    # --- Callbacks ---
    # ==========================================================
    def _imu_callback(self, sensor_data):
        t_ms = sensor_data.timestamp * 1000.0
        accel = np.array([
            sensor_data.accelerometer.x,
            sensor_data.accelerometer.y,
            sensor_data.accelerometer.z
        ])
        gyro = np.array([
            sensor_data.gyroscope.x,
            sensor_data.gyroscope.y,
            sensor_data.gyroscope.z
        ])
        self.imu_f.data.append(accel)
        self.imu_f.t.append(t_ms)
        self.imu_w.data.append(gyro)
        self.imu_w.t.append(t_ms)

    def _gnss_callback(self, sensor_data):
        t_ms = sensor_data.timestamp * 1000.0
        p_world = np.array([
            sensor_data.latitude,
            sensor_data.longitude,
            sensor_data.altitude
        ])
        self.gnss.data.append(p_world)
        self.gnss.t.append(t_ms)

    def _lidar_callback(self, point_cloud):
        t_ms = point_cloud.timestamp * 1000.0
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))[:, :3]
        self.lidar.data.append(points)
        self.lidar.t.append(t_ms)

    # ==========================================================
    # --- Helper Functions ---
    # ==========================================================
    def _carla_tf_to_numpy(self, tf: carla.Transform):
        p = np.array([tf.location.x, tf.location.y, tf.location.z])
        r = np.radians([tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw])
        return to_mat(p, r)

    def save_calibration(self, filename='sensor_calibration.npz'):
        """Save calibration (sensor→vehicle transforms) to a .npz file."""
        np.savez(filename, **self.calibrations)
        print(f"💾 Calibration saved to {filename}")

    def stop_sensors(self):
        """Stop all active sensors and clean up."""
        for sensor in [self.imu_sensor, self.gnss_sensor, self.lidar_sensor]:
            if sensor:
                sensor.stop()
                sensor.destroy()
        print("🛑 Sensors stopped and destroyed.")
