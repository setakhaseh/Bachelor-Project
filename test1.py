import carla
import time
import numpy as np
import pickle
from threading import Lock

# -------------------------
# کلاس‌های ذخیره‌سازی داده
# -------------------------
class Data:
    def __init__(self):
        self._p = []
        self._v = []
        self._a = []
        self._r = []
        self._w = []
        self._alpha = []
        self._t = []

class StampedData:
    def __init__(self):
        self.data = []
        self.t = []

class DataRecorder:
    def __init__(self):
        self.gt = Data()
        self.imu_f = StampedData()
        self.imu_w = StampedData()
        self.gnss = StampedData()
        self.lidar = StampedData()
        self.lock = Lock()

    # ground truth
    def add_gt(self, p, v, a, r, w, alpha, timestamp):
        with self.lock:
            self.gt._p.append(p)
            self.gt._v.append(v)
            self.gt._a.append(a)
            self.gt._r.append(r)
            self.gt._w.append(w)
            self.gt._alpha.append(alpha)
            self.gt._t.append(timestamp)

    # IMU
    def add_imu_f(self, data, timestamp):
        with self.lock:
            self.imu_f.data.append(data)
            self.imu_f.t.append(timestamp)

    def add_imu_w(self, data, timestamp):
        with self.lock:
            self.imu_w.data.append(data)
            self.imu_w.t.append(timestamp)

    # GNSS
    def add_gnss(self, data, timestamp):
        with self.lock:
            self.gnss.data.append(data)
            self.gnss.t.append(timestamp)

    # LIDAR
    def add_lidar(self, data, timestamp):
        with self.lock:
            self.lidar.data.append(data)
            self.lidar.t.append(timestamp)

    # ذخیره نهایی
    def save(self, filename='carla_real_data.pkl'):
        with self.lock:
            gt = {
                'p': np.array(self.gt._p),
                'v': np.array(self.gt._v),
                'a': np.array(self.gt._a),
                'r': np.array(self.gt._r),
                'w': np.array(self.gt._w),
                'alpha': np.array(self.gt._alpha),
                't': np.array(self.gt._t)
            }

            imu_f = {
                'data': np.array(self.imu_f.data),
                't': np.array(self.imu_f.t)
            }

            imu_w = {
                'data': np.array(self.imu_w.data),
                't': np.array(self.imu_w.t)
            }

            gnss = {
                'data': np.array(self.gnss.data),
                't': np.array(self.gnss.t)
            }

            lidar = {
                'data': np.array(self.lidar.data, dtype=object),
                't': np.array(self.lidar.t)
            }

            data_dict = {
                'gt': gt,
                'imu_f': imu_f,
                'imu_w': imu_w,
                'gnss': gnss,
                'lidar': lidar
            }

            with open(filename, 'wb') as f:
                pickle.dump(data_dict, f)

            print(f"داده‌ها با موفقیت ذخیره شدند: {filename}")

# -------------------------
# اتصال به CARLA
# -------------------------
recorder = DataRecorder()

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# خودروی شبیه‌سازی
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# -------------------------
# سنسورها
# -------------------------
# IMU
imu_bp = blueprint_library.find('sensor.other.imu')
imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)

def imu_callback(imu_data):
    f = np.array([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z])
    w = np.array([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z])
    timestamp = imu_data.timestamp
    recorder.add_imu_f(f, timestamp)
    recorder.add_imu_w(w, timestamp)

imu_sensor.listen(imu_callback)

# GNSS
gnss_bp = blueprint_library.find('sensor.other.gnss')
gnss_sensor = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)

def gnss_callback(gnss_data):
    pos = np.array([gnss_data.latitude, gnss_data.longitude, gnss_data.altitude])
    timestamp = gnss_data.timestamp
    recorder.add_gnss(pos, timestamp)

gnss_sensor.listen(gnss_callback)

# LIDAR
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('points_per_second', '10000')
lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(), attach_to=vehicle)

def lidar_callback(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
    timestamp = lidar_data.timestamp
    recorder.add_lidar(points, timestamp)

lidar_sensor.listen(lidar_callback)

# -------------------------
# حلقه اصلی جمع‌آوری داده
# -------------------------
try:
    vehicle.set_autopilot(True)
    duration = 15  # ثانیه
    start_time = time.time()

    while time.time() - start_time < duration:
        snapshot = world.get_snapshot()
        frame_time = snapshot.timestamp.elapsed_seconds

        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        accel = vehicle.get_acceleration()
        rot = transform.rotation
        omega = vehicle.get_angular_velocity()
        alpha = [0,0,0]  # می‌توان محاسبه کرد

        recorder.add_gt(
            np.array([transform.location.x, transform.location.y, transform.location.z]),
            np.array([velocity.x, velocity.y, velocity.z]),
            np.array([accel.x, accel.y, accel.z]),
            np.array([rot.roll, rot.pitch, rot.yaw]),
            np.array([omega.x, omega.y, omega.z]),
            np.array(alpha),
            frame_time
        )

        time.sleep(0.05)

finally:
    recorder.save('carla_real_data.pkl')
    imu_sensor.destroy()
    gnss_sensor.destroy()
    lidar_sensor.destroy()
    vehicle.destroy()
