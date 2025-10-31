import carla
import numpy as np
import time
from queue import Queue

# اتصال به CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()

# ---- تنظیم حالت synchronous و نرخ 50Hz ----
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.02  # یعنی 50Hz
world.apply_settings(settings)

# ---- پاکسازی بازیگران قبلی ----
for actor in world.get_actors():
    if 'vehicle' in actor.type_id or 'sensor' in actor.type_id:
        actor.destroy()

# ---- اسپاون خودرو ----
bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.filter('model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# ---- ساخت سنسورها ----
imu_bp = bp_lib.find('sensor.other.imu')
gnss_bp = bp_lib.find('sensor.other.gnss')
lidar_bp = bp_lib.find('sensor.lidar.ray_cast')

# چون در حالت sync هستیم، sensor_tick = 0 تا با tick کنترل بشن
imu_bp.set_attribute('sensor_tick', '0.0')
gnss_bp.set_attribute('sensor_tick', '0.0')
lidar_bp.set_attribute('sensor_tick', '0.0')

# پیکربندی LIDAR
lidar_bp.set_attribute('range', '100.0')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('rotation_frequency', '20.0')

# مکان نصب سنسورها روی خودرو
imu_loc = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
gnss_loc = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.0))
lidar_loc = carla.Transform(carla.Location(x=0.5, y=0.0, z=1.5))

# اسپاون
imu_sensor = world.spawn_actor(imu_bp, imu_loc, attach_to=vehicle)
gnss_sensor = world.spawn_actor(gnss_bp, gnss_loc, attach_to=vehicle)
lidar_sensor = world.spawn_actor(lidar_bp, lidar_loc, attach_to=vehicle)

# ---- صف‌ها برای جمع‌آوری داده‌ها ----
imu_queue = Queue()
gnss_queue = Queue()
lidar_queue = Queue()

imu_sensor.listen(lambda data: imu_queue.put(data))
gnss_sensor.listen(lambda data: gnss_queue.put(data))
lidar_sensor.listen(lambda data: lidar_queue.put(data))

print("Sensors running synchronously at 50 Hz.")

# ---- ذخیره داده ----
imu_f_data, imu_w_data, imu_t = [], [], []
gnss_data, gnss_t = [], []
lidar_data, lidar_t = [], []

# ---- حلقه اصلی (50Hz synchronous) ----
duration = 10  # ثانیه
num_ticks = int(duration / settings.fixed_delta_seconds)
print(f" Collecting {num_ticks} frames (~{duration} sec at 50Hz)...")

for i in range(num_ticks):
    world.tick()  # همه سنسورها هم‌زمان داده می‌دن

    # دریافت داده‌ها از صف‌ها
    imu_data = imu_queue.get(timeout=1.0)
    gnss_data_in = gnss_queue.get(timeout=1.0)
    lidar_data_in = lidar_queue.get(timeout=1.0)

    # ذخیره IMU
    imu_f_data.append([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z])
    imu_w_data.append([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z])
    imu_t.append(imu_data.timestamp)

    # ذخیره GNSS
    gnss_data.append([gnss_data_in.latitude, gnss_data_in.longitude, gnss_data_in.altitude])
    gnss_t.append(gnss_data_in.timestamp)

    # ذخیره LIDAR (میانگین نقاط به عنوان تخمین موقعیت)
    points = np.frombuffer(lidar_data_in.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
    lidar_data.append(points.mean(axis=0))
    lidar_t.append(lidar_data_in.timestamp)

    print(f"Frame {i+1}/{num_ticks} captured at {imu_data.timestamp:.3f}")

# ---- توقف سنسورها ----
imu_sensor.stop()
gnss_sensor.stop()
lidar_sensor.stop()

# ---- بازگرداندن حالت عادی ----
settings.synchronous_mode = False
world.apply_settings(settings)

# ---- خروجی‌ها ----
imu_f = {'data': np.array(imu_f_data), 't': np.array(imu_t)}
imu_w = {'data': np.array(imu_w_data), 't': np.array(imu_t)}
gnss = {'data': np.array(gnss_data), 't': np.array(gnss_t)}
lidar = {'data': np.array(lidar_data), 't': np.array(lidar_t)}

data = {'imu_f': imu_f, 'imu_w': imu_w, 'gnss': gnss, 'lidar': lidar}

print("Synchronous 50Hz data collection complete.")
print(f"Samples collected: IMU={len(imu_t)}, GNSS={len(gnss_t)}, LIDAR={len(lidar_t)}")
