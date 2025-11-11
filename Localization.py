import carla
import numpy as np
import cv2
from queue import Queue

# ---- اتصال به CARLA ----
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# ---- تنظیم synchronous mode و نرخ 50Hz ----
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.02
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

# ---- فعال کردن Traffic Manager و Autopilot ----
tm = client.get_trafficmanager(8000)
tm.set_synchronous_mode(True)
tm.set_hybrid_physics_mode(True)  # باعث می‌شود خودرو با فیزیک واقعی حرکت کند
vehicle.set_autopilot(True, tm.get_port())
# ---- ساخت سنسورها ----
# IMU
imu_bp = bp_lib.find('sensor.other.imu')
imu_bp.set_attribute('sensor_tick', '0.0')
imu_sensor = world.spawn_actor(imu_bp, carla.Transform(carla.Location(z=1.0)), attach_to=vehicle)
imu_queue = Queue()
imu_sensor.listen(lambda data: imu_queue.put(data))

# GNSS
gnss_bp = bp_lib.find('sensor.other.gnss')
gnss_bp.set_attribute('sensor_tick', '0.0')
gnss_sensor = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(z=2.0)), attach_to=vehicle)
gnss_queue = Queue()
gnss_sensor.listen(lambda data: gnss_queue.put(data))

# LIDAR
lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
lidar_bp.set_attribute('sensor_tick', '0.0')
lidar_bp.set_attribute('range', '100.0')
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('rotation_frequency', '20.0')
lidar_sensor = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0.5, z=1.5)), attach_to=vehicle)
lidar_queue = Queue()
lidar_sensor.listen(lambda data: lidar_queue.put(data))

# ---- دوربین RGB ----
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '90')
camera_sensor = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)
camera_queue = Queue()
camera_sensor.listen(lambda image: camera_queue.put(image))

print("Sensors and camera running synchronously at 50 Hz.")

# ---- ذخیره داده‌ها (فقط سنسورها، بدون ذخیره دوربین) ----
imu_f_data, imu_w_data, imu_t = [], [], []
gnss_data, gnss_t = [], []
lidar_data, lidar_t = [], []

# ---- حلقه اصلی (50Hz synchronous) ----
duration = 10  # ثانیه
num_ticks = int(duration / settings.fixed_delta_seconds)
print(f"Collecting {num_ticks} frames (~{duration} sec at 50Hz)...")

try:
    for i in range(num_ticks):
        world.tick()  # حرکت خودرو + سنسورها
        
        # دریافت داده‌ها
        imu_data = imu_queue.get(timeout=1.0)
        gnss_data_in = gnss_queue.get(timeout=1.0)
        lidar_data_in = lidar_queue.get(timeout=1.0)
        camera_data = camera_queue.get(timeout=1.0)

        # ذخیره داده سنسورها
        imu_f_data.append([imu_data.accelerometer.x, imu_data.accelerometer.y, imu_data.accelerometer.z])
        imu_w_data.append([imu_data.gyroscope.x, imu_data.gyroscope.y, imu_data.gyroscope.z])
        imu_t.append(imu_data.timestamp)

        gnss_data.append([gnss_data_in.latitude, gnss_data_in.longitude, gnss_data_in.altitude])
        gnss_t.append(gnss_data_in.timestamp)

        points = np.frombuffer(lidar_data_in.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        lidar_data.append(points.mean(axis=0))
        lidar_t.append(lidar_data_in.timestamp)

        # نمایش تصویر دوربین با OpenCV
        image = np.array(camera_data.raw_data).reshape((camera_data.height, camera_data.width, 4))
        rgb_image = image[:, :, :3][:, :, ::-1]  # BGRA -> BGR
        cv2.imshow("Camera", rgb_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(f"Frame {i+1}/{num_ticks} captured at {imu_data.timestamp:.3f}")

finally:
    # ---- توقف سنسورها ----
    imu_sensor.stop()
    gnss_sensor.stop()
    lidar_sensor.stop()
    camera_sensor.stop()
    cv2.destroyAllWindows()

    # ---- بازگرداندن حالت عادی ----
    settings.synchronous_mode = False
    world.apply_settings(settings)

    print("Data collection complete.")
    print(f"Samples collected: IMU={len(imu_t)}, GNSS={len(gnss_t)}, LIDAR={len(lidar_t)}")