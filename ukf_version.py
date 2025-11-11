import numpy as np
from numpy.linalg import cholesky, inv
import math
import pickle
import matplotlib.pyplot as plt

# --------------------------
# توابع کمکی
# --------------------------
def wraptopi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

# مدل حرکت خودرو (differential drive)
def motion_model(x, u, dt):
    """x: [x, y, theta], u: [v, omega]"""
    theta = x[2]
    x_new = np.zeros(3)
    x_new[0] = x[0] + u[0] * math.cos(theta) * dt
    x_new[1] = x[1] + u[0] * math.sin(theta) * dt
    x_new[2] = wraptopi(x[2] + u[1] * dt)
    return x_new

# مدل اندازه گیری LiDAR
def measurement_model(x, landmark, d):
    dx = landmark[0] - x[0] - d * math.cos(x[2])
    dy = landmark[1] - x[1] - d * math.sin(x[2])
    r = np.sqrt(dx**2 + dy**2)
    phi = wraptopi(math.atan2(dy, dx) - x[2])
    return np.array([r, phi])

# --------------------------
# UKF Helper Functions
# --------------------------
def generate_sigma_points(x, P, kappa):
    n = len(x)
    sigma_points = np.zeros((2 * n + 1, n))
    sigma_points[0] = x
    U = cholesky((n + kappa) * P)
    for i in range(n):
        sigma_points[i + 1]     = x + U[:, i]
        sigma_points[i + 1 + n] = x - U[:, i]
    return sigma_points

def unscented_transform(sigma_points, Wm, Wc, noise_cov):
    n_sigma = sigma_points.shape[0]
    mean = np.zeros(sigma_points.shape[1])
    for i in range(n_sigma):
        mean += Wm[i] * sigma_points[i]
    cov = noise_cov.copy()
    for i in range(n_sigma):
        diff = sigma_points[i] - mean
        diff[2] = wraptopi(diff[2])
        cov += Wc[i] * np.outer(diff, diff)
    return mean, cov

# --------------------------
# Load Data
# --------------------------
with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']
x_init  = data['x_init']
y_init  = data['y_init']
th_init = data['th_init']

v  = data['v']
om = data['om']

b = data['b']
r = data['r']
l = data['l']
d = data['d']

# --------------------------
# UKF Parameters
# --------------------------
n = 3  # state dimension
kappa = 0  # scaling parameter
alpha = 0.001
beta = 2
lambda_ = alpha**2 * (n + kappa) - n
gamma = np.sqrt(n + lambda_)

Wm = np.full(2*n+1, 0.5/(n+lambda_))
Wc = np.full(2*n+1, 0.5/(n+lambda_))
Wm[0] = lambda_/(n+lambda_)
Wc[0] = lambda_/(n+lambda_) + (1 - alpha**2 + beta)

Q_km = np.diag([1,5])
R = np.diag([0.01,10])

# --------------------------
# Initialize state
# --------------------------
x_est = np.zeros([len(t), 3])
P_est = np.zeros([len(t), 3,3])

x_est[0] = [x_init, y_init, th_init]
P_est[0] = np.diag([1,1,0.1])

# --------------------------
# UKF Loop
# --------------------------
for k in range(1, len(t)):
    dt = t[k] - t[k-1]

    # Prediction Step
    sigma_points = generate_sigma_points(x_est[k-1], P_est[k-1], kappa)
    sigma_points_pred = np.zeros_like(sigma_points)
    for i in range(2*n+1):
        sigma_points_pred[i] = motion_model(sigma_points[i], [v[k-1], om[k-1]], dt)
    x_pred, P_pred = unscented_transform(sigma_points_pred, Wm, Wc, Q_km)

    # Measurement Update
    x_upd = x_pred.copy()
    P_upd = P_pred.copy()
    for i_l, landmark in enumerate(l):
        y_sigma = np.zeros((2*n+1, 2))
        for i in range(2*n+1):
            y_sigma[i] = measurement_model(sigma_points_pred[i], landmark, d)
        y_pred, P_yy = unscented_transform(y_sigma, Wm, Wc, R)
        
        # Cross-covariance
        P_xy = np.zeros((n,2))
        for i in range(2*n+1):
            dx = sigma_points_pred[i] - x_pred
            dx[2] = wraptopi(dx[2])
            dy = y_sigma[i] - y_pred
            dy[1] = wraptopi(dy[1])
            P_xy += Wc[i] * np.outer(dx, dy)
        
        # Kalman gain
        K = P_xy @ inv(P_yy)
        y_meas = np.array([r[k,i_l], wraptopi(b[k,i_l])])
        dx = y_meas - y_pred
        dx[1] = wraptopi(dx[1])
        x_upd += K @ dx
        x_upd[2] = wraptopi(x_upd[2])
        P_upd -= K @ P_yy @ K.T

    x_est[k] = x_upd
    P_est[k] = P_upd

# --------------------------
# Plot Results
# --------------------------
plt.figure()
plt.plot(x_est[:,0], x_est[:,1])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('UKF Estimated Trajectory')
plt.axis('equal')
plt.show()

plt.figure()
plt.plot(t, x_est[:,2])
plt.xlabel('Time [s]')
plt.ylabel('theta [rad]')
plt.title('UKF Estimated Orientation')
plt.show()
