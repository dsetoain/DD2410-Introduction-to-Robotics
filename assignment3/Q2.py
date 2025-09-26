import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class PendulumModel:
    def __init__(self, mass, link_inertia, inertia, length, joint_friction, motor_friction, reduction_ratio):
        self.m = mass
        self.I = link_inertia
        self.I_m = inertia
        self.L = length
        self.d = joint_friction
        self.d_m = motor_friction
        self.K_r = reduction_ratio
        self.gravity = 9.81

    def Lagrangian(self, theta, theta_dot):
        kinetic_energy = 0.5 * (self.I + self.I_m * self.K_r**2) * theta_dot**2
        potential_energy = self.m * self.gravity * self.L * (1 - np.cos(theta))
        return kinetic_energy - potential_energy

    def derivatives(self, y, tau):
        theta, theta_dot = y
        theta_dot_dot = (tau - (self.d + self.d_m * self.K_r**2) * theta_dot - self.m * self.gravity * self.L * np.sin(theta)) / (self.I + self.I_m * self.K_r**2)
        return np.array([theta_dot, theta_dot_dot])

    def controller(self, theta, theta_dot, theta_desired, Kp, Kd):
        error = theta_desired - theta
        torque = Kp * error - Kd * theta_dot
        return torque


def simulate_pendulum(model, theta_desired, initial_conditions, time_span, Kp, Kd):
    t_eval = np.linspace(time_span[0], time_span[1], 1000)
    torque_history = []

    def system_equations(t, y):
        theta, theta_dot = y
        model = PendulumModel(mass, link_inertia, inertia, length, joint_friction, motor_friction, reduction_ratio)
        torque = model.controller(theta, theta_dot, theta_desired, Kp, Kd)
        torque_history.append(torque)
        return model.derivatives(y, torque)

    solution = solve_ivp(system_equations, time_span, initial_conditions, t_eval=t_eval)

    torque_interp = np.interp(solution.t, np.linspace(time_span[0], time_span[1], len(torque_history)), torque_history)

    return solution.t, solution.y, torque_interp


# Parameters
mass = 5  # kg
link_inertia = 25  # kg*m^2
inertia = 2  # kg*m^2
length = 0.5  # m
joint_friction = 25  # Nms/rad
motor_friction = 5  # Nms/rad
reduction_ratio = 10  # unitless

model = PendulumModel(mass, link_inertia, inertia, length, joint_friction, motor_friction, reduction_ratio)

# Simulation settings
theta_desired = np.deg2rad(45)  # Desired angle in radians
initial_conditions = np.array([0, 0])  # Initial conditions: [theta, theta_dot]
time_span = (0, 30)  # Simulate for 5 seconds

# Controller gains
Kp = 130  # Proportional gain
Kd = 15   # Derivative gain

# Run simulation
t, y, torque = simulate_pendulum(model, theta_desired, initial_conditions, time_span, Kp, Kd)

# Plot results
plt.figure(figsize=(12, 6))

# plot link pose
plt.subplot(2, 1, 1)
plt.plot(t, y[0], label='angle')
plt.axhline(theta_desired, color='r', linestyle='--', label='desired angle')
plt.title('link pose over time')
plt.xlabel('time')
plt.ylabel('angle')
plt.legend()
plt.grid()

# plot joint torque
plt.subplot(2, 1, 2)
plt.plot(t, torque, label='torque')
plt.title('joint torque over time')
plt.xlabel('time')
plt.ylabel('torque')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
