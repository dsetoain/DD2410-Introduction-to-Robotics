import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def jacobian_matrix(theta_1, length_1, theta_2, length_2):
    j_11 = -length_1 * np.sin(theta_1) - length_2 * np.sin(theta_1 + theta_2)
    j_12 = -length_2 * np.sin(theta_1 + theta_2)
    j_21 = length_1 * np.cos(theta_1) + length_2 * np.cos(theta_1 + theta_2)
    j_22 = length_2 * np.cos(theta_1 + theta_2)
    return np.array([[j_11, j_12], [j_21, j_22]])

class DynamicModel():
    def __init__(self, length_list, mass_list, inertia_list, friction_list):
        self.length_list = length_list
        self.mass_list = mass_list  
        self.inertia_list = inertia_list
        self.friction_list = friction_list
        self.gravity = 9.81

    def mass_matrix(self, q):
        length_1 = self.length_list[0]
        length_2 = self.length_list[1]
        mass_1 = self.mass_list[0]
        mass_2 = self.mass_list[1]
        link_inertia_1 = self.inertia_list[0]
        link_inertia_2 = self.inertia_list[1]
        theta_2 = q[1]
        
        com_1 = length_1 / 2
        com_2 = length_2 / 2

        m_11 = (mass_1 * (com_1 ** 2)) + link_inertia_1 + (mass_2 * (length_1 ** 2 + com_2 ** 2 + 2 * length_1 * com_2 * np.cos(theta_2))) + link_inertia_2
        m_12 = mass_2 * (com_2 ** 2 + length_1 * com_2 * np.cos(theta_2)) + link_inertia_2
        m_21 = m_12
        m_22 = (mass_2 * (com_2 ** 2)) + link_inertia_2

        return np.array([[m_11, m_12], [m_21, m_22]])

    def coriolis_matrix(self, q, q_dot):
        theta_2 = q[1]
        theta_dot_1, theta_dot_2 = q_dot
        mass_2 = self.mass_list[1]
        length_1 = self.length_list[0]
        com_2 = self.length_list[1] / 2
        coefficient = (-mass_2) * length_1 * com_2 * np.sin(theta_2) 
        velocity_matrix = np.array([[theta_dot_2, (theta_dot_1 + theta_dot_2)], [(-theta_dot_1), 0]])
        return coefficient * velocity_matrix

    def gravity_matrix(self, q):
        mass_1, mass_2 = self.mass_list
        length_1, length_2 = self.length_list
        com_1 = length_1 / 2
        com_2 = length_2 / 2
        theta_1, theta_2 = q
        g_1 = ((mass_1 * com_1) + (mass_2 * length_1)) * self.gravity * np.cos(theta_1) + (mass_2 * self.gravity * com_2 * np.cos(theta_1 + theta_2))
        g_2 = (mass_2 * self.gravity * com_2 * np.cos(theta_1 + theta_2))
        return np.array([g_1, g_2])

    def damping_matrix(self, q_dot):
        friction_1, friction_2 = self.friction_list
        velocity_1, velocity_2 = q_dot
        return np.array([friction_1 * velocity_1, friction_2 * velocity_2])

    def forward(self, t, y, desired_angles, external_force, Kp, Kd):
        q = y[0:2]
        q_dot = y[2:4]
        g = self.gravity_matrix(q)
        error = desired_angles - q
        control_torque = Kp * error - Kd * q_dot

        # Add external forces
        tau = control_torque + g + external_force

        M = self.mass_matrix(q)
        C_dot_q = self.coriolis_matrix(q, q_dot) @ q_dot
        D = self.damping_matrix(q_dot)

        q_acceleration = np.linalg.inv(M) @ (tau - C_dot_q - D)
        dy_dt = np.concatenate((q_dot, q_acceleration))
        return dy_dt

def end_effector_position(lengths, thetas):
    """ Calculate the end effector position """
    x = lengths[0] * np.cos(thetas[0]) + lengths[1] * np.cos(thetas[0] + thetas[1])
    y = lengths[0] * np.sin(thetas[0]) + lengths[1] * np.sin(thetas[0] + thetas[1])
    return x, y

if __name__ == "__main__":
    length_list = [2, 1]
    mass_list = [50, 25]
    inertia_list = [10, 5]
    friction_list = [50, 25]

    model = DynamicModel(length_list, mass_list, inertia_list, friction_list)

    x_desired = np.sqrt(3)
    y_desired = 0
    theta_desired = -np.pi / 2
    desired_angles = np.array([np.arctan2(y_desired, x_desired), theta_desired])
    initial_conditions = np.array([0.01, 0.01, 0, 0])  # Small offset to avoid singularity
    t_span = (0, 15)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    Kp = np.array([150, 150])
    Kd = np.array([20, 20])

    external_force = np.array([0, 100])  # External force upwards

    # Solve the system
    rate_of_change = solve_ivp(model.forward, t_span, initial_conditions, t_eval=t_eval, args=(desired_angles, external_force, Kp, Kd))

    # Plotting the results
    plt.figure(figsize=(12, 6))

    # Plot joint angles
    plt.subplot(3, 1, 1)
    plt.plot(rate_of_change.t, rate_of_change.y[0], label='Theta 1 (rad)')
    plt.plot(rate_of_change.t, rate_of_change.y[1], label='Theta 2 (rad)')
    plt.axhline(desired_angles[0], color='r', linestyle='--', label='Desired Theta 1')
    plt.axhline(desired_angles[1], color='g', linestyle='--', label='Desired Theta 2')
    plt.title('Joint Angles over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (radians)')
    plt.legend()
    plt.grid()

    # Plot end-effector position
    end_effector_x = []
    end_effector_y = []
    for theta1, theta2 in zip(rate_of_change.y[0], rate_of_change.y[1]):
        x, y = end_effector_position(length_list, [theta1, theta2])
        end_effector_x.append(x)
        end_effector_y.append(y)

    plt.subplot(3, 1, 2)
    plt.plot(rate_of_change.t, end_effector_x, label='End Effector X (m)')
    plt.plot(rate_of_change.t, end_effector_y, label='End Effector Y (m)')
    plt.title('End-Effector Position over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid()

    # Calculate joint torques
    joint_torques = []
    for q, q_dot in zip(rate_of_change.y[:2].T, rate_of_change.y[2:4].T):
        M = model.mass_matrix(q)
        C = model.coriolis_matrix(q, q_dot)
        g = model.gravity_matrix(q)
        torque = M @ (desired_angles - q) + g - C @ q_dot
        joint_torques.append(torque)

    joint_torques = np.array(joint_torques)

    plt.subplot(3, 1, 3)
    plt.plot(rate_of_change.t, joint_torques[:, 0], label='Joint Torque 1 (Nm)')
    plt.plot(rate_of_change.t, joint_torques[:, 1], label='Joint Torque 2 (Nm)')
    plt.title('Joint Torques over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
