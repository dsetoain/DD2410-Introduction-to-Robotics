import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



def jacobian_matrix(theta_1, length_1, theta_2, length_2):
    """ This function calculates the Jacobian matrix of the robotic manipulator"""

    j_11 = -length_1 * np.sin(theta_1) - length_2 * np.sin(theta_1 + theta_2)
    j_12 = -length_2 * np.sin(theta_1 + theta_2)
    j_21 = length_1 * np.cos(theta_1) + length_2 * np.cos(theta_1 + theta_2)
    j_22 = length_2 * np.cos(theta_1 + theta_2)

    return np.array([[j_11, j_12], [j_21, j_22]])


class DynamicModel():
    """ This is a class for dynamic model calculations"""

    def __init__(self, length_list, mass_list, inertia_list, friction_list) -> None:
        """ A constructor """
        self.length_list = length_list
        self.mass_list = mass_list  
        self.inertia_list = inertia_list
        self.friction_list = friction_list
        self.gravity = 9.81

    def mass_matrix(self, q):
        """
            A function to calculate mass matrix M
            params: q: joint angles
            return: mass matrix M
        """

        # Get related values
        lenght_1 = self.length_list[0]
        lenght_2 = self.length_list[1]

        mass_1 = self.mass_list[0]
        mass_2 = self.mass_list[1]

        link_inertia_1 = self.inertia_list[0]
        link_inertia_2 = self.inertia_list[1]

        theta_2 = q[1]

        # Calculate the center of mass of each link
        com_1 = lenght_1/2
        com_2 = lenght_2/2

        # Calculate the elements of mass matrix
        m_11 = (mass_1*(com_1**2)) + link_inertia_1 + (mass_2*((lenght_1**2) + (com_2**2) + (2*lenght_1*com_2*np.cos(theta_2)))) + link_inertia_2

        m_12 = mass_2*((com_2**2)+ (lenght_1*com_2*np.cos(theta_2))) + link_inertia_2

        m_21 = m_12

        m_22 = (mass_2*(com_2**2)) + link_inertia_2

        M = np.array([[m_11, m_12], [m_21, m_22]])

        return M


    def coriolis_matrix(self,q, q_dot):
        """"
            A function to calculate Coriolis matrix C
            params: q: joint angles
                    q_dot: joint velocities
            return: Coriolis matrix C
        """

        theta_2 = q[1]

        theta_dot_1, theta_dot_2 = q_dot

        mass_2 = self.mass_list[1]

        length_1, length_2 = self.length_list

        com_2 = length_2/2

        coefficient = (-mass_2) * length_1 * com_2 * np.sin(theta_2) 

        velocity_matrix = np.array([[theta_dot_2, (theta_dot_1+theta_dot_2)], [(-theta_dot_1),0]])

        C = coefficient * velocity_matrix

        return C
        

    def gravity_matrix(self, q):
        """"
            This function calculates the gravity matrix g
            params: q: joint angles
            return: gravity matrix g
        """
        
        mass_1, mass_2 = self.mass_list
        length_1, length_2 = self.length_list
        
        com_1 = length_1/2
        com_2 = length_2/2

        theta_1, theta_2 = q

        g_1 = ((mass_1*com_1) + (mass_2*length_1))*(self.gravity)*np.cos(theta_1) + (mass_2*self.gravity*com_2* np.cos(theta_1+theta_2))
        g_2 = (mass_2*self.gravity*com_2* np.cos(theta_1+theta_2))

        return np.array([g_1,g_2])


    def damping_matrix(self, q_dot):
        """"
            This function calculates the damping matrix D
            params: q_dot: joint velocities
            return: damping matrix D
        """

        friction_1, friction_2 = self.friction_list
        velocity_1, velocity_2 = q_dot
        
        return np.array([friction_1 * velocity_1, friction_2 * velocity_2])
    
    def forward(self, t, y, tau, external_forces):
        """ 
            This function is to conduct given experiments with tau and external forces
            params: tau: torque
                    external_forces: external forces
                    q: joint angles
                    q_dot: joint velocities
            return: q_acceleration
        """

        q = y[0:2]
        q_dot = y[2:4]

        # Define the mass matrix
        M = self.mass_matrix(q)

        # Define the Coriolis matrix
        C_dot_q = self.coriolis_matrix(q, q_dot) @ q_dot

        # Define the gravity matrix
        g = self.gravity_matrix(q)

        # Define the damping matrix
        D = self.damping_matrix(q_dot)

        # Define the Jacobian matrix
        J = jacobian_matrix(q[0], self.length_list[0], q[1], self.length_list[1])

        # Transpose of the Jacobian matrix
        J_T = np.transpose(J)

        # Calculate the acceleration
        q_acceleration = np.linalg.inv(M) @ (tau + J_T @ external_forces - C_dot_q - g - D)

        dy_dt = np.concatenate((q_dot, q_acceleration))

        return dy_dt
    
def experiment_2(mass_list, length_list, inertia_list, angle_list, gravity_value):

    """
        This function is to conduct the second experiment.
        params: mass_list: list of masses of the links
                length_list: list of lengths of the links
                inertia_list: list of inertias of the links
                angle_list: list of angles of the joints
                gravity_value: gravity value
        return: torque values
    """
    
    # Initialize variables
    length_1 = length_list[0]
    length_2 = length_list[1]

    com_1 = length_1/2
    com_2 = length_2/2

    mass_1 = mass_list[0]
    mass_2 = mass_list[1]

    inertia_1 = inertia_list[0]
    inertia_2 = inertia_list[1]

    theta_1 = angle_list[0]
    theta_2 = angle_list[1]

    gravity = gravity_value

    # Calculate tau (torque) values
    tau_1 = mass_1*com_1*gravity*np.cos(theta_1) + mass_2*gravity*(length_1*np.cos(theta_1) + com_2*np.cos(theta_1+theta_2))
    tau_2 = mass_2*com_2*gravity*np.cos(theta_1+theta_2)

    return tau_1, tau_2


def experiment_3(angle_list, jacobian, external_forces, gravity ,length_list):
    """
        This function is to conduct the third experiment.
        params: angle_list: list of angles of the joints
                jacobian: jacobian matrix
                external_forces: external forces
                gravity: gravity matrix
                length_list: list of lengths of the links
        return: torque values
    """


    # Initialize variables
    theta_1 = angle_list[0]
    theta_2 = angle_list[1]

    # Calculate the Jacobian matrix
    J = jacobian(theta_1, length_list[0], theta_2, length_list[1])

    # Transpose of the Jacobian matrix
    J_T = np.transpose(J)

    # Calculate the torque values
    tau = gravity - (J_T @ external_forces)

    return tau

if __name__ == "__main__":

    length_list = [2, 1]
    mass_list = [50, 25]
    inertia_list = [10, 5]
    friction_list = [50, 25]

    model = DynamicModel(length_list, mass_list, inertia_list, friction_list)

    # Experiment 1
    q = np.array([0,0])
    q_dot = np.array([0,0])

    y_0 = np.array([q[0], q[1], q_dot[0], q_dot[1]]) # Initial state to do differential equation

    t_span = (0,20) # Time span
    t_eval = np.linspace(0, 20, 1000) # Time values to evaluate in the differential equation
    tau = np.array([0,0])
    external_forces = np.array([0,0])

    # Solve the differential equation to see the rate of change
    rate_of_change = solve_ivp(model.forward, t_span, y_0, t_eval=t_eval, args=(tau, external_forces))

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(rate_of_change.t, rate_of_change.y[0], label='Rate of change for Joint 1')
    plt.plot(rate_of_change.t, rate_of_change.y[1], label='Rate of change for Joint 2')
    plt.title('Rate of Change for Accelaration Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle in Radians')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Experiment 2

    # Variables for Experiment 2
    angle_list = [0, 0]
    gravity_value = 9.81

    tau_1, tau_2 = experiment_2(mass_list, length_list, inertia_list, angle_list, gravity_value)
    print("Experiment for when external forces and general variables are 0 and the robotics manipulator is at rest at the beginning")
    print(f'Torque_1: {tau_1} Nm')
    print(f'Torque_2: {tau_2} Nm')


    # Experiment 3

    # Initialize the model
    model = DynamicModel(length_list, mass_list, inertia_list, friction_list)

    # Variables for Experiment 3
    angle_list = [np.deg2rad(60), np.deg2rad(-150)]
    gravity_matrix = model.gravity_matrix(angle_list)
    external_forces = np.array([0, 100])
    tau = experiment_3(angle_list, jacobian_matrix, external_forces, gravity_matrix, length_list)
    print("Experiment for when external forces are applied to the robotics manipulator")
    print(f'Torque_1: {tau[0]} Nm') 
    print(f'Torque_2: {tau[1]} Nm')
    
