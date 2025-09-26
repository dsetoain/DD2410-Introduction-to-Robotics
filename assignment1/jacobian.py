import matplotlib.pyplot as plt
import numpy as np

class DOFMachine():
    """ This class defines a planar 3 DOF machine with 3 links of lengths length_1, length_2 and length_3."""

    def __init__(self,length_1,length_2,length_3):
        """ 
            Constructor that initailizes the lengths of the links of the machine.

            params: length_1, length_2, length_3: lengths of the links
        """
        self.length_1 = length_1
        self.length_2 = length_2
        self.length_3 = length_3
    
    def velocity(self, theta_1, theta_2, theta_3):
        """ 
            This function calculates the end effector position and orientation.

            params: theta_1, theta_2, theta_3: angles of the joints

            return: x, y, orientation: position and orientation of the end effector
        """
        x_position = (self.length_1 * np.cos(theta_1)) + (self.length_2 * np.cos(theta_1 + theta_2)) + (self.length_3 * np.cos(theta_1 + theta_2 + theta_3))

        y_position = (self.length_1 * np.sin(theta_1)) + (self.length_2 * np.sin(theta_1 + theta_2)) + (self.length_3 * np.sin(theta_1 + theta_2 + theta_3))

        orientation = theta_1 + theta_2 + theta_3

        return x_position, y_position, orientation
    
    def jacobian_calculation(self, theta_1, theta_2, theta_3):
        """"
            This function calculates the Jacobian matrix based on the formula we generated in the Exercise 6.

            params: theta_1, theta_2, theta_3: angles of the joints

            return: Jacobian matrix
        """

        x_1 = -self.length_1 * np.sin(theta_1) - self.length_2 * np.sin(theta_1 + theta_2) - self.length_3 * np.sin(theta_1 + theta_2 + theta_3)

        x_2 = -self.length_2 * np.sin(theta_1 + theta_2)  - self.length_3 * np.sin(theta_1 + theta_2 + theta_3)

        x_3 = -self.length_3 * np.sin(theta_1 + theta_2 + theta_3)

        y_1 = (self.length_1 * np.cos(theta_1)) + (self.length_2 * np.cos(theta_1 + theta_2)) + (self.length_3 * np.cos(theta_1 + theta_2 + theta_3))

        y_2 = (self.length_2 * np.cos(theta_1 + theta_2)) + (self.length_3 * np.cos(theta_1 + theta_2 + theta_3))

        y_3 = (self.length_3 * np.cos(theta_1 + theta_2 + theta_3))

        return np.array([[x_1, x_2, x_3], [y_1, y_2, y_3], [1, 1, 1]])
    
    
    def inverse_jacobian(self, pose_vector, theta_vector, max_iterations=1000):
        """"
            This function calculates the inverse Jacobian and updates the joint angles to reach the desired pose.

            params: pose_vector: desired pose of the end effector
                    theta_vector: initial joint angles
                    max_iterations: maximum number of iterations

            return: result_theta_list: list of joint angles at each iteration
        """

        # Initalize the iteration count as 0
        iterations = 0

        # Initalize an array to store the resulted joint angles at each iteration
        result_theta_list = []
        
        # Loop until the maximum number of iterations is reached
        while iterations < max_iterations:
            
            # Get the theta (angle) values from the vector
            theta_1 = theta_vector[0]
            theta_2 = theta_vector[1]
            theta_3 = theta_vector[2]

            # Calculate the end effector position and orientation
            x, y, orientation = self.velocity(theta_1, theta_2, theta_3)

            # Create a vector of the current pose
            current_pose_vector = np.array([x, y, orientation])

            # Calculate the error vector by using the formula from the slides
            space_error = np.subtract(pose_vector, current_pose_vector)

            # Check if there is a convergence
            if np.linalg.norm(space_error) < 1e-6:
                print("There is a convergence\n")
                break
            
            # Calculate the Jacobian matrix
            jacobian = self.jacobian_calculation(theta_1, theta_2, theta_3)

            # Calculate inverse Jacobian by using Moore Penrose Pseudo Inverse method
            inverse_jacobian =  np.linalg.pinv(jacobian)

            # Calculate the positive definite matrix K
            k_matrix = np.dot(0.1, np.eye(3))

            # Calculate the difference of angles
            delta_theta = (k_matrix @ (inverse_jacobian @ space_error))

            # Update the joint angles
            theta_vector = theta_vector + delta_theta

            # Append the resulted joint angles to the list
            result_theta_list.append(theta_vector.copy())

            # Increase the iteration count
            iterations += 1

        # Return the list of joint angles at each iteration   
        return result_theta_list
    
    def plot_trajectories(self, theta_list):

        # Convert the theta list to numpy array
        theta_list = np.array(theta_list)

        # Get the three different values of the joint angles
        theta_1 = theta_list[:,0]
        theta_2 = theta_list[:,1]
        theta_3 = theta_list[:,2]

        # Create a list of x axis of the table which are the number of iterations
        x_axis = np.arange(0, len(theta_list), 1)
        
        # Create a plot
        plt.figure(figsize=(10, 10))

        # Give the title of the plot
        plt.title("Plot of Trajectories")

        # Give the name of x and y axis
        plt.xlabel("Number of Iterations")
        plt.ylabel("Angles in radians")

        # Plot the angles
        plt.plot(x_axis, theta_1, label="Theta 1")
        plt.plot(x_axis, theta_2, label="Theta 2")
        plt.plot(x_axis, theta_3, label="Theta 3")

        # Show the legend
        plt.legend()

        # Save the plot
        plt.savefig("trajectories.png")

        # Show the plot
        plt.show()

        # Close the plot
        plt.close()





if __name__ == "__main__":

    pose_vector = np.array([2*np.sqrt(2), (2*np.sqrt(2)) + 1, np.deg2rad(90)])

    machine = DOFMachine(2,2,1)

    initial_degrees = np.array([0, 0, 0])

    theta_history = machine.inverse_jacobian(pose_vector, initial_degrees)

    if theta_history:
        machine.plot_trajectories(theta_history)
    else:
        print("No solution found!\n")