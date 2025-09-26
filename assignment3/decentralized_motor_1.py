import numpy as np
import matplotlib.pyplot as plt
import control

class DecentralizedDCMotor:
    
    def __init__(self, inertia, torque_electric_constant, viscous_coeff, inductor, resistance):
        """"
            The constructor of the class.
            params:
                inertia: the inertia of the rotor.
                torque_electric_constant: torque constant (K_m) and electric (K_e) constant.
                                          Because they are same value, we are using the same variable.
                viscous_coeff: rotor's viscous coefficient.
                inductor: float, the inductor of the motor.
                resistance: float, the resistance of the motor.
        """
        self.inertia = inertia
        self.torque_electric_constant = torque_electric_constant
        self.viscous_coeff = viscous_coeff
        self.inductor = inductor
        self.resistance = resistance

    def transferFunction(self, external_load):
        """"
            Similar to MATLAB example from the lecture slides, this function returns the G_w system by using transfer function.

            To calculate the transfer function, we need to use the following formula:
            G_w(s) := (K_m) / ((I_m * L)* s^2  + (R * I_m + d_m * L)*s + (d_m*R + K_e * K_m +  K_e * external_load))

            However, because we are using TransferFunction() method from the control library, 
            we do not need to use time step s in the denominator.
            
            return: the transfer function of the system.
        """

        # Define the numerator in the shape of a list
        numerator = [self.torque_electric_constant]

        # Define the denominator in the shape of a list
        denominator = [(self.inertia * self.inductor), 
                       (self.resistance * self.inertia + self.viscous_coeff * self.inductor), 
                       (self.viscous_coeff * self.resistance + self.torque_electric_constant * self.torque_electric_constant + self.torque_electric_constant * external_load)]

        # Define the system with the TransferFunction() method
        sys_w = control.TransferFunction(numerator, denominator)

        return sys_w
    
    def piControllerwithVelocityLimit(self, max_velocity, kp, ki, external_load):
        """"
            This function implements the PI controller with velocity limit.
            
        """
        
        # Get the G
        G = self.transferFunction(external_load)

        # Calculate the PI-controller by using the PI formula: C(s) = K_p + K_i / s
        C = control.TransferFunction([kp, ki], [1, 0])

        # Calculate the closed loop system 
        closed_loop = control.feedback(C*G)

        # Define the time vector
        time = np.linspace(0, 10, 20000)

        # Calculate the step response of the closed loop system
        # By multiplying the closed loop with max velocity , we scale the step response by max velocity.
        output_time_vector, angular_velocities = control.step_response(closed_loop*max_velocity, time)

        # Calculate the angular positions by integrate the angular velocities
        angular_positions = np.cumsum(angular_velocities) * (output_time_vector[1] - output_time_vector[0])

        # Calculate the voltage vector
        error = max_velocity - angular_velocities
        integral_error = np.cumsum(error) * (output_time_vector[1] - output_time_vector[0])
        voltage_vector = kp * error + ki * integral_error


        return angular_velocities, angular_positions, voltage_vector, output_time_vector
    
    def piControllerwithPositionLimit(self, max_position, kp, ki, external_load):
        """
            This function implements the PI controller with position limit.
            params:
                max_position: the maximum position.
                kp: the proportional gain.
                ki: the integral gain.
                external_load: the external load.
        """

        G = self.transferFunction(external_load)

        # Initialize the Laplace variable
        s = control.TransferFunction([1, 0], [1])

        # Initialize G_theta => Transfer function from voltage to position
        G_theta = G / s

        # Initialize the PI Controller
        C = control.TransferFunction([kp, ki], [1, 0])

        # Close loop system
        close_loop = control.feedback(C * G_theta)

        time = np.linspace(0, 10, 20000)

        # Calculate the step response of the closed loop system
        output_time_vector, angular_positions = control.step_response(close_loop*max_position, time)

        # Compute the angular velocities
        angular_velocities = np.gradient(angular_positions, output_time_vector)

        # Calculate the voltage vector
        error = max_position - angular_positions
        integral_error = np.cumsum(error) * (output_time_vector[1] - output_time_vector[0])
        voltage_vector = kp * error + ki * integral_error

        return angular_velocities, angular_positions, voltage_vector, output_time_vector
    
    def transferFunction_current(self):
        """
            Returns the transfer function from voltage to current: I(s)/V(s)
            G_I(s) = 1 / (L*s + R)
        """
        numerator = [1]
        denominator = [self.inductor, self.resistance]
        G_I = control.TransferFunction(numerator, denominator)
        return G_I

    def transferFunction_velocity(self):
        """
            Returns the transfer function from current to angular velocity: ω(s)/I(s)
            G_omega(s) = K_m / (J*s + d_m)
        """
        numerator = [self.torque_electric_constant]
        denominator = [self.inertia, self.viscous_coeff]
        G_omega = control.TransferFunction(numerator, denominator)
        return G_omega

    def transferFunction_position(self):
        """
            Returns the transfer function from angular velocity to position: θ(s)/ω(s)
            G_theta(s) = 1 / s
        """
        numerator = [1]
        denominator = [1, 0]
        G_theta = control.TransferFunction(numerator, denominator)
        return G_theta

    


def experiment_1(dc_motor, kp, ki, external_load:float=0.0):
    """
        This function is the first experiment of the assignment.
        params:
            dc_motor: the motor object.
            external_load: the external load.
    """

    # Define the maximum velocity
    max_velocity = 10

    # The velocity should not be overshoot
    while True:
        # Calculate the PI controller with velocity limit
        velocities, positions, voltages, time = dc_motor.piControllerwithVelocityLimit(max_velocity, kp, ki, external_load)

        final_steady_vector = velocities[-1]

        max_velocity_calc = np.max(velocities)

        if final_steady_vector != 0 and not np.isnan(final_steady_vector):
            overshoot = ((max_velocity_calc - final_steady_vector) / final_steady_vector) * 100
        else:
            overshoot = 0  # or handle appropriately

        # No overshoot
        if overshoot > 0:
            break
        else:
            #Otherwise find the best solution
            ki += 0.1
            kp += 0.01


    # Plot the results
    plt.figure(figsize=(8, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, velocities)
    plt.title(f"Angular Velocities with K_i = {ki} and K_p is {kp}")    
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocities (rad/s)")

    plt.subplots_adjust(hspace=0.5)

    plt.subplot(3, 1, 2)
    plt.plot(time, positions)
    plt.title(f"Angular Positions with K_i = {ki} and K_p is {kp}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Positions (rad)")

    plt.subplot(3, 1, 3)
    plt.plot(time, voltages)
    plt.title(f"Voltage with K_i = {ki} and K_p is {kp}")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(f"plots/question1_a/angular_positions_ki_{ki}_kp_{kp}.png")


    max_voltage = np.max(voltages)
    print("Max Voltage: ", max_voltage)

def experiment_2(dc_motor, kp, ki, external_load:float=0.0):
    """
        This function is the second experiment of the assignment.
        params:
            dc_motor: the motor object.
            external_load: the external load.
    """

    # Define the maximum velocity
    max_velocity = 10

    while True:
        
        # Calculate the PI controller with velocity limit
        velocities, positions, voltages, time = dc_motor.piControllerwithVelocityLimit(max_velocity, kp, ki, external_load)

        # Get steady state value
        last_steady_state = velocities[-1]
        last_steady_state = np.mean(last_steady_state)

        # Get maximum velocity got in the experiment
        max_velocity_got = np.max(velocities)
        
        if last_steady_state != 0 and not np.isnan(last_steady_state):
            overshoot = ((max_velocity_got - last_steady_state) / last_steady_state) * 100
        else:
            overshoot = 0  # or handle appropriately



        # If the overshoot is more than 50%, break the loop
        if overshoot >= 50.00:
            break
        else:
            # Increase the ki and kp values
            ki += 0.1
            kp += 1.0

    print(f"Overshoot: {overshoot} in time {time}%")
        # Plot the results
    plt.figure(figsize=(8, 6))

    plt.subplot(3, 1, 1)
    plt.plot(time, velocities)
    plt.title(f"Angular Velocities with K_i = {ki} and K_p is {kp}")    
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocities (rad/s)")

    plt.subplots_adjust(hspace=0.5)

    plt.subplot(3, 1, 2)
    plt.plot(time, positions)
    plt.title(f"Angular Positions with K_i = {ki} and K_p is {kp}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Positions (rad)")

    plt.subplot(3, 1, 3)
    plt.plot(time, voltages)
    plt.title(f"Voltage with K_i = {ki} and K_p is {kp}")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(f"plots/question1_b/__angular_positions_ki_{ki}_kp_{kp}_overshoot{overshoot}.png")


    max_voltage = np.max(voltages)
    print("Max Voltage: ", max_voltage)

def experiment3(dc_motor, kp, ki, external_load):
    
    max_position = 2

    while True:
        # Calculate the PI controller with position limit
        velocities, positions, voltages, time = dc_motor.piControllerwithPositionLimit(max_position, kp, ki, external_load)

        # Get steady state value
        last_steady_state = positions[-1]
        last_steady_state = np.mean(last_steady_state)

        # Get maximum velocity got in the experiment
        max_position_got = np.max(positions)
        
        if last_steady_state != 0 and not np.isnan(last_steady_state):
            overshoot = ((max_position_got - last_steady_state) / last_steady_state) * 100
        else:
            overshoot = 0

        # If there is an overshoot break the loop
        if overshoot > 0.00:
            break
        else:
            # Increase the ki and kp values
            ki += 0.1
            kp += 1.0
    

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(time, positions)
    plt.title(f"Angular Positions with K_i = {ki} and K_p is {kp}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Positions (rad)")

    plt.subplots_adjust(hspace=0.5)

    plt.subplot(3, 1, 2)
    plt.plot(time, velocities)
    plt.title(f"Angular Velocities with K_i = {ki} and K_p is {kp}")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Velocities (rad/s)")

    plt.subplot(3, 1, 3)
    plt.plot(time, voltages)
    plt.title(f"Voltage with K_i = {ki} and K_p is {kp}")
    plt.xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(f"plots/question1_c/angular_positions_ki_{ki}_kp_{kp}_overshoot{overshoot}.png")


def experiment4(dc_motor, kp_p, ki_p, kp_v, ki_v, kp_i, ki_i, external_load=0.0):
    """
        This function is the fourth experiment of the assignment.
        params:
            dc_motor: the motor object.
            kp_p: the proportional gain of the position controller.
            ki_p: the integral gain of the position controller.
            kp_v: the proportional gain of the velocity controller.
            ki_v: the integral gain of the velocity controller.
            kp_i: the proportional gain of the current controller.
            ki_i: the integral gain of the current controller.
            external_load: the external load.
    """

    # Desired position
    max_position = 1.0

    # Time parameters
    dt = 0.001
    t_end = 5
    time = np.arange(0, t_end, dt)

    # Initialize variables
    theta = 0.0
    omega = 0.0
    current = 0.0
    voltage = 0.0

    # Initialize the errors
    current_error = 0.0
    velocity_error = 0.0
    position_error = 0.0

    # Initalize the lists for the results
    positions = []
    velocities = []
    currents = []
    voltages = []

    # Loop over the time and calculate three loops: outer, middle and inner loop
    for t in time:

        # Position control
        position_error = max_position - theta
        position_error_integral = np.sum(position_error) * dt
        omega_ref = kp_p * position_error + ki_p * position_error_integral

        # Velocity control
        velocity_error = omega_ref - omega
        velocity_error_integral = np.sum(velocity_error) * dt
        current_ref = kp_v * velocity_error + ki_v * velocity_error_integral

        # Limit the current based on the question
        current_ref = np.clip(current_ref, -60.0, 60.0)

        # Current control
        current_error = current_ref - current
        current_error_integral = np.sum(current_error) * dt
        voltage = kp_i * current_error + ki_i * current_error_integral

        # Electrical equation
        current_derivative = (voltage - dc_motor.resistance * current - dc_motor.torque_electric_constant * omega) / dc_motor.inductor
        current += current_derivative * dt

        # Mechanical equation
        omega_derivative = (dc_motor.torque_electric_constant * current - dc_motor.viscous_coeff * omega - external_load) / dc_motor.inertia
        omega += omega_derivative * dt

        # Update the position
        theta += omega * dt

        # Append the results
        positions.append(theta)
        velocities.append(omega)
        currents.append(current)
        voltages.append(voltage)

    
    # Converting plots to the numpy arrays
    positions = np.array(positions)
    velocities = np.array(velocities)
    currents = np.array(currents)
    voltages = np.array(voltages)

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time, positions, label='Angular Position')
    plt.axhline(max_position, color='r', linestyle='--', label='Desired Position')
    plt.title("Angular Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time, velocities, label='Angular Velocity')
    plt.title("Angular Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time, voltages, label='Voltage')
    plt.axhline(60 * dc_motor.resistance, color='r', linestyle='--', label='Voltage Limit')
    plt.axhline(-60 * dc_motor.resistance, color='r', linestyle='--')
    plt.title("Voltage")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("plots/question1_d/cascaded_control_position.png")
    

if __name__ == "__main__":

    # Electrical and mechanical parameters of the DC motor.
    inertia = 0.02 #I_m
    torque_electric_constant = 0.1 #K_m = K_e
    viscous_coeff = 0.1 #d_m
    inductor = 0.1 #L
    resistance = 0.3 #R

    # Create the motor object.
    dc_motor = DecentralizedDCMotor(inertia, torque_electric_constant, viscous_coeff, inductor, resistance)

    # Experiment 1
    #experiment_1(dc_motor, kp=0.0, ki=0.0, external_load=0.0)

    # Experiment 2
    #experiment_2(dc_motor, external_load=0.0,kp=0.00, ki=0.00)

    # Experiment 3
    #experiment3(dc_motor, external_load=1.0, kp=0.00, ki=0.00)

    # Experiment 4
experiment4(dc_motor, kp_p=10.0, ki_p=0.0, kp_v=10.0, ki_v=1.0, kp_i=5.0, ki_i=1000.0, external_load=0.0)

            