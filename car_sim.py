import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d

############################################
# Select vehicle, environment and scenario
############################################
vehicle = "XC40"
# environment = "flat_downhill_flat"
# environment = "flat_downhill_uphill_flat"
environment = "flat"
simulation = "stop_brake_early"
# simulation = "stop_brake_late"
# simulation = "constant_speed_low_speed"
# simulation = "allow_overshoot"
# simulation = "allow_small_overshoot"
# simulation = "allow_small_overshoot_low_speed"

############################################
# Global constants
############################################
gravity = 9.81
time_step = 0.01
driver_force_filt_const = 0.05

############################################
# Global variables
############################################
driver_force = 0
driver_integral = 0
driver_integral_gain = 10
driver_propotional_gain = 10000

############################################
# Open and read config files
############################################
p = Path(__file__).with_name("vehicles.json")
with p.open("r") as f:
    vehicle_config = json.load(f).get(vehicle)
p = Path(__file__).with_name("simulations.json")
with p.open("r") as f:
    simulation_config = json.load(f).get(simulation)
p = Path(__file__).with_name("environments.json")
with p.open("r") as f:
    environment_config = json.load(f).get(environment)

############################################
# Assign parameters from config
############################################
road_gradient_profile_distance = environment_config[
    "road_gradient_profile_distance"
]  # km
road_gradient_profile = environment_config[
    "road_gradient_profile"
]  # degrees, positive is uphill
road_gradient_function = interp1d(
    road_gradient_profile_distance,
    road_gradient_profile,
    kind="linear",
    fill_value="extrapolate",
)

mass = vehicle_config["mass"]
drag_coefficient = vehicle_config["drag_coefficient"]
frontal_area = vehicle_config["frontal_area"]
machine_efficiency = vehicle_config["machine_efficiency"]
switch_loss = vehicle_config["switch_loss"]  # W

max_distance = min(
    simulation_config["max_distance"] * 1000, road_gradient_profile_distance[-1] * 1000
)
max_time = simulation_config["max_time"]
initial_speed = simulation_config["initial_speed"] / 3.6

min_speed_profile = simulation_config["min_speed"]
min_speed_distance = simulation_config["min_speed_distance"]    #km 
if isinstance(min_speed_profile,(list)):
    min_speed_function = interp1d(
        simulation_config["min_speed_distance"],    #km
        simulation_config["min_speed"],
        kind="linear",
        fill_value="extrapolate",
    )
else:
    def min_speed_function(x):
        return np.array(min_speed_profile)

max_speed_profile = simulation_config["max_speed"]
max_speed_distance = simulation_config["max_speed_distance"]    #km
if isinstance(max_speed_profile,(list)):
    max_speed_function = interp1d(
        max_speed_distance,
        max_speed_profile,
        kind="linear",
        fill_value="extrapolate",
    )
else:
    def max_speed_function(x):
        return np.array(max_speed_profile)


############################################
# Define functions
############################################
def LP(new_value, prev_value, filter_constant):
    global time_step
    return time_step / max(filter_constant, time_step) * (new_value - prev_value) + prev_value

def get_road_gradient(distance):
    """
    Output: road gradient (rad)
    Input: distance (m)"""
    return road_gradient_function(distance / 1000).item()

def get_max_speed(distance):
    return max_speed_function(distance/1000).item() / 3.6

def get_min_speed(distance):
    return min_speed_function(distance/1000).item() / 3.6

def get_road_load(road_gradient, speed):
    """
    Output: current road load force (N)
    Input: distance (m) and speed (m/s)
    """
    air_resistance_force = 0.5 * drag_coefficient * frontal_area * speed**2
    road_gradient_force = mass * gravity * np.sin(road_gradient * np.pi / 180)
    rolling_resistance_force = 300
    return air_resistance_force + rolling_resistance_force + road_gradient_force

"""
def get_driver_force(speed, road_load, speed_max, speed_min):
    global driver_propotional_gain
    global driver_integral_gain
    global driver_integral
    if speed_max == speed_min:
        force = road_load
    elif speed > speed_max:
        driver_integral += driver_integral_gain * (speed_max - speed)
        force = driver_integral + driver_propotional_gain * (speed_max - speed)
    elif speed < speed_min:
        driver_integral += driver_integral_gain * (speed_min - speed)
        force = driver_integral + driver_propotional_gain * (speed_min - speed)
    else:
        driver_integral = max(0, abs(driver_integral)) - 1 * np.sign(driver_integral)
        force = driver_integral
    return force
"""

def get_driver_force(speed, road_load, speed_max, speed_min, prev_force):
    global driver_integral
    global driver_integral_gain
    global driver_propotional_gain
    global driver_force_filt_const

    if speed <= speed_min:
        force = road_load + driver_propotional_gain * (speed_min - speed)
    elif speed >= speed_max:
        force = road_load + driver_propotional_gain * (speed_max - speed)
    else:
        force = 0
    force_filt = LP(force, prev_force, driver_force_filt_const)
    return force_filt


def get_machine_power(speed, driver_force):
    return speed * driver_force / machine_efficiency


############################################
# Run simulation
############################################

speed = initial_speed
energy = 0
machine_energy = 0
switching_energy = 0
distance = 0
time = 0
road_gradient = 0
road_load = 0
driver_force = 0
elevation = 0
battery_power = 0
battery_energy = 0

speeds = []
max_speeds = []
min_speeds = []
battery_energies = []
distances = []
times = []
gradients = []
road_loads = []
driver_forces = []
switching_energies = []
elevations = []
battery_powers = []

# Initialize driver integral
road_gradient = get_road_gradient(distance)
road_load = get_road_load(road_gradient, speed)
driver_integral = road_load

while distance < max_distance and time < max_time and speed > 0.1:
    road_gradient = get_road_gradient(distance)
    road_load = get_road_load(road_gradient, speed)
    max_speed = get_max_speed(distance)
    min_speed = get_min_speed(distance)
    driver_force = get_driver_force(speed, road_load, max_speed, min_speed, driver_force)
    net_force = driver_force - road_load

    # Power
    machine_power = get_machine_power(speed, driver_force)
    if abs(driver_force) > 0.1:
        switching_power = switch_loss
    else:
        switching_power = 0
    
    # Energy
    machine_energy += (machine_power + switching_power) * time_step
    switching_energy += switching_power * time_step
    battery_power = machine_power + switching_power
    battery_energy += battery_power * time_step

    # Vehicle movement
    acceleration = net_force / mass
    speed += acceleration * time_step
    distance += speed * time_step

    time += time_step

    elevation += np.arctan(road_gradient * np.pi / 180) * speed * time_step

    if elevation < -1:
        hej = "debug"

    speeds.append(speed * 3.6)
    max_speeds.append(max_speed * 3.6)
    min_speeds.append(min_speed * 3.6)
    battery_powers.append(battery_power)
    battery_energies.append(battery_energy)
    distances.append(distance)
    times.append(time)
    gradients.append(road_gradient)
    road_loads.append(road_load)
    driver_forces.append(driver_force)
    switching_energies.append(switching_energy)
    elevations.append(elevation)


############################################
# Plot and print
############################################
# Convert for plot
# energies2 = np.array(energies) / (3600 * 1000)  # J -> kWh
battery_energies_plot = [x / (3600 * 1000) for x in battery_energies]   # J -> kWh
battery_powers_plot = [x / 1000 for x in battery_powers]    # W -> kW
switching_energies = np.array(switching_energies) / (3600 * 1000)  # J -> kWh

result_string = "Battery energy consumption: " + str(battery_energies_plot[-1]) + " kWh"
print(result_string)

plt.figure(figsize=(17, 10))

plt.subplot(4, 3, 1)
plt.plot(times, gradients)
plt.xlabel("Time (s)")
plt.ylabel("Road Gradient (deg)")
plt.grid()

plt.subplot(4, 3, 2)
plt.plot(times, speeds)
plt.plot(times, max_speeds, linestyle='--')
plt.plot(times, min_speeds, linestyle='--')
plt.xlabel("Time (s)")
plt.ylabel("vehicle Speed (km/h)")
plt.grid()

plt.subplot(4, 3, 3)
plt.plot(times, battery_energies_plot)
plt.xlabel("Time (s)")
plt.ylabel("Battery Energy (kWh)")
plt.grid()

plt.subplot(4, 3, 4)
plt.plot(times, distances)
plt.xlabel("Time (s)")
plt.ylabel("Distance Traveled (m)")
plt.grid()

plt.subplot(4, 3, 5)
plt.plot(times, road_loads)
plt.xlabel("Time (s)")
plt.ylabel("Road Load (N)")
plt.grid()

plt.subplot(4, 3, 6)
plt.plot(times, driver_forces)
plt.xlabel("Time (s)")
plt.ylabel("Driver Force (N)")
plt.grid()

plt.subplot(4, 3, 7)
plt.plot(times, switching_energies)
plt.xlabel("Time (s)")
plt.ylabel("Switching Energy (kWh)")
plt.grid()

plt.subplot(4, 3, 8)
plt.plot(times, elevations)
plt.xlabel("Time (s)")
plt.ylabel("Elevation (m)")
plt.grid()

plt.subplot(4, 3, 9)
plt.plot(times, battery_powers_plot)
plt.xlabel("Time (s)")
plt.ylabel("Battery Power (kW)")
plt.grid()

plt.tight_layout()

############################################
# Save result
############################################

current_datetime = datetime.datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d %H_%M_%S")

p = Path(__file__).with_name(date_time_string + ".png")
plt.savefig(p)

p = Path(__file__).with_name(date_time_string + ".txt")
with open(p, 'w') as file:
    file.write(result_string + "\n" + vehicle + "\n" + simulation + "\n" + environment + "\n")

############################################
# Show plot
############################################

plt.show()