import numpy as np
import datetime
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
import os
import glob

############################################
# Select vehicle, environment and scenario
############################################
config_selection = {"vehicle": "XC40",
                    "environment": "flat",
                    "simulation": "stop_brake_early",
                    "battery": "HVBATT1",
                    "machine": "PM"
                    }

# environment = "flat_downhill_flat"
# environment = "flat_downhill_uphill_flat"
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
script_dir = os.path.dirname(os.path.abspath(__file__))
config_files = glob.glob(os.path.join(script_dir, "*.json"))
config_data = {}

for file_path in config_files:
    # Extract the filename without the extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Read the JSON data from the file
    with open(file_path, 'r') as json_file:
        data = json.load(json_file).get(config_selection[file_name])

    # Store the JSON data in the dictionary
    config_data[file_name] = data

############################################
# Assign parameters from config
############################################
road_gradient_profile_distance = config_data["environment"]["road_gradient_profile_distance"]  # km
road_gradient_profile = config_data["environment"]["road_gradient_profile"]  # degrees, positive is uphill
road_gradient_function = interp1d(
    road_gradient_profile_distance,
    road_gradient_profile,
    kind="linear",
    fill_value="extrapolate",
)

mass = config_data["vehicle"]["mass"]
drag_coefficient = config_data["vehicle"]["drag_coefficient"]
frontal_area = config_data["vehicle"]["frontal_area"]
machine_efficiency = config_data["machine"]["efficiency"]
switch_loss = config_data["machine"]["switch_loss"]  # W
wheel_radius = config_data["vehicle"]["wheel_radius"]
wheel_circm = wheel_radius * 2 * np.pi
gear_ratio = config_data["machine"]["gear_ratio"]
max_machine_torque = config_data["machine"]["max_torque"]
max_machine_mech_power = config_data["machine"]["max_mech_power"] * 1000    # W
min_machine_torque = config_data["machine"]["min_torque"]
min_machine_mech_power = config_data["machine"]["min_mech_power"] * 1000    # W
max_battery_power = config_data["battery"]["max_power"] * 1000    # W
min_battery_power = config_data["battery"]["min_power"] * 1000    # W

max_distance = min(
    config_data["simulation"]["max_distance"] * 1000, road_gradient_profile_distance[-1] * 1000
)
max_time = config_data["simulation"]["max_time"]
initial_speed = config_data["simulation"]["initial_speed"] / 3.6

min_speed_profile = config_data["simulation"]["min_speed"]
min_speed_distance = config_data["simulation"]["min_speed_distance"]    #km 
if isinstance(min_speed_profile,(list)):
    min_speed_function = interp1d(
        config_data["simulation"]["min_speed_distance"],    #km
        config_data["simulation"]["min_speed"],
        kind="linear",
        fill_value="extrapolate",
    )
else:
    def min_speed_function(x):
        return np.array(min_speed_profile)

max_speed_profile = config_data["simulation"]["max_speed"]
max_speed_distance = config_data["simulation"]["max_speed_distance"]    #km
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
def get_max_force(speed):
    # Add battery power here later (with machine efficiency)
    max_wheel_torque = min(max_machine_torque * gear_ratio, max_machine_mech_power / (speed / wheel_circm * np.pi))
    max_force = max_wheel_torque / wheel_radius
    return max_force

def get_min_force(speed):
    # Add battery power here later (with machine efficiency)
    min_wheel_torque = max(min_machine_torque * gear_ratio, min_machine_mech_power / (speed / wheel_circm * np.pi))
    min_force = min_wheel_torque / wheel_radius
    return min_force

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

def get_driver_force(speed, road_load, speed_max, speed_min, prev_force):
    global driver_integral
    global driver_integral_gain
    global driver_propotional_gain
    global driver_force_filt_const

    max_force = get_max_force(speed)
    min_force = get_min_force(speed)

    if speed <= speed_min:
        force = road_load + driver_propotional_gain * (speed_min - speed)
    elif speed >= speed_max:
        force = road_load + driver_propotional_gain * (speed_max - speed)
    else:
        force = 0
    force_filt = LP(force, prev_force, driver_force_filt_const)
    return max(min(force_filt, max_force), min_force)


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

print("Total distance: " + str(max_distance/1000) + " km")
progress_bar = tqdm(total=max_distance, desc="Driven distance")

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
    battery_power = max(machine_power + switching_power, min_battery_power)     # Friction fillin braking assumed

    # Energy
    machine_energy += (machine_power + switching_power) * time_step
    switching_energy += switching_power * time_step
    battery_energy += battery_power * time_step

    # Vehicle movement
    acceleration = net_force / mass
    speed += acceleration * time_step
    distance_step = speed * time_step
    distance += distance_step

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

    progress_bar.update(distance_step)

progress_bar.close()

############################################
# Plot and print
############################################
# Convert for plot
# energies2 = np.array(energies) / (3600 * 1000)  # J -> kWh
battery_energies_plot = [x / (3600 * 1000) for x in battery_energies]   # J -> kWh
battery_powers_plot = [x / 1000 for x in battery_powers]    # W -> kW
switching_energies = np.array(switching_energies) / (3600 * 1000)  # J -> kWh

average_speed = (distances[-1] / 1000) / (times[-1] / 3600)

result_string = "Battery energy consumption: " + str(battery_energies_plot[-1]) + " kWh"
average_speed_string = "Average speed: " + str(average_speed * 3.6) + " km/h"
print(average_speed_string)
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
    file.write(average_speed_string + "\n" + result_string + "\n" + config_selection["vehicle"] + 
               "\n" + config_selection["simulation"] + "\n" + config_selection["environment"] + "\n")

############################################
# Show plot
############################################

plt.show()