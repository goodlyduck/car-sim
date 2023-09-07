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
environment = "flat_downhill_uphill_flat"
simulation = "constant_speed"
# simulation = "constant_speed_low_speed"
# simulation = "allow_overshoot"
# simulation = "allow_small_overshoot"
# simulation = "allow_small_overshoot_low_speed"

############################################
# Global constants
############################################
gravity = 9.81
time_step = 0.01

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
efficiency = vehicle_config["efficiency"]
switch_loss = vehicle_config["switch_loss"]  # W

max_distance = min(
    simulation_config["max_distance"], road_gradient_profile_distance[-1] * 1000
)
max_time = simulation_config["max_time"]
initial_speed = simulation_config["initial_speed"]
min_speed = simulation_config["min_speed"]
max_speed = simulation_config["max_speed"]

############################################
# Define functions
############################################
def get_road_gradient(distance):
    """
    Output: road gradient (rad)
    Input: distance (m)"""
    return road_gradient_function(distance / 1000)


def get_road_load(road_gradient, speed):
    """
    Output: current road load force (N)
    Input: distance (m) and speed (m/s)
    """
    air_resistance_force = 0.5 * drag_coefficient * frontal_area * speed**2
    road_gradient_force = mass * gravity * np.sin(road_gradient * np.pi / 180)
    rolling_resistance_force = 300
    return air_resistance_force + rolling_resistance_force + road_gradient_force


def get_driver_force(speed, road_load):
    if max_speed == min_speed:
        return road_load
    elif speed > max_speed and road_load < 0:
        return road_load - 1
    elif speed < min_speed and road_load > 0:
        return road_load + 1
    else:
        return 0


def get_machine_power(speed, driver_force):
    return speed * driver_force / efficiency


def sim():
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

    speeds = [speed]
    energies = [0]
    distances = [0]
    times = [0]
    gradients = [0]
    road_loads = [0]
    driver_forces = [0]
    switching_energies = [0]
    elevations = [0]

    while distance < max_distance and time < max_time and speed > 0.1:
        road_gradient = get_road_gradient(distance)
        road_load = get_road_load(road_gradient, speed)
        driver_force = get_driver_force(speed, road_load)
        net_force = driver_force - road_load

        acceleration = net_force / mass
        speed += acceleration * time_step

        machine_energy += get_machine_power(speed, driver_force) * time_step
        if abs(driver_force) > 0.1:
            switching_energy += switch_loss * time_step
        energy = machine_energy + switching_energy

        distance += speed * time_step

        time += time_step

        elevation += np.arctan(road_gradient * np.pi / 180) * speed * time_step

        speeds.append(speed)
        energies.append(energy)
        distances.append(distance)
        times.append(time)
        gradients.append(road_gradient)
        road_loads.append(road_load)
        driver_forces.append(driver_force)
        switching_energies.append(switching_energy)
        elevations.append(elevation)

    return (
        speeds,
        energies,
        distances,
        times,
        gradients,
        road_loads,
        driver_forces,
        switching_energies,
        elevations,
    )

############################################
# Run simulation
############################################
(
    speeds,
    energies,
    distances,
    times,
    gradients,
    road_loads,
    driver_forces,
    switching_energies,
    elevations,
) = sim()

############################################
# Plot and print
############################################
# Convert for plot
# energies2 = np.array(energies) / (3600 * 1000)  # J -> kWh
energies2 = [x / (3600 * 1000) for x in energies]
switching_energies = np.array(switching_energies) / (3600 * 1000)  # J -> kWh

result_string = "Energy consumption: " + str(energies2[-1]) + " kWh"
print(result_string)

plt.figure(figsize=(17, 10))

plt.subplot(4, 2, 1)
plt.plot(times, gradients)
plt.xlabel("Time (s)")
plt.ylabel("Road Gradient (deg)")
plt.grid()

plt.subplot(4, 2, 2)
plt.plot(times, speeds)
plt.xlabel("Time (s)")
plt.ylabel("vehicle Speed (m/s)")
plt.grid()

plt.subplot(4, 2, 3)
plt.plot(times, energies2)
plt.xlabel("Time (s)")
plt.ylabel("Total Energy (kWh)")
plt.grid()

plt.subplot(4, 2, 4)
plt.plot(times, distances)
plt.xlabel("Time (s)")
plt.ylabel("Distance Traveled (m)")
plt.grid()

plt.subplot(4, 2, 5)
plt.plot(times, road_loads)
plt.xlabel("Time (s)")
plt.ylabel("Road Load (N)")
plt.grid()

plt.subplot(4, 2, 6)
plt.plot(times, driver_forces)
plt.xlabel("Time (s)")
plt.ylabel("Driver Force (N)")
plt.grid()

plt.subplot(4, 2, 7)
plt.plot(times, switching_energies)
plt.xlabel("Time (s)")
plt.ylabel("Switching Energy (kWh)")
plt.grid()

plt.subplot(4, 2, 8)
plt.plot(times, elevations)
plt.xlabel("Time (s)")
plt.ylabel("Elevation (m)")
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