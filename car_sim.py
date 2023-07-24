import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

mass = 2100  # kg
drag_coefficient = 0.34  # XC40
frontal_area = 2.56  # m^2, XC40
gravity = 9.81  # m/s^2
efficiency = 0.9
time_step = 0.01  # seconds
simulation_distance = 3000  # m
simulation_max_time = 1000  # s
initial_speed = 25  # m/s
min_speed = 25
max_speed = 25
switch_loss = 300  # W

road_gradient_profile_dist = [0, 0.5, 1, 1.5, 2, 2.5]  # km
road_gradient_profile = [0, 0, -5, 0, 0, 0]  # degrees, positive is uphill
road_gradient_function = interp1d(
    road_gradient_profile_dist,
    road_gradient_profile,
    kind="linear",
    fill_value="extrapolate",
)


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
    elif speed > max_speed or speed < min_speed:
        return road_load
    else:
        return 0


def get_machine_power(speed, driver_force):
    return speed * driver_force / efficiency


# print(get_road_load(0, 10))


def sim():
    speed = initial_speed
    energy = 0
    machine_energy = 0
    switching_energy = 0
    distance = 0
    time = 0
    road_gradient = 0
    road_load = 0

    speeds = [speed]
    energies = [0]
    distances = [0]
    times = [0]
    gradients = [0]
    road_loads = [0]
    driver_forces = [0]
    switching_energies = [0]

    while distance < simulation_distance and time < simulation_max_time and speed > 0.1:
        road_gradient = get_road_gradient(distance)
        road_load = get_road_load(road_gradient, speed)
        driver_force = get_driver_force(speed, road_load)
        net_force = driver_force - road_load

        acceleration = net_force / mass
        speed += acceleration * time_step

        machine_energy += get_machine_power(speed, driver_force) * time_step
        if abs(driver_force) > 0.1:
            switching_energy += switch_loss * time_step
        energy += machine_energy + switching_energy

        distance += speed * time_step

        time += time_step

        speeds.append(speed)
        energies.append(energy)
        distances.append(distance)
        times.append(time)
        gradients.append(road_gradient)
        road_loads.append(road_load)
        driver_forces.append(driver_force)
        switching_energies.append(switching_energy)

    return (
        speeds,
        energies,
        distances,
        times,
        gradients,
        road_loads,
        driver_forces,
        switching_energies,
    )


# Run simulation
(
    speeds,
    energies,
    distances,
    times,
    gradients,
    road_loads,
    driver_forces,
    switching_energies,
) = sim()

# Convert for plot
# energies2 = np.array(energies) / (3600 * 1000)  # J -> kWh
energies2 = [x / (3600 * 1000) for x in energies]
switching_energies = np.array(switching_energies) / (3600 * 1000)  # J -> kWh

plt.figure(figsize=(17, 10))

plt.subplot(4, 2, 1)
plt.plot(times, gradients)
plt.xlabel("Time (s)")
plt.ylabel("Road Gradient (degrees)")
plt.grid()

plt.subplot(4, 2, 2)
plt.plot(times, speeds)
plt.xlabel("Time (s)")
plt.ylabel("Vehicle Speed (m/s)")
plt.grid()

plt.subplot(4, 2, 3)
plt.plot(times, energies2)
plt.xlabel("Time (s)")
plt.ylabel("Consumed Energy (kWh)")
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

plt.tight_layout()
plt.show()
