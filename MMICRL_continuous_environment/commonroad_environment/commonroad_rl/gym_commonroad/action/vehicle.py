""" Module for managing the vehicle in the CommonRoad Gym environment
"""
import copy

import numpy as np
import commonroad_dc.pycrcc as pycrcc
from typing import List, Tuple
from abc import ABC, abstractmethod
from aenum import extend_enum
from scipy.optimize import Bounds
from commonroad.scenario.trajectory import State
from commonroad_rl.gym_commonroad.utils.scenario import make_valid_orientation
from commonroad.common.solution import VehicleModel, VehicleType
from commonroad_dc.collision.trajectory_queries import trajectory_queries
from commonroad_dc.feasibility.vehicle_dynamics import VehicleDynamics, FrictionCircleException
from vehiclemodels.vehicle_parameters import VehicleParameters
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.parameters_vehicle3 import parameters_vehicle3

N_INTEGRATION_STEPS = 100

extend_enum(VehicleModel, 'YawRate', len(VehicleModel))


# Using VehicleParameterMapping from feasibility checker causes bugs
def to_vehicle_parameter(vehicle_type: VehicleType):
    if vehicle_type == VehicleType.FORD_ESCORT:
        return parameters_vehicle1()
    elif vehicle_type == VehicleType.BMW_320i:
        return parameters_vehicle2()
    elif vehicle_type == VehicleType.VW_VANAGON:
        return parameters_vehicle3()
    else:
        raise TypeError(f"Vehicle type {vehicle_type} not supported!")


def assert_vehicle_model(vehicle_model: VehicleModel):
    if vehicle_model == VehicleModel.MB:
        raise NotImplementedError(f"Vehicle model {vehicle_model} is not implemented yet!")
    else:
        return vehicle_model


class Vehicle(ABC):
    """
    Description:
        Abstract base class of all vehicles
    """

    def __init__(self, params_dict: dict) -> None:
        """ Initialize empty object """
        vehicle_type = VehicleType(params_dict["vehicle_type"])
        vehicle_model = VehicleModel(params_dict["vehicle_model"])
        self.vehicle_type = vehicle_type
        self.vehicle_model = assert_vehicle_model(vehicle_model)
        self.parameters = to_vehicle_parameter(vehicle_type)
        self.name = None
        self.dt = None
        self._collision_object = None
        self.initial_state = None
        self.state_list = None

    @property
    def state(self) -> State:
        """
        Get the current state of the vehicle

        :return: The current state of the vehicle
        """
        return self.state_list[-1]

    @property
    def previous_state(self) -> State:
        """
        Get the previous state of the vehicle

        :return: The previous state of the vehicle
        """
        if len(self.state_list) > 1:
            return self.state_list[-2]
        else:
            return self.initial_state

    @state.setter
    def state(self, state: State):
        """ Set the current state of the vehicle is not supported """
        raise ValueError("To set the state of the vehicle directly is prohibited!")

    @property
    def collision_object(self) -> pycrcc.RectOBB:
        """
        Get the collision object of the vehicle

        :return: The collision object of the vehicle
        """
        return self._collision_object

    @collision_object.setter
    def collision_object(self, collision_object: pycrcc.RectOBB):
        """ Set the collision_object of the vehicle is not supported """
        raise ValueError("To set the collision_object of the vehicle directly is prohibited!")

    @property
    def current_time_step(self):
        return self.state.time_step

    @current_time_step.setter
    def current_time_step(self, current_time_step):
        raise ValueError("To set the current time step of the vehicle directly is prohibited!")

    def create_obb_collision_object(self, state: State):
        return pycrcc.RectOBB(self.parameters.l / 2,
                              self.parameters.w / 2,
                              state.orientation,
                              state.position[0],
                              state.position[1])

    def update_collision_object(self, create_convex_hull=True):
        """ Updates the collision_object of the vehicle """
        if create_convex_hull:
            self._collision_object = pycrcc.TimeVariantCollisionObject(self.previous_state.time_step)
            self._collision_object.append_obstacle(self.create_obb_collision_object(self.previous_state))
            self._collision_object.append_obstacle(self.create_obb_collision_object(self.state))
            self._collision_object, err = trajectory_queries.trajectory_preprocess_obb_sum(self._collision_object)
            if not err:
                return
            raise Exception("trajectory preprocessing error")
        else:
            self._collision_object = pycrcc.TimeVariantCollisionObject(self.state.time_step)
            self._collision_object.append_obstacle(self.create_obb_collision_object(self.state))

    @abstractmethod
    def set_current_state(self, new_state: State):
        """
        Update state list
        """
        raise NotImplementedError

    def reset(self, initial_state: State, dt: float) -> None:
        """
        Reset vehicle parameters.

        :param initial_state: The initial state of the vehicle
        :param dt: Simulation dt of the scenario
        :return: None
        """
        self.dt = dt
        if self.vehicle_model == VehicleModel.PM:
            orientation = initial_state.orientation if hasattr(initial_state, "orientation") else 0.0
            self.initial_state = State(**{"position": initial_state.position,
                                          "orientation": orientation,
                                          "time_step": initial_state.time_step,
                                          "velocity": initial_state.velocity * np.cos(orientation),
                                          "velocity_y": initial_state.velocity * np.sin(orientation),
                                          "acceleration": initial_state.acceleration * np.cos(orientation)
                                          if hasattr(initial_state, "acceleration") else 0.0,
                                          "acceleration_y": initial_state.acceleration * np.sin(orientation)
                                          if hasattr(initial_state, "acceleration") else 0.0})
        else:
            self.initial_state = State(**{"position": initial_state.position,
                                          "steering_angle": initial_state.steering_angle
                                          if hasattr(initial_state, "steering_angle")
                                          else 0.0,
                                          "orientation": initial_state.orientation
                                          if hasattr(initial_state, "orientation")
                                          else 0.0,
                                          "yaw_rate": initial_state.yaw_rate
                                          if hasattr(initial_state, "yaw_rate")
                                          else 0.0,
                                          "time_step": initial_state.time_step,
                                          "velocity": initial_state.velocity,
                                          "acceleration": initial_state.acceleration
                                          if hasattr(initial_state, "acceleration")
                                          else 0.0})
        self.state_list: List[State] = [self.initial_state]
        self.update_collision_object(create_convex_hull=self._continuous_collision_checking)

    def rescale_action(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Rescales the normalized action from [-1,1] to the required range

        :param normalized_action: action from the CommonroadEnv.
        :return: rescaled action
        """
        pass


class ContinuousVehicle(Vehicle):
    """
    Description:
        Class for vehicle when trained in continuous action space
    """

    def __init__(self, params_dict: dict, continuous_collision_checking=True):
        """ Initialize empty object """
        super().__init__(params_dict)
        self.violate_friction = False
        self.jerk_bounds = np.array([-10, 10])
        self._continuous_collision_checking = continuous_collision_checking

        try:
            self.vehicle_dynamic = VehicleDynamics.from_model(self.vehicle_model, self.vehicle_type)
        except:
            if self.vehicle_model == VehicleModel.YawRate:
                # customize YawRate VehicleModel
                self.vehicle_dynamic = self._vehicle_dynamics_yaw_rate(self.vehicle_type)
                self.parameters = self.vehicle_dynamic.parameters
            else:
                raise ValueError(f"Unknown vehicle model: {self.vehicle_model}")

    def _vehicle_dynamics_yaw_rate(self, vehicle_type):
        self.vehicle_model = VehicleModel.YawRate
        self.vehicle_type = vehicle_type

        return YawRateDynamics(vehicle_type)

    def set_current_state(self, new_state: State):
        """
        Update state list

        :param new_state: new state
        :return: None
        """
        self.state_list.append(new_state)
        self.update_collision_object(create_convex_hull=self._continuous_collision_checking)

    def propagate_one_time_step(self, current_state: State, action: np.ndarray, action_base: str) -> State:
        """Generate the next state from a given state for the given action.

        :param current_state: current state of vehicle to propagate from
        :param action: control inputs of vehicle (real input)
        :param action_base: aspect on which the action should be based ("jerk", "acceleration")
        :return: propagated state
        """
        if action_base == "acceleration":
            u_input = action
        elif action_base == "jerk":
            u_input = self._jerk_to_acc(action)
        else:
            raise ValueError(f"Unknown action base: {action_base}")

        if self.vehicle_model == VehicleModel.PM:
            # using vehicle_dynamics.state_to_array(current_state) causes error since state has orientation and velocity
            x_current = np.array(
                [
                    current_state.position[0],
                    current_state.position[1],
                    current_state.velocity,
                    current_state.velocity_y,
                ]
            )

            # if maximum absolute acceleration is exceeded, rescale the acceleration
            absolute_acc = u_input[0] ** 2 + u_input[1] ** 2
            if absolute_acc > self.parameters.longitudinal.a_max ** 2:
                rescale_factor = (self.parameters.longitudinal.a_max - 1e-6) / np.sqrt(absolute_acc)
                # rescale the acceleration to satisfy friction circle constraint
                u_input[0] *= rescale_factor
                u_input[1] *= rescale_factor
        else:
            x_current = np.array([
                current_state.position[0],
                current_state.position[1],
                current_state.steering_angle,
                current_state.velocity,
                current_state.orientation,
            ])

        try:
            x_current_old = copy.deepcopy(x_current)
            x_current = self.vehicle_dynamic.forward_simulation(x_current, u_input, self.dt, throw=True)
            self.violate_friction = False
        except FrictionCircleException:
            self.violate_friction = True
            for _ in range(N_INTEGRATION_STEPS):
                # simulate state transition - t parameter is set to vehicle.dt but irrelevant for the current vehicle models
                # TODO：x_dot of KS model considers the action constraints, which YR and PM model have not included yet
                x_dot = np.array(self.vehicle_dynamic.dynamics(self.dt, x_current, u_input))
                # update state
                x_current = x_current + x_dot * (self.dt / N_INTEGRATION_STEPS)

        # feed in required slots
        if self.vehicle_model == VehicleModel.PM:
            # simulated_state.acceleration = u_input[0]
            # simulated_state.acceleration_y = u_input[1]
            # simulated_state.orientation = np.arctan2(simulated_state.velocity_y, simulated_state.velocity)
            kwarg = {
                "position": np.array([x_current[0], x_current[1]]),
                "velocity": x_current[2],
                "velocity_y": x_current[3],
                "acceleration": u_input[0],
                "acceleration_y": u_input[1],
                "orientation": make_valid_orientation(np.arctan2(x_current[3], x_current[2])),
                "time_step": current_state.time_step + 1,
            }
        elif self.vehicle_model == VehicleModel.KS:
            # simulated_state.acceleration = u_input[1]
            # simulated_state.yaw_rate = (simulated_state.orientation - x_current_old[4]) / self.dt
            kwarg = {
                "position": np.array([x_current[0], x_current[1]]),
                "steering_angle": x_current[2],
                "velocity": x_current[3],
                "orientation": make_valid_orientation(x_current[4]),
                "acceleration": u_input[1],
                "yaw_rate": (x_current[4] - x_current_old[4]) / self.dt,
                "time_step": current_state.time_step + 1,
            }
        elif self.vehicle_model == VehicleModel.YawRate:
            # simulated_state.acceleration = u_input[0]
            # simulated_state.yaw_rate = u_input[1]
            kwarg = {
                "position": np.array([x_current[0], x_current[1]]),
                "steering_angle": x_current[2],
                "velocity": x_current[3],
                "orientation": make_valid_orientation(x_current[4]),
                "acceleration": u_input[1],
                "yaw_rate": u_input[0],
                "time_step": current_state.time_step + 1,
            }

        return State(**kwarg)

    def get_new_state(self, action: np.ndarray, action_base: str) -> State:
        """Generate the next state from current state for the given action.

        :params action: rescaled action
        :params action_base: aspect on which the action should be based ("jerk", "acceleration")
        :return: next state of vehicle"""

        current_state = self.state

        return self.propagate_one_time_step(current_state, action, action_base)

    def _jerk_to_acc(self, action: np.ndarray) -> np.ndarray:
        """
        computes the acceleration based input on jerk based actions
        :param action: action based on jerk
        :return: input based on acceleration
        """
        if self.vehicle_model == VehicleModel.PM:
            # action[jerk_x, jerk_y]
            action = np.array([np.clip(action[0], self.jerk_bounds[0], self.jerk_bounds[1]),
                               np.clip(action[1], self.jerk_bounds[0], self.jerk_bounds[1])])
            u_input = np.array([self.state.acceleration + action[0] * self.dt,
                                self.state.acceleration_y + action[1] * self.dt])

        elif self.vehicle_model == VehicleModel.KS:
            # action[steering angel speed, jerk]
            action = np.array([action[0], np.clip(action[1], self.jerk_bounds[0], self.jerk_bounds[1])])
            u_input = np.array([action[0], self.state.acceleration + action[1] * self.dt])

        elif self.vehicle_model == VehicleModel.YawRate:
            # action[yaw, jerk]
            action = np.array([action[0], np.clip(action[1], self.jerk_bounds[0], self.jerk_bounds[1])])
            u_input = np.array([action[0], self.state.acceleration + action[1] * self.dt])
        else:
            raise ValueError(f"Unknown vehicle model: {self.vehicle_model}")

        return u_input


class YawParameters():
    def __init__(self):
        # constraints regarding yaw
        self.v_min = []  # minimum yaw velocity [rad/s]
        self.v_max = []  # maximum yaw velocity [rad/s]


def extend_vehicle_params(p: VehicleParameters) -> VehicleParameters:
    p.yaw = YawParameters()
    p.yaw.v_min = -2.  # minimum yaw velocity [rad/s]
    p.yaw.v_max = 2.  # maximum yaw velocity [rad/s]
    return p


class YawRateDynamics(VehicleDynamics):
    """
    Description:
        Class for the calculation of vehicle dynamics of YawRate vehicle model
    """

    def __init__(self, vehicle_type: VehicleType):
        super(YawRateDynamics, self).__init__(VehicleModel.YawRate, vehicle_type)
        self.parameters = extend_vehicle_params(self.parameters)
        self.l = self.parameters.a + self.parameters.b

        self.velocity = None

    def dynamics(self, t, x, u) -> List[float]:
        """
        Yaw Rate model dynamics function.

        :param x: state values, [position x, position y, steering angle, longitudinal velocity, orientation(yaw angle)]
        :param u: input values, [yaw rate, longitudinal acceleration]

        :return: system dynamics
        """
        velocity_x = x[3] * np.cos(x[4])
        velocity_y = x[3] * np.sin(x[4])
        self.velocity = x[3]

        # steering angle velocity depends on longitudinal velocity and yaw rate (as well as vehicle parameters)
        steering_ang_velocity = -u[0] * self.l / (x[3] ** 2 + u[0] * self.l ** 2)

        return [velocity_x, velocity_y, steering_ang_velocity, u[1], u[0]]

    @property
    def input_bounds(self) -> Bounds:
        """
        Overrides the bounds method of Vehicle Model in order to return bounds for the Yaw Rate Model inputs.

        Bounds are
            - -max longitudinal acc <= acceleration <= max longitudinal acc
            - mini yaw velocity <= yaw_rate <= max yaw velocity

        :return: Bounds
        """
        return Bounds([self.parameters.yaw.v_min - 1e-4, -self.parameters.longitudinal.a_max],
                      [self.parameters.yaw.v_max + 1e-4, self.parameters.longitudinal.a_max])

    def _state_to_array(self, state: State, steering_angle_default=0.0) -> Tuple[np.array, int]:
        """ Implementation of the VehicleDynamics abstract method. """
        values = [
            state.position[0],
            state.position[1],
            getattr(state, 'steering_angle', steering_angle_default),  # not defined in initial state
            state.velocity,
            state.orientation,
        ]
        return np.array(values), state.time_step

    def _array_to_state(self, x: np.array, time_step: int) -> State:
        """ Implementation of the VehicleDynamics abstract method. """
        values = {
            'position': np.array([x[0], x[1]]),
            'steering_angle': x[2],
            'velocity': x[3],
            'orientation': x[4],
        }
        state = State(**values, time_step=time_step)
        return state

    def _input_to_array(self, input: State) -> Tuple[np.array, int]:
        """
        Actual conversion of input to array happens here. Vehicles can override this method to implement their own converter.
        """
        values = [
            input.yaw_rate,
            input.acceleration,
        ]
        return np.array(values), input.time_step

    def _array_to_input(self, u: np.array, time_step: int) -> State:
        """
        Actual conversion of input array to input happens here. Vehicles can override this method to implement their
        own converter.
        """
        values = {
            'yaw_rate': u[0],
            'acceleration': u[1],
        }
        return State(**values, time_step=time_step)