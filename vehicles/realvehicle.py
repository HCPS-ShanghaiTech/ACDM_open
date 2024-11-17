import carla
import utils.globalvalues as gv
import utils.dyglobalvalues as dgv
import math
from .virtualvehicle import VirtualVehicle
from utils import pickleable


class RealVehicle:
    """
    The actual vehicle generated in Carla which contains additional physical information
    """

    def __init__(
        self,
        vehicle,
        spawn_point,
        controller,
        scalar_velocity=0,
        control_action="MAINTAIN",
        virtual_pickleable=False,
    ):
        self.vehicle = vehicle
        self.spawn_point = spawn_point  # The default initial location for Carla is (0,0,0), so an input for the initial location is required
        self.scalar_velocity = scalar_velocity
        self.control_action = control_action
        self.controller = controller
        self.next_waypoint = None
        self.target_lane_id = None
        self.lane_changing_route = []
        self.changing_lane_pace = 0  # Determine whether they are in the process of changing lanes and how much has been completed
        self.nade_observer = None  # Only for D2RL experiment
        self.virtual_pickleable = virtual_pickleable  # True for multiprocessing mode

    def __del__(self):
        pass

    def run_step(self, realvehicle_id_list, **kwargs):
        """One decision step"""
        self.control_action = self.controller.run_forward(
            realvehicle_id_list, network=kwargs.get("network")
        )

    def run_partial_step(self, realvehicle_id_list, **kwargs):
        """
        Assuming excluding some vehicles in the scene
        """
        return self.controller.run_forward(realvehicle_id_list)

    def clone_to_virtual(self):
        """
        Create a VirtualVehicle with same state as self
        """
        waypoint = dgv.get_map().get_waypoint(self.vehicle.get_location())
        transform = self.vehicle.get_transform()
        if self.virtual_pickleable:
            waypoint = pickleable.simp_carla_wp(waypoint)
            transform = pickleable.from_carla_to_transform(transform)
        return VirtualVehicle(
            self.vehicle.id,
            waypoint,
            transform,
            self.scalar_velocity,
            self.control_action,
            self.virtual_pickleable,
        )

    def cal_lanechanging_route(self):
        """
        Calculate the path during the lane change process
        Return: List[carla.Transform]
        """
        traj_length = int(gv.LANE_CHANGE_TIME / gv.STEP_DT)
        route = [None] * (traj_length + 1)
        # The first element of Route contains the last WP before the lane change begins, \
        # and the trajectory should start from route [1] when called
        route[0] = self.vehicle.get_transform()
        route[-1] = self.target_dest_wp.transform
        # Diection
        direction_vector = route[-1].location - route[0].location
        yaw = math.degrees(math.atan2(direction_vector.y, direction_vector.x))
        rotation = carla.Rotation(pitch=0, yaw=yaw, roll=0)
        for i in range(1, traj_length):
            location = carla.Location(
                x=route[0].location.x + i * direction_vector.x / (traj_length - 1),
                y=route[0].location.y + i * direction_vector.y / (traj_length - 1),
                z=route[0].location.z,
            )
            wp = dgv.get_map().get_waypoint(location)
            location.z = wp.transform.location.z
            rotation.pitch = wp.transform.rotation.pitch
            route[i] = carla.Transform(
                location=location,
                rotation=rotation,
            )
        return route

    def descrete_control(self):
        """
        Convert discrete actions into path planning
        """
        longitude_action_list = ["MAINTAIN", "ACCELERATE", "DECELERATE"]
        lateral_action_list = ["SLIDE_LEFT", "SLIDE_RIGHT", "SLIDE"]
        if self.next_waypoint == None:
            self.next_waypoint = dgv.get_map().get_waypoint(self.spawn_point.location)
        if self.control_action in lateral_action_list:
            # Actions involving lane changes, setting a target waypoint and driving to the target location in one second
            current_waypoint = dgv.get_map().get_waypoint(self.vehicle.get_location())
            self.current_dest_wp = current_waypoint.next(self.scalar_velocity)[0]
            if self.control_action == "SLIDE":
                if current_waypoint.lane_id == gv.LANE_ID["Left"]:
                    self.control_action = "SLIDE_RIGHT"
                if current_waypoint.lane_id == gv.LANE_ID["Right"]:
                    self.control_action = "SLIDE_LEFT"
            if self.control_action == "SLIDE_LEFT":
                if self.current_dest_wp.lane_id == gv.LANE_ID["Right"]:
                    self.target_dest_wp = self.current_dest_wp.get_left_lane()
                else:
                    self.target_dest_wp = self.current_dest_wp
            if self.control_action == "SLIDE_RIGHT":
                if self.current_dest_wp.lane_id == gv.LANE_ID["Left"]:
                    self.target_dest_wp = self.current_dest_wp.get_right_lane()
                else:
                    self.target_dest_wp = self.current_dest_wp
            if self.changing_lane_pace == 0:
                self.lane_changing_route = self.cal_lanechanging_route()
                self.target_lane_id = (
                    dgv.get_map()
                    .get_waypoint(self.lane_changing_route[-1].location)
                    .lane_id
                )
            self.changing_lane_pace += 1
            if self.changing_lane_pace < len(self.lane_changing_route) - 1:
                self.next_waypoint = dgv.get_map().get_waypoint(
                    self.lane_changing_route[self.changing_lane_pace].location
                )
                self.vehicle.set_transform(
                    self.lane_changing_route[self.changing_lane_pace]
                )
            else:
                self.changing_lane_pace = 0
                self.next_waypoint = (
                    dgv.get_map()
                    .get_waypoint(self.lane_changing_route[-1].location)
                    .next(self.scalar_velocity * gv.STEP_DT)[0]
                )
                self.vehicle.set_transform(self.next_waypoint.transform)
        if (
            self.control_action in longitude_action_list
            or type(self.control_action) != str
        ):
            if (
                self.scalar_velocity <= gv.MIN_SPEED
                and self.control_action == "DECELERATE"
            ):
                self.control_action = "MAINTAIN"
            if type(self.control_action) == str:
                self.scalar_velocity += (
                    gv.LON_ACC_DICT.get(self.control_action) * gv.STEP_DT
                )
            else:
                self.scalar_velocity += self.control_action * gv.STEP_DT
            wp_gap = self.scalar_velocity * gv.STEP_DT
            if wp_gap == 0:
                pass
            if wp_gap > 0:
                self.next_waypoint = self.next_waypoint.next(wp_gap)[0]
            if wp_gap < 0:
                self.next_waypoint = self.next_waypoint.previous(-wp_gap)[0]
            self.target_lane_id = self.next_waypoint.lane_id
            self.vehicle.set_transform(self.next_waypoint.transform)

    def collision_callback(self, other_vehicle):
        """
        Determine whether two vehicles have collided
        """
        # If the horizontal distance between two vehicles is greater than the sum of \
        # the diagonal lengths of the bounding box, then there must have been no collision
        if (
            self.vehicle.get_location() == carla.Location(0, 0, 0)
            or self.vehicle.get_location().distance_squared_2d(
                other_vehicle.get_location()
            )
            > (
                self.vehicle.bounding_box.extent.x**2
                + self.vehicle.bounding_box.extent.y**2
            )
            * 4
        ):
            return False
        # Otherwise, calculate the projection of the relative position vectors of \
        # the two vehicles in the forward direction of the main vehicle and compare \
        # it with the length of the vehicle body
        distance_vector = other_vehicle.get_location() - self.vehicle.get_location()
        forward_vector = self.vehicle.get_transform().get_forward_vector()
        projection = abs(
            distance_vector.dot_2d(forward_vector)
            / forward_vector.distance_2d(carla.Vector3D(0, 0, 0))
        )
        if (
            distance_vector.distance_squared_2d(carla.Vector3D(0, 0, 0))
            - projection * projection
            > 0.01
        ):
            normal = math.sqrt(
                distance_vector.distance_squared_2d(carla.Vector3D(0, 0, 0))
                - projection * projection
            )
        else:
            normal = 0
        if (
            projection
            <= (
                self.vehicle.bounding_box.extent.x + other_vehicle.bounding_box.extent.x
            )
            * 0.9
            and normal
            <= (
                self.vehicle.bounding_box.extent.y + other_vehicle.bounding_box.extent.y
            )
            * 0.9
        ):
            return True
        else:
            return False
