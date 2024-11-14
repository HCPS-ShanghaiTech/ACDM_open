import carla
import math
from utils.globalvalues import CAR_WIDTH, CAR_LENGTH
from utils.pickleable import Transform, Location, Rotation


class VirtualVehicle:
    """
    Vehicles representing future states which will not appear in Simulator
    """

    def __init__(
        self, id, waypoint, transform, scalar_velocity, control_action, pickleable=False
    ):
        self.id = id
        self.waypoint = waypoint
        self.transform = transform
        self.scalar_velocity = scalar_velocity
        self.control_action = control_action
        self.pickleable = pickleable

    def clone_self(self):
        if self.pickleable:
            transform = Transform(
                Location(
                    self.transform.location.x,
                    self.transform.location.y,
                    self.transform.location.z,
                ),
                Rotation(
                    self.transform.rotation.pitch,
                    self.transform.rotation.yaw,
                    self.transform.rotation.roll,
                ),
            )
        else:
            transform = carla.Transform(
                carla.Location(
                    self.transform.location.x,
                    self.transform.location.y,
                    self.transform.location.z,
                ),
                carla.Rotation(
                    self.transform.rotation.pitch,
                    self.transform.rotation.yaw,
                    self.transform.rotation.roll,
                ),
            )
        return VirtualVehicle(
            self.id,
            self.waypoint.next(1e-9)[0],
            transform,
            self.scalar_velocity,
            self.control_action,
            self.pickleable,
        )

    def judge_collision(self, other_virtual_vehicle):
        if (
            self.transform.location == carla.Location(0, 0, 0)
            or self.transform.location.distance_2d(
                other_virtual_vehicle.transform.location
            )
            > 6
        ):
            return False
        if (
            (self.control_action not in ["SLIDE_LEFT", "SLIDE_RIGHT"])
            and (
                other_virtual_vehicle.control_action
                not in ["SLIDE_LEFT", "SLIDE_RIGHT"]
            )
            and (self.waypoint.lane_id != other_virtual_vehicle.waypoint.lane_id)
        ):
            return False
        if (
            self.transform.location == carla.Location(0, 0, 0)
            or self.transform.location.distance_2d(
                other_virtual_vehicle.transform.location
            )
            <= CAR_WIDTH
        ):
            return True
        distance_vector = (
            other_virtual_vehicle.transform.location - self.transform.location
        )
        forward_vector = self.transform.get_forward_vector()
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
        if projection <= CAR_LENGTH * 1.2 and normal <= CAR_WIDTH * 1.2:
            return True
        else:
            return False
