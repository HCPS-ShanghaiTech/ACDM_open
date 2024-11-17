"For multi process"

from scipy.spatial import distance
import utils.dyglobalvalues as dgv
import math
import carla


class Vector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: "Vector3D"):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def dot_2d(self, vector: "Vector3D"):
        return self.x * vector.x + self.y * vector.y

    def distance_2d(self, vector: "Vector3D"):
        return distance.euclidean((self.x, self.y), (vector.x, vector.y))

    def distance_squared_2d(self, vector: "Vector3D"):
        return distance.sqeuclidean((self.x, self.y), (vector.x, vector.y))


class Location(Vector3D):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)

    def __add__(self, other: "Location"):
        return Location(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Location"):
        return Location(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, num):
        return Location(self.x / num, self.y / num, self.z / num)


class Rotation:
    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll


class Transform:
    def __init__(self, location: "Location", rotation: "Rotation"):
        self.location = location
        self.rotation = rotation

    def get_forward_vector(self):
        yaw_rad = math.radians(self.rotation.yaw)
        pitch_rad = math.radians(self.rotation.pitch)

        c_yaw = math.cos(yaw_rad)
        s_yaw = math.sin(yaw_rad)
        c_pitch = math.cos(pitch_rad)
        s_pitch = math.sin(pitch_rad)

        forward_x = c_yaw * c_pitch
        forward_y = s_yaw * c_pitch
        forward_z = s_pitch

        return Vector3D(forward_x, forward_y, forward_z)


class SimpleWaypoint:
    def __init__(self, transform, lane_id, s, road_id):
        self.transform = transform
        self.lane_id = lane_id
        self.s = s
        self.road_id = road_id

    def get_left_lane(self):
        carla_wp = dgv.get_map().get_waypoint(
            from_location_to_carla(self.transform.location)
        )
        return simp_carla_wp(carla_wp.get_left_lane())

    def get_right_lane(self):
        carla_wp = dgv.get_map().get_waypoint(
            from_location_to_carla(self.transform.location)
        )
        return simp_carla_wp(carla_wp.get_right_lane())

    def next(self, gap):
        carla_wp = dgv.get_map().get_waypoint(
            from_location_to_carla(self.transform.location)
        )
        next_wp_list = carla_wp.next(gap)
        return [simp_carla_wp(next_wp_list[0])]

    def previous(self, gap):
        carla_wp = dgv.get_map().get_waypoint(
            from_location_to_carla(self.transform.location)
        )
        previous_wp_list = carla_wp.previous(gap)
        return [simp_carla_wp(previous_wp_list[0])]


def from_carla_to_transform(carla_transform):
    return Transform(
        location=Location(
            carla_transform.location.x,
            carla_transform.location.y,
            carla_transform.location.z,
        ),
        rotation=Rotation(
            carla_transform.rotation.pitch,
            carla_transform.rotation.yaw,
            carla_transform.rotation.roll,
        ),
    )


def simp_carla_wp(carla_wp):
    return SimpleWaypoint(
        from_carla_to_transform(carla_wp.transform),
        carla_wp.lane_id,
        carla_wp.s,
        carla_wp.road_id,
    )


def from_location_to_carla(location: Location):
    return carla.Location(x=location.x, y=location.y, z=location.z)
