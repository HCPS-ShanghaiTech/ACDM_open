import math
import carla
from utils.globalvalues import CAR_LENGTH
from utils.extendmath import cal_length, cal_rel_location_curve


class EllipseGenerator:
    """
    Used to generate auxiliary ellipses for risk calculation.
    """

    def __init__(self, c1_location, c2_location, c) -> None:
        self.c1_location = c1_location
        self.c2_location = c2_location
        self.c = c
        self.car_mindis = CAR_LENGTH
        self.disriskfunc_inner_b = 60
        self.disriskfunc_outer_b = 8
        self.disriskfunc_outer_xaxis_pos = 22

    def cal_risk_vector(self, car_location):
        """
        Calculate the risk vector and return the squared length of Vector2D.
        """
        a = (
            cal_length(cal_rel_location_curve(car_location, self.c1_location))
            + cal_length(cal_rel_location_curve(car_location, self.c2_location))
        ) / 2
        b = math.sqrt(max(a * a - self.c * self.c, 0))
        o_location = (self.c1_location + self.c2_location) / 2
        # No ellipse, calculate distance
        if abs(b) < 1.2 or a * a - self.c * self.c <= 0:
            ego_norm = cal_rel_location_curve(self.c2_location, self.c1_location)
            ego_norm = carla.Vector2D(ego_norm.x, ego_norm.y)
            ellipse_risk = self.disriskfunc_inner(
                max(
                    cal_length(cal_rel_location_curve(car_location, self.c1_location)),
                    self.car_mindis,
                )
            )
            a = 0
            b = 0
        else:
            rel_location = cal_rel_location_curve(car_location, o_location)
            ego_norm = carla.Vector2D(
                rel_location.x / (a * a), rel_location.y / (b * b)
            )
            ellipse_risk = self.disriskfun_outer(b)

        if ego_norm.length() > 0:
            ego_norm = (
                ego_norm.make_unit_vector()
            )  # Not sure why the documentation states that it returns a Vector3D; it should be wrong.
        return (ego_norm * ellipse_risk).length()

    def disriskfun_outer(self, x):
        """
        Risk calculation outside of the focal distance.
        """
        if x <= 0:
            return self.disriskfunc_outer_b

        outer_a = -self.disriskfunc_outer_b / (
            self.disriskfunc_outer_xaxis_pos * self.disriskfunc_outer_xaxis_pos
        )
        res = outer_a * x * x + self.disriskfunc_outer_b

        return max(0, res)

    def disriskfunc_inner(self, x):
        """
        Risk calculation inside of the focal distance.
        """
        if x > 2 * self.c:
            return self.disriskfunc_outer_b
        if x < self.car_mindis:
            return self.disriskfunc_inner_b

        inner_a = self.disriskfunc_inner_b / ((self.c * 2 - self.car_mindis) ** 2)
        res = inner_a * ((x - self.c * 2) ** 2) + self.disriskfunc_outer_b
        return res
