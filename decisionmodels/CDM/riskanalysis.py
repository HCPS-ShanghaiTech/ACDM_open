import math
from utils.extendmath import cal_magnitude, cal_rel_location_curve


class EllipseGenerator:
    """
    用来生成计算risk的辅助椭圆
    """

    def __init__(self, map, c1_location, c2_location, ego_location, c) -> None:
        self._map = map
        self._c1_location = c1_location
        self._c2_location = c2_location
        self._ego_location = ego_location
        self._c = c
        self._car_mindis = 4.5
        self._disriskfunc_inner_b = 60
        self._disriskfunc_outer_b = 8
        self._disriskfunc_outer_xaxis_pos = 22

    def cal_risk_vector(self, car_location):
        """
        计算风险向量
        """
        a = (
            cal_magnitude(cal_rel_location_curve(car_location - self._c1_location))
            + cal_magnitude(cal_rel_location_curve(car_location - self._c2_location))
        ) / 2
        b = math.sqrt(a * a - self._c * self._c)

    def disriskfun_outer(self, x):
        """
        在焦距外的风险计算
        """
        if x <= 0:
            return self._disriskfunc_outer_b

        outer_a = -self._disriskfunc_outer_b / (
            self._disriskfunc_outer_xaxis_pos * self._disriskfunc_outer_xaxis_pos
        )
        res = outer_a * x * x + self._disriskfunc_outer_b

        return max(0, res)

    def disriskfunc_inner(self, x):
        """
        在焦距内的风险计算
        """
        if x > 2 * self._c:
            return self._disriskfunc_outer_b
        if x < self._car_mindis:
            return self._disriskfunc_inner_b

        inner_a = self._disriskfunc_inner_b / ((self._c * 2 - self._car_mindis) ** 2)
        res = inner_a * (x - self._c * 2) ** 2 + self._disriskfunc_outer_b

        return res
