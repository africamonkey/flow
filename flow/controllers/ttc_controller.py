import numpy as np
import ttc_utils
from flow.controllers.base_controller import BaseController


MAX_DECEL = 4.5
WARNING_DECEL = 4.0
OBSERVE_RADIUS = 30


class TTCController(BaseController):
    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 dt=0.1,
                 noise=0,
                 fail_safe=None,
                 car_following_params=None):
        """Instantiate an IDM controller."""
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise)
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.dt = dt

    def get_accel(self, env):
        v = env.k.vehicle.get_speed(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)
        ori = env.k.vehicle.get_orientation(self.veh_id)
        length = env.k.vehicle.get_length(self.veh_id)
        min_ttc = 1000
        for oth_id in env.k.vehicle.get_ids():
            ori_oth = env.k.vehicle.get_orientation(oth_id)
            v_oth = env.k.vehicle.get_speed(oth_id)
            if oth_id != self.veh_id and \
                    ttc_utils.dis(ori, ori_oth) < OBSERVE_RADIUS:
                ttc, dis = ttc_utils.car_ttc(ori, ori_oth, v, v_oth, self.s0 + length)
                if min_ttc > ttc:
                    min_ttc = ttc
                h = min(h, dis)

        print("min_ttc:", min_ttc, ", v:", v)

        if v / WARNING_DECEL > min_ttc:
            return - MAX_DECEL

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        lead_id = env.k.vehicle.get_leader(self.veh_id)
        if lead_id is None or lead_id == '':
            s_star = 0
        else:
            v_lead = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - v_lead) /
                   (2 * np.sqrt(self.a * self.b)))

        return self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
