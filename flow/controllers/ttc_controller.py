import numpy as np
from flow.utils import ttc_utils
from flow.controllers.base_controller import BaseController


MAX_DECEL = 4.5
CONSECUTIVE_GREATER_THRESHOLD = 2
DEBUG = False


class TTCController(BaseController):
    def __init__(self,
                 veh_id,
                 ttc_threshold,
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
        self.go_action = False
        self.greater_threshold = 0
        self.ttc_threshold = ttc_threshold

    def get_accel(self, env):
        v = env.k.vehicle.get_speed(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)
        pos = env.k.vehicle.get_position(self.veh_id)
        ori = env.k.vehicle.get_orientation(self.veh_id)
        length = env.k.vehicle.get_length(self.veh_id)
        if pos <= 25:
            h = 25.1 - pos
        else:
            min_ttc = 1000
            min_ori_oth = None
            min_v_oth = None
            for oth_id in env.k.vehicle.get_ids():
                ori_oth = env.k.vehicle.get_orientation(oth_id)
                v_oth = env.k.vehicle.get_speed(oth_id)
                if oth_id != self.veh_id:
                    ttc = ttc_utils.car_ttc(ori, ori_oth, v, v_oth)
                    if min_ttc > ttc:
                        min_ttc = ttc
                        min_ori_oth = ori_oth
                        min_v_oth = v_oth

            if DEBUG:
                print("min_ttc {0}, v {1}, ori {2}, v_oth {3}, ori_oth {4}"
                      .format(min_ttc,
                              v,
                              ori,
                              min_v_oth,
                              min_ori_oth))

            if min_ttc > self.ttc_threshold:
                self.greater_threshold += 1
                if self.greater_threshold >= CONSECUTIVE_GREATER_THRESHOLD:
                    self.go_action = True
            else:
                self.greater_threshold = 0
                # self.go_action = False

            if not self.go_action:
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
