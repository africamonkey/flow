import numpy as np
from flow.controllers.base_controller import BaseController


MAX_DECEL = 4.5
CONSECUTIVE_LESS_THRESHOLD = 2
DEBUG = True


def get_risk(d_i, v_i, t_c, d_s, lambda_a):
    if d_i > v_i * t_c:
        return 0.0
    if d_i < d_s:
        return 1.0
    return np.exp(-lambda_a * (d_i - d_s))


class PRMController(BaseController):
    def __init__(self,
                 veh_id,
                 t_c,
                 d_s,
                 lambda_a,
                 r_go,
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
        self.t_c = t_c
        self.d_s = d_s
        self.lambda_a = lambda_a
        self.r_go = r_go
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.dt = dt
        self.go_action = False
        self.less_threshold = 0

    def get_accel(self, env):
        v = env.k.vehicle.get_speed(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)
        edg = env.k.vehicle.get_edge(self.veh_id)
        pos = env.k.vehicle.get_position(self.veh_id)
        ori = env.k.vehicle.get_orientation(self.veh_id)
        rou = env.k.vehicle.get_route(self.veh_id)

        max_risk = 0
        for oth_id in env.k.vehicle.get_ids():
            if oth_id == self.veh_id:
                continue
            oth_route = env.k.vehicle.get_route(oth_id)
            oth_e2w = (oth_route == ('E2M', 'M2WW', 'WW2W'))
            self_e = (rou == ('SS2M', 'M2EE'))
            if self_e and oth_e2w:
                continue
            ori_oth = env.k.vehicle.get_orientation(oth_id)
            v_oth = env.k.vehicle.get_speed(oth_id)
            d = ori_oth[0] - ori[0]
            if not oth_e2w:
                d = -d
            if d < 0:
                continue
            risk = get_risk(d, v_oth, self.t_c, self.d_s, self.lambda_a)
            max_risk = max(max_risk, risk)

        if DEBUG:
            print("max_risk {0}, v {1}, ori {2}"
                  .format(max_risk,
                          v,
                          ori))

        if max_risk < self.r_go:
            self.less_threshold += 1
            if self.less_threshold >= CONSECUTIVE_LESS_THRESHOLD:
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
