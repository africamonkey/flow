import numpy as np
from flow.controllers.base_controller import BaseController


CONSECUTIVE_LESS_THRESHOLD = 2


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
                 max_accel,
                 max_deaccel,
                 desire_v,
                 d_nudge,
                 discount,
                 debug=False,
                 time_delay=0.0,
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
        self.max_accel = max_accel
        self.max_deaccel = max_deaccel
        self.desire_v = desire_v
        self.d_nudge = d_nudge
        self.discount = discount
        self.go_action = False
        self.less_threshold = 0
        self.debug = debug

    def get_accel(self, env):
        v = env.k.vehicle.get_speed(self.veh_id)
        edg = env.k.vehicle.get_edge(self.veh_id)
        pos = env.k.vehicle.get_position(self.veh_id)
        ori = env.k.vehicle.get_orientation(self.veh_id)
        rou = env.k.vehicle.get_route(self.veh_id)

        if edg == 'SS2M' and pos <= self.d_nudge:
            v_e = (1. - (pos / self.d_nudge) ** 2) * self.discount * self.desire_v
        else:
            max_risk = 0
            max_risk_v_oth = 0
            max_risk_ori_oth = 0
            max_risk_d = 0
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
                if risk > max_risk:
                    max_risk = risk
                    max_risk_v_oth = v_oth
                    max_risk_ori_oth = ori_oth
                    max_risk_d = d

            if self.debug:
                print("max_risk {0}, v {1}, ori {2}, v_oth {3}, ori_oth {4}, d {5}"
                      .format(max_risk,
                              v,
                              ori,
                              max_risk_v_oth,
                              max_risk_ori_oth,
                              max_risk_d))

            if max_risk < self.r_go:
                self.less_threshold += 1
                if self.less_threshold >= CONSECUTIVE_LESS_THRESHOLD:
                    self.go_action = True
            else:
                self.less_threshold = 0
                # self.go_action = False

            if not self.go_action:
                v_e = 0
            else:
                v_e = self.desire_v

        accel = (v_e - v) / env.env_params.sims_per_step
        accel = min(accel, self.max_accel)
        accel = max(accel, - np.abs(self.max_deaccel))
        return accel
