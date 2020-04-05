from flow.controllers.base_controller import BaseController


class DieController(BaseController):
    def __init__(self,
                 veh_id,
                 a=1,
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
        self.a = a
        self.dt = dt
        self.go_action = False

    def get_accel(self, env):
        return self.a
