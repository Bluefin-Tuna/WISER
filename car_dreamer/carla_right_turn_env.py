import carla

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import get_vehicle_pos


class CarlaRightTurnEnv(CarlaWptFixedEnv):
    """
    Vehicle passes the crossing (turn right) and avoid collision.

    **Provided Tasks**: ``carla_right_turn_simple``, ``carla_right_turn_medium``, ``carla_right_turn_hard``
    """

    _DEFAULT_FLOW_CLEANUP_BOUNDS = {
        "min_x": -38.4,
        "max_x": 31.6,
        "min_y": None,
        "max_y": -81.2,
    }

    def on_reset(self) -> None:
        super().on_reset()
        self._flow_cleanup_bounds = self._resolve_flow_cleanup_bounds()
        self._spawn_pileup_vehicles()

    def on_step(self) -> None:
        self._cleanup_actor_flow()
        super().on_step()

    def _spawn_pileup_vehicles(self) -> None:
        self.pileup_vehicles = []
        pileup_positions = getattr(self._config, "pileup_positions", ())
        for position in pileup_positions:
            if len(position) != 4:
                raise ValueError(
                    f"Expected pileup position [x, y, z, yaw], got: {position}"
                )
            transform = carla.Transform(
                carla.Location(x=position[0], y=position[1], z=position[2]),
                carla.Rotation(yaw=position[3]),
            )
            vehicle = self._world.try_spawn_actor(transform=transform)
            if vehicle is None:
                continue

            control = carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True)
            vehicle.apply_control(control)
            self.pileup_vehicles.append(vehicle)

    def _resolve_flow_cleanup_bounds(self):
        bounds = getattr(self._config, "flow_cleanup_bounds", None)

        def bound(key):
            value = None
            if bounds is not None:
                value = getattr(bounds, key, None)
                if value is None and isinstance(bounds, dict):
                    value = bounds.get(key)
            if value is None:
                return self._DEFAULT_FLOW_CLEANUP_BOUNDS[key]
            return float(value)

        return {
            "min_x": bound("min_x"),
            "max_x": bound("max_x"),
            "min_y": bound("min_y"),
            "max_y": bound("max_y"),
        }

    def _cleanup_actor_flow(self) -> None:
        if len(self.actor_flow) == 0:
            return

        vehicle = self.actor_flow[0]
        x, y = get_vehicle_pos(vehicle)
        if self._is_outside_flow_cleanup_bounds(x, y):
            self._world.destroy_actor(vehicle.id)
            self.actor_flow.popleft()

    def _is_outside_flow_cleanup_bounds(self, x: float, y: float) -> bool:
        bounds = getattr(
            self, "_flow_cleanup_bounds", self._DEFAULT_FLOW_CLEANUP_BOUNDS
        )
        min_x = bounds.get("min_x")
        max_x = bounds.get("max_x")
        min_y = bounds.get("min_y")
        max_y = bounds.get("max_y")

        if min_x is not None and x < min_x:
            return True
        if max_x is not None and x > max_x:
            return True
        if min_y is not None and y < min_y:
            return True
        if max_y is not None and y > max_y:
            return True
        return False
