import carla

from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import get_vehicle_pos


class CarlaRightTurnEnvOOD(CarlaWptFixedEnv):
    """
    Vehicle passes the crossing (turn right) and avoid collision.

    **Provided Tasks**: ``carla_right_turn_simple``, ``carla_right_turn_medium``, ``carla_right_turn_hard``
    
    Additional config parameters:
    
    * ``pileup_positions``: List of [x, y, z, yaw] positions for stationary pile-up vehicles
    """

    def on_reset(self) -> None:
        super().on_reset()
        
        # Spawn stationary vehicles representing a pile-up in front of the car
        # Positions can be configured via pileup_positions in task config
        self.pileup_vehicles = []
        
        # Get pile-up positions from config, or use empty list if not configured
        if hasattr(self._config, 'pileup_positions') and self._config.pileup_positions:
            pileup_positions = self._config.pileup_positions
            
            for pos in pileup_positions:
                transform = carla.Transform(
                    carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                    carla.Rotation(yaw=pos[3])
                )
                vehicle = self._world.try_spawn_actor(transform=transform)
                if vehicle is not None:
                    # Apply handbrake to keep the vehicle stationary
                    control = carla.VehicleControl()
                    control.hand_brake = True
                    vehicle.apply_control(control)
                    self.pileup_vehicles.append(vehicle)

    def on_step(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(self.actor_flow[0])
            # Cleanup bounds - destroy vehicles that leave the intersection area
            
            # # Town04 bounds (for ego near [179.5, -169.5], flow spawn at [195.0, -175.0])
            # if y > -130.0 or y < -200.0 or x < 160.0 or x > 220.0:

            # Town04 bounds: ego at [179.5, -169.5], flow spawns at [220, -169.5] heading west
            # Destroy when: past ego (x < 165), too far east (x > 230), or off road (y drift)
            # if x < 165.0 or x > 230.0 or y > -155.0 or y < -185.0:
            #     self._world.destroy_actor(vehicle.id)
            #     self.actor_flow.popleft()
        super().on_step()
