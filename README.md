
We train Multi-Sensor and Single Sensor DreamerV3 agents on our built-in tasks with a single 4090. Depending on the observation spaces, the memory overhead ranges from 10GB-20GB alongwith 3GB reserved for CARLA. We provide all scripts as well as instructions for additional configs that are necessary to run different World Model denoising and rejection scoring methods. In addition, we provide highly customizable noise injection to test our world models. In this branch (See other branches for Safety Gymnasium and Cosmos configurations), we provide multi-sensor checkpoints (trained with and without multi-representation dropout) for tasks:


| Right Turn Simple | Roundabout | Left turn hard | Lane merge | Overtake |
| :-------------: | :--------: | :------------: | :--------: | :---------------: |
| ![Right turn hard](https://ucd-dare.github.io/wiser.github.io/static/gifs/camera/right_turn_hard.gif) | ![Roundabout](https://ucd-dare.github.io/wiser.github.io/static/gifs/camera/roundabout.gif) | ![Lane merge](https://ucd-dare.github.io/wiser.github.io/static/gifs/camera/lane_merge.gif) | ![Right turn simple](https://ucd-dare.github.io/wiser.github.io/static/gifs/camera/overtake.gif) |

| Stop Sign |
| :-------: |
| ![Stop Sign](https://ucd-dare.github.io/wiser.github.io/static/gifs/bev/stop%20sign.gif) |



## 📋 Prerequisites

### WISER Dependencies

To install WISER tasks or the development suite, clone the repository:

```bash
git clone https://github.com/Bluefin-Tuna/WISER.git
cd WISER
```

Download [CARLA release](https://github.com/carla-simulator/carla/releases) of version `0.9.15`. Set the following environment variables:

```bash
export CARLA_ROOT="</path/to/carla>"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
```

Install the package using flit. The `--symlink` flag is used to create a symlink to the package in the Python environment, so that changes to the package are immediately available without reinstallation. (`--pth-file` also works, as an alternative to `--symlink`.)

```bash
conda create python=3.10 --name wiser
conda activate wiser
pip install flit
cd dreamerv3
pip install -r requirements.txt
flit install --symlink
```

### Model Dependencies

For this branch, the model backbones are decoupled from Wiser tasks or the development sutie. Users can install model dependencies on their own demands. To install DreamerV3, check out the guidelines [DreamerV3](https://github.com/ucd-dare/Wiser/tree/master/dreamerv3).

### Quick Start

### :mechanical_arm: Training

We suggest starting with Carla as we provide results for both multi-sensor settings and single-sensor settings. To train DreamerV3 agents, use

```bash
# Example 1: Use default settings to train an agent
bash train_dm3.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane
# Example 2: Override task and model parameters
bash train_dm3.sh 2000 0 --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/carla_right_turn_simple \
    --dreamerv3.run.steps=5e6
```

The command will launch CARLA at 2000 port, load task a built-in task named `carla_four_lane`, and start the visualization tool at port 9000 (2000+7000) which can be accessed through `http://localhost:9000/`. You can append flags to the command to overwrite yaml configurations.

### :rocket: Evaluation

We provide evaluation scripts that allow for selection of noise type, noise intensity, and noise proportion. 

### Creating Tasks and Adding Noises:

The section explains how to create Wiser tasks in a standalone mode without loading our integrated models. This can be helpful **if you want to train and evaluate your own models**.

Each task class can be instantiated with various configurations. For instance, the right-turn task can be set up with simple, medium, or hard settings. These settings are defined in YAML blocks within [tasks.yaml](https://github.com/ucd-dare/Wiser/blob/main/car_dreamer/configs/tasks.yaml). The task creation API retrieves the given identifier (e.g., `carla_four_lane_hard`) from these YAML task blocks and injects the settings into the task class to create a gym task instance.

```python
# Create a gym environment with default task configurations
import car_dreamer
task, task_configs = car_dreamer.create_task('carla_four_lane_hard')

# Or load default environment configurations without instantiation
task_configs = car_dreamer.load_task_configs('carla_right_turn_hard')
```

To create your own driving tasks using the development suite, refer to [Wiser Docs: Customization](https://wiser.readthedocs.io/en/latest/customization.html).

### Observation Customization

`Wiser` employs an `Observer-Handler` architecture to manage complex **multi-modal** observation spaces. Each handler defines its own observation space and lifecycle for stepping, resetting, or fetching information, similar to a gym environment. The agent communicates with the environment through an observer that manages these handlers.

Users can enable built-in observation handlers such as BEV, camera, LiDAR, and spectator in task configurations. Check out [common.yaml](https://github.com/ucd-dare/Wiser/blob/master/car_dreamer/configs/common.yaml) for all available built-in handlers. Additionally, users can customize observation handlers and settings to suit their specific needs.

#### Handler Implementation

To implement new handlers for different observation sources and modalities (e.g., text, velocity, locations, or even more complex data), `Wiser` provides two methods:

1. Register a callback as a [SimpleHandler](https://github.com/ucd-dare/Wiser/blob/master/car_dreamer/toolkit/observer/handlers/simple_handler.py) to fetch data at each step.
1. For observations requiring complex workflows that cannot be conveyed by a `SimpleHandler`, create an handler maintaining the full lifecycle of that observation, similar to our built-in message, BEV, spectator handlers.

For more details on defining new observation sources, see [Wiser Docs: Defining a new observation source](https://wiser.readthedocs.io/en/latest/customization.html#defining-a-new-observation-source).

#### Observation Handler Configurations

Each handler can access yaml configurations for further customization. For example, a BEV handler setting can be defined as:

```yaml
birdeye_view:
   # Specify the handler name used to produce `birdeye_view` observation
   handler: birdeye
   # The observation key
   key: birdeye_view
   # Define what to render in the birdeye view
   entities: [roadmap, waypoints, background_waypoints, fov_lines, ego_vehicle, background_vehicles]
   # ... other settings used by the BEV handler
```

The handler field specifies which handler implementation is used to manage that observation key. Then, users can simply enable this observation in the task settings.

```yaml
your_task_name:
  env:
    observation.enabled: [camera, collision, spectator, birdeye_view]
```

#### Environment & Observer Communications

One might need transfer information from the environements to a handler to compute their observations. E.g., a BEV handler might need a location to render the destination spot. These environment information can be accessed either through [WorldManager](https://wiser.readthedocs.io/en/latest/api/toolkit.html#car_dreamer.toolkit.WorldManager) APIs, or through environment state management.

A `WorldManager` instance is passed in the handler during its initialization. The environment states are defined by an environment's `get_state()` API, and passed as parameters to handler's `get_observation()`.

```python
class MyHandler(BaseHandler):
    def __init__(self, world: WorldManager, config):
        super().__init__(world, config)
        self._world = world

def get_observation(self, env_state: Dict) -> Tuple[Dict, Dict]:
    # Get the waypoints through environment states
    waypoints = env_state.get("waypoints")
    # Get actors through the world manager API
    actors = self._world.actors
    # ...

class MyEnv(CarlaBaseEnv):
    # ...
    def get_state(self):
        return {
            # Expose the waypoints through get_state()
            'waypoints': self.waypoints,
        }
```

## :computer: Visualization Tool

We stream observations, rewards, terminal conditions, and custom metrics to a web browser for each training session in real-time, making it easier to engineer rewards and debug.

<table style="margin-left: auto; margin-right: auto;">
  <tr>
    <td class="center-text">Visualization Server</td>
  </tr>
  <tr>
    <td><img src="https://ucd-dare.github.io/wiser.github.io/static/images/visualization.png" style="width: 100%"></td>
  </tr>
</table>

## :hammer: System

...

To easily customize your own driving tasks, and observation spaces, etc., please refer to our [Wiser API Documents](https://wiser.readthedocs.io/en/latest/).

![Wiser](https://ucd-dare.github.io/wiser.github.io/static/images/WiserSystem.png)

# :star2: Citation

If you find this repository useful, please cite this paper:

**[ArXiv paper link](https://arxiv.org/abs/2512.01119v1)**

```
@misc{zollicoffer2025worldmodelrobustnesssurprise,
      title={World Model Robustness via Surprise Recognition}, 
      author={Geigh Zollicoffer and Tanush Chopra and Mingkuan Yan and Xiaoxu Ma and Kenneth Eaton and Mark Riedl},
      year={2025},
      eprint={2512.01119},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.01119}, 
}
```

# Supplementary Material

## World model imagination

<p align="center">
  Birdeye view imagination
</p>
<img src="https://ucd-dare.github.io/wiser.github.io/static/gifs/right_turn_hard_pre_bev.gif">
<p align="center">
  Camera view imagination
</p>
<img src="https://ucd-dare.github.io/wiser.github.io/static/gifs/right_turn_hard_pre_camera.gif">
<p align="center">
  LiDAR view imagination
</p>
<img src="https://ucd-dare.github.io/wiser.github.io/static/gifs/right_turn_hard_pre_lidar.gif">


```bash
# Setup pre-commit tool
pip install pre-commit
pre-commit install
# Run pre-commit
pre-commit run --all-files
```

### Credits

`WISER` builds on the several projects within the autonomous driving and machine learning communities.

- [gym-carla](https://github.com/cjy1992/gym-carla)
- [CarDreamer](https://github.com/ucd-dare/CarDreamer)
- [DreamerV3](https://github.com/danijar/dreamerv3)
- [CuriousReplay](https://github.com/AutonomousAgentsLab/curiousreplay)
- [Cosmos](https://arxiv.org/abs/2501.03575)
- [SafeDreamer](https://arxiv.org/abs/2307.07176)

