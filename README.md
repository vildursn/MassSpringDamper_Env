# Custom gym environment: Mass-spring-damper-system

### TODO: Visualize spring and damper

### OBS: Baselines/spinningups  PPO/TRPO/ACKTR does not take action limits into account. So must either change agent (clipping action), or the environmnet(allow all forces) for it to work.

A custom made gym environment for the classic, super simple control problem of a mass spring damper system. 




1. Add `MassSpringDamper_env.py` to .../gym/envs/classic_control, ... being wherever your gym installation is.

2. Add the following line
```
from gym.envs.classic_control.MassSpringDamper_env import MassSpringDamperEnv
```
to `gym/envs/classic_control/__init__.py`

3. Add the following lines
```
register(
    id='MassSpringDamper-v0',
    entry_point='gym.envs.classic_control:MassSpringDamperEnv',
)
```

to `gym/envs/__init__.py`

4. Run it like a normal gym environment! For example:
```
import gym
import gym.envs

env = gym.make('MassSpringDamper-v0')

env.reset(goal_x, goal_x_dot) # if no goal is stated it is set to 0,0
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action) # take a random action
env.close()
```


