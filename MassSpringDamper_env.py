import gym
#import gym_environments.envs

from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

#UNCOMMENT if you intend to use the manipulator or GYM-ROS interface.
#import ros_message_listener.eavesdrop as eavesdrop
#from ManipulatorAction import ManipulatorAction

class MassSpringDamperEnv(gym.Env):
	metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
	def __init__(self): #, spring_stiffness, damper_factor, mass, max_force
		self.spring_stiffness = 1.5 #spring_stiffness
		self.damper_factor = 6 #damper_factor
		self.mass = 1.5 #mass
		self.max_force = np.array([2]) #max_force
		self.step_length = 0.15 # in seconds

		self.x_treshold = 4.8
		obs_high = np.array([self.x_treshold,np.finfo(np.float32).max])

		self.observation_space = spaces.Box(-obs_high,obs_high, dtype=np.float32)
		self.action_space = spaces.Box(-self.max_force,self.max_force,dtype=np.float32)

		self.state = None # Some random init here?
		self.goal_state = None  # Some position with no velocity

		self.viewer = None
		self.steps_beyond_done = None


	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state
		x, x_dot = state
		acceleration = (-self.spring_stiffness*x -self.damper_factor*x_dot + action)/self.mass
		x_dot = acceleration*self.step_length
		x = x_dot*self.step_length
		self.state = (x,x_dot)
		done = False
		if x == self.goal_state[0] and x_dot == self.goal_state[1]:
			done = True
		if x < -self.x_treshold or x > self.x_treshold:
			done = True

		if not done:
			reward = -np.abs(x - self.goal_state[0])**2 - np.abs(x_dot - self.goal_state[1])**2
		elif self.steps_beyond_done is None:
			self.steps_beyond_done = 0
			reward = -np.abs(x - self.goal_state[0])**2 - np.abs(x_dot - self.goal_state[1])**2
		else:
			if self.steps_beyond_done == 0:
				logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
			self.steps_beyond_done += 1
			reward = 0.0

		return np.array(self.state), reward, done,  {}

	def reset(self,goal_x=0.0,goal_x_dot=0.0):
		self.state = (np.random.uniform(low = -self.x_treshold, high = self.x_treshold),np.random.uniform(low = -0.1, high = 0.1))
		self.steps_beyond_done = None
		self.goal_state =(goal_x,goal_x_dot)
		return np.array(self.state)

	def render(self, mode='human'):
		screen_width = 600
		screen_height = 400

		world_width = self.x_treshold*2
		scale = screen_width/world_width

		mass_y =100.0
		mass_width = 50.0
		mass_height = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -mass_width/2, mass_width/2,mass_height/2, -mass_height/2
			axleoffset = mass_height/4.0
			mass = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
			self.mass_trans = rendering.Transform()
			mass.add_attr(self.mass_trans)
			self.viewer.add_geom(mass)
			self.track = rendering.Line((0,mass_y), (screen_width,mass_y))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

		if self.state is None: return None

		x = self.state
		mass_x = x[0]*scale+screen_width/2.0 # MIDDLE OF MASS
		self.mass_trans.set_translation(mass_x,mass_y)

		return self.viewer.render(return_rgb_array = mode =='rgb_array')

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None

	def get_obs(self):

		return self.state

if __name__ == '__main__':

	#Environment unit test!
	#Cant be executed before environment is registered with gym, in envs/__init__.py
	 env = gym.make('MassSpringDamper-v0')

	 env.step(action_space.get_random_action())

	 env.reset()
