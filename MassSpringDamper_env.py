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
	def __init__(self): #, spring_stiffness, damper_factor, mass, max_force4
		self.x_treshold = 4.8
		self.spring_stiffness = 0.5 #spring_stiffness
		self.damper_factor = 0.25 #damper_factor
		self.mass = 1 #mass
		self.max_force = np.array([1]) #max_force
		self.step_length = 0.1 # in seconds

		obs_high = np.array([self.x_treshold,np.finfo(np.float32).max])#,self.x_treshold,np.finfo(np.float32).max])

		self.observation_space = spaces.Box(-obs_high,obs_high, dtype=np.float32)
		self.action_space = spaces.Box(-self.max_force,self.max_force,dtype=np.float32)
		self.state = None # Some random init here?
		self.goal_state = None  # Some position with no velocity

		self.viewer = None
		self.steps_taken = None
		self.steps_beyond_done = None


	def step(self, action):
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		state = self.state
		x, x_dot = state
		acceleration = (-self.spring_stiffness*x -self.damper_factor*x_dot + action)/self.mass

		x_dot += acceleration*self.step_length
		x += x_dot*self.step_length
		self.state = (x[0],x_dot[0]) #!!

		x_goal,x_dot_goal = self.goal_state
		done = False
		reward = -np.abs(x - x_goal)**2

		#print("Reward = -(",x,"-",x_goal,"^2 - (" ,x_dot," -" ,x_dot_goal,")^2 = ", reward)
		if (np.abs(x- x_goal) < 0.1):# and (np.abs(x_dot -x_dot_goal)< 0.01):
			#print("YAY! Managed to arrive at righ.")
			reward = 1
			if (np.abs(x_dot -x_dot_goal)< 0.1):
				reward = 100
				if (np.abs(x_dot -x_dot_goal)< 0.01):

					#print("Getting there")
					reward = 500
					if (np.abs(x_dot -x_dot_goal)< 0.005):
						reward = 1000
						done = True
						print("YAY! Stopped at right position - ",x,x_dot,self.steps_taken)
			#return np.array(self.state), reward, done,  {}

		if x < -self.x_treshold or x > self.x_treshold:
			reward = -100
			done = True
		if done:
			if self.steps_beyond_done is None:
				self.steps_beyond_done = 0
			else:
				if self.steps_beyond_done == 0:
					logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
					self.steps_beyond_done += 1
					reward = 0.0
		if self.steps_taken == None:
			self.steps_taken = 1
		else:
			self.steps_taken += 1
			if self.steps_taken == 200:
				done = True
				self.steps_taken = None
		return np.array(self.state), reward, done,  {}

	def reset(self,goal_x=-2.0,goal_x_dot=0.0):
		self.state = (np.random.uniform(low = -self.x_treshold/2.0, high = self.x_treshold/2.0),np.random.uniform(low = -0.01, high = 0.01))
		self.steps_taken = None
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
			goal_mass = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
			goal_mass.set_color(100,0,0)
			self.goal_mass_trans = rendering.Transform()
			goal_mass.add_attr(self.goal_mass_trans)
			self.viewer.add_geom(goal_mass)

			mass = rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
			self.mass_trans = rendering.Transform()
			mass.add_attr(self.mass_trans)
			self.viewer.add_geom(mass)



			self.track = rendering.Line((0,mass_y), (screen_width,mass_y))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

			### ADD SPRING
			#p1 = [l,(t-b)/2.0]
			#x = self.state
			#mass_x = x[0]*scale+screen_width/2.0
			#left = mass_x - mass_width/2.0
			#y_m = mass_y + 0.25*mass_height
			#y_t = mass_y + 0.5*mass_height
			#y_b = mass_y

			#self.p1 = rendering.Line(((3/4)*left,y_m),(left,y_m))
			#self.p1.set_color(100,0,0)
			#self.viewer.add_geom(self.p1)


		if self.state is None: return None

		x = self.state
		print(x, self.goal_state)
		mass_x = x[0]*scale+screen_width/2.0 # MIDDLE OF MASS
		self.mass_trans.set_translation(mass_x,mass_y)
		self.goal_mass_trans.set_translation(self.goal_state[0]*scale + screen_width/2.0,mass_y)

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
