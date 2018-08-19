import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.sim.reset()
        self.sim.init_pose = self.sim.pose
        self.action_repeat = 4

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        #self.target = np.append(self.target_pos, np.array([0,0,0]))
        
        if np.linalg.norm(self.sim.init_pose[:3] - self.target_pos) < 1.0:
            self.norm_targ_dist = 1.0
        else:
            self.norm_targ_dist = np.linalg.norm(self.sim.init_pose[:3] - self.target_pos) 
    
    def get_reward(self,rotor_speeds):
        """Uses current pose of sim to return reward."""
        s0 = abs(self.target_pos-self.sim.pose[:3])
        #s_xyz= np.sign(s0)*np.log(np.absolute(s0)+1)
        #norm_dist = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        s_thetapsiphi = np.copy(self.sim.pose[3:])
        s_thetapsiphi[s_thetapsiphi>np.pi] -= 2*np.pi
        s_thetapsiphi /= np.pi
        #reward = np.tanh(1. - 0.3 * np.power(np.log(s0[:3] + 1),2.0).sum())
        #reward = np.tanh(1. - 0.6 * (np.log(s0[2] + 1))**2.0 - 0.3 *np.power(np.log(np.power(np.power(s0[:2],2.0).sum(),0.5)+1.0),2.0)- 0.01*abs(s_thetapsiphi).sum())
        reward = np.tanh(2.0 - 0.6 * (np.log(s0[2] + 1))**2.0 - 0.3 * np.power(np.log(np.power(np.power(s0[:2],2.0).sum(),0.5)+1.0),2.0) - 0.00*abs(self.sim.angular_accels).sum()- 0.00*abs(self.sim.linear_accel).sum()- 0.00*abs(s_thetapsiphi).sum())/self.action_repeat
        #reward = np.tanh(2. - 0.3 * (np.log(s0[2] + 1))**2.0 - 0.15 *np.power(np.log(np.power(np.power(s0[:2],2.0).sum(),0.5)+1.0),2.0)- 0.01*abs(self.sim.angular_v).sum())
        #reward = np.tanh(1. - 0.6 * (np.log(s0[2] + 1))**2.0 - 0.3 *np.power(np.log(np.power(np.power(s0[:2],2.0).sum(),0.5)+1.0),2.0)- 0.001*abs(self.sim.angular_v.sum()))
        #reward = np.tanh(1. - 0.3 * np.log(s0[2] + 1)**2.0 - 0.05 * np.power(np.log(s0[:2] + 1),2.0).sum())
        #reward = np.tanh(1.0 - 0.8 * np.log(norm_dist + 1)**1.0 - 1* abs(s_thetapsiphi).sum())
#         reward = np.tanh(1.0 - 0.3 * np.log(norm_dist + 1) - .1 * abs(s_thetapsiphi).sum())
        #reward = 1. - 0.2* (norm_dist/self.norm_targ_dist) - .2 * abs(s_thetapsiphi).sum()
        #reward = 1.0 - 0.1* (norm_dist) - 0.3 * abs(self.sim.pose[:3]).sum()
#         if reward < -2.0:
#             reward = -2.0
        #if np.max(s0) > 20:
        #    reward -=0.5
        #return np.around(reward,decimals=2)
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward(rotor_speeds)
#             if done:
#                 reward -= 0.5            
            #normalize the distance
            #s_xyz = (self.sim.pose[:3]-self.target_pos)/self.norm_targ_dist
            s0 = (self.sim.pose[:3] - self.target_pos)
            s_xyz= np.sign(s0)*np.log(np.absolute(s0)+1)
            
            # setting the range from [0, 2pi] to [-pi,pi]
            s_thetapsiphi=np.copy(self.sim.pose[3:])
            s_thetapsiphi[s_thetapsiphi>np.pi] -= 2*np.pi
            s_thetapsiphi /= np.pi
            
            #append all xyz and thetapsiphi to the list
            pose_all.append(np.concatenate([s_xyz,s_thetapsiphi]))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #s_xyz = (self.sim.pose[:3]-self.target_pos)/self.norm_targ_dist
        s0 = (self.sim.pose[:3] - self.target_pos)
        s_xyz= np.sign(s0)*np.log(np.absolute(s0)+1)
        
        # setting the range from [0, 2pi] to [-pi,pi]
        s_thetapsiphi=np.copy(self.sim.pose[3:])
        s_thetapsiphi[s_thetapsiphi>np.pi] -= 2*np.pi
        s_thetapsiphi /= np.pi
        
        state = np.concatenate([s_xyz,s_thetapsiphi]* self.action_repeat)
        #state = (np.concatenate([self.sim.pose] * self.action_repeat)
        return state

