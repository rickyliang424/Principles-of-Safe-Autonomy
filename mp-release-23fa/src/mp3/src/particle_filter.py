import numpy as np
from maze import Maze, Particle, Robot
import bisect
import rospy
from gazebo_msgs.msg import  ModelState
from gazebo_msgs.srv import GetModelState
import shutil
from std_msgs.msg import Float32MultiArray
from scipy.integrate import ode
import random
import math
import matplotlib.pyplot as plt
import pathlib
import os
import time

import signal # Import signal module

#signal handler function
should_exit = False
def SignalHandler_SIGINT(SignalNumber,Frame):
    global should_exit
    should_exit = True
    
signal.signal(signal.SIGINT,SignalHandler_SIGINT)
cur_dir = os.path.dirname(os.path.abspath(__file__))

def vehicle_dynamics(t, vars, vr, delta):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    
    dx = vr * np.cos(curr_theta)
    dy = vr * np.sin(curr_theta)
    dtheta = delta
    return [dx,dy,dtheta]

class particleFilter:
    def __init__(self, bob, world, num_particles, sensor_limit, x_start, y_start, argv):
        self.num_particles = num_particles  # The number of particles for the particle filter
        self.sensor_limit = sensor_limit    # The sensor limit of the sensor
        self.is_vis = argv.vis_prefix is not None
        self.vis_prefix = argv.vis_prefix
        particles = list()
        ##### TODO:  #####
        # Modify the initial particle distribution to be within the top-right quadrant of the world, and compare the performance with the whole map distribution.
        for i in range(num_particles):
            ## (Default) The whole map
            
            if argv.quadrant == 1:
                x = np.random.uniform(world.width // 2, world.width)
                y = np.random.uniform(0, world.height // 2)
            elif argv.quadrant == 2:
                x = np.random.uniform(0, world.width // 2)
                y = np.random.uniform(0, world.height // 2)
            elif argv.quadrant == 3:
                x = np.random.uniform(0, world.width // 2)
                y = np.random.uniform(world.height // 2, world.height)
            elif argv.quadrant == 4:
                x = np.random.uniform(world.width // 2, world.width)
                y = np.random.uniform(world.height // 2, world.height)
            elif argv.quadrant == -1: # uniform distribute
                x = np.random.uniform(0, world.width)
                y = np.random.uniform(0, world.height)
            else:
                print("Error in quadrant arg")
                exit(1)

            particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))
        ###############
        self.particles = particles          # Randomly assign particles at the begining
        self.bob = bob                      # The estimated robot state
        self.world = world                  # The map of the maze
        self.x_start = x_start              # The starting position of the map in the gazebo simulator
        self.y_start = y_start              # The starting position of the map in the gazebo simulator
        self.modelStatePub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.controlSub = rospy.Subscriber("/gem/control", Float32MultiArray, self.__controlHandler, queue_size = 1)
        self.control = []                   # A list of control signal from the vehicle
        self.control_start_idx = 0
        self.num_control = 0
        self.pos_errors = []
        self.heading_errors = []

    def __controlHandler(self,data):
        """
        Description:
            Subscriber callback for /gem/control. Store control input from gem controller to be used in particleMotionModel.
        """
        tmp = list(data.data)
        self.num_control += 1
        self.control.append(tmp)

    def getModelState(self):
        """
        Description:
            Requests the current state of the polaris model when called
        Returns:
            modelState: contains the current model state of the polaris vehicle in gazebo
        """
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            serviceResponse = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            modelState = serviceResponse(model_name='polaris')
        except rospy.ServiceException as exc:
            rospy.loginfo("Service did not process request: "+str(exc))
        return modelState

    def weight_gaussian_kernel(self, x1, x2, std = 5000):
        if x1 is None: # If the robot recieved no sensor measurement, the weights are in uniform distribution.
            return 1./len(self.particles)
        else:
            tmp1 = np.array(x1)
            tmp2 = np.array(x2)
            return np.sum(np.exp(-((tmp2-tmp1) ** 2) / (2 * std)))

    def updateWeight(self, readings_robot):
        """
        Description:
            Update the weight of each particles according to the sensor reading from the robot 
        Input:
            readings_robot: List, contains the distance between robot and wall in [front, right, rear, left] direction.
        """
        ## TODO #####
        weight_array = np.zeros((self.num_particles,1))
        for i in range(self.num_particles):
            reading_particle = self.particles[i].read_sensor()
            weight_array[i,0] = self.weight_gaussian_kernel(x1=reading_particle, x2=readings_robot)
        
        weight_array_norm = weight_array / np.sum(weight_array)
        for i in range(self.num_particles):
            self.particles[i].weight = weight_array_norm[i][0]
        
    def resampleParticle(self):
        """
        Description:
            Perform resample to get a new list of particles 
        """
        particles_new = list()
        ## TODO #####
        px, pw = np.zeros((3, self.num_particles)), np.zeros((1, self.num_particles))
        for i in range(self.num_particles):
            px[:,i] = self.particles[i].state
            pw[0,i] = self.particles[i].weight
        NP = self.num_particles
        w_cum = np.cumsum(pw)
        base = np.arange(0.0, 1.0, 1 / NP)
        re_sample_id = base + np.random.uniform(0, 1 / NP)
        indexes = []
        ind = 0
        for ip in range(NP):
            while re_sample_id[ip] > w_cum[ind]:
                ind += 1
            indexes.append(ind)
            p = self.particles[ind]
            new_p = Particle(x = p.x, y = p.y, heading=p.heading, maze = self.world, sensor_limit = p.sensor_limit, noisy = True)
            particles_new.append(new_p)
        
        self.particles = particles_new
        for i in range(self.num_particles):
            self.particles[i].weight = 1.0 / NP

    def particleMotionModel(self):
        """
        Description:
            Estimate the next state for each particle according to the control input from actual robot 
        """
        ## TODO #####
        control_end_idx = len(self.control)
        ts = 0.01
        
        for control in self.control[self.control_start_idx:control_end_idx]:
            v = control[0]
            delta = control[1]
            
            for i in range(self.num_particles):
                pt = self.particles[i]
                pt.x += ts * v * np.cos(pt.heading)
                pt.y += ts * v * np.sin(pt.heading)
                pt.heading += ts * delta
                
                # Check if the particle is still in the maze
                maze = self.world
                if pt.y < 0 or pt.y >= maze.num_rows or pt.x < 0 or pt.x >= maze.num_cols:
                    pt.x = np.random.uniform(0, maze.width)
                    pt.y = np.random.uniform(0, maze.height)
                        
        self.control_start_idx = control_end_idx

    def computeError(self):
        """ 
            Compute and save position/heading error to plot.
        """
        pos_error = 0.
        heading_error = 0.
        for particle in self.particles:
            dist = math.hypot(particle.x - self.bob.x, particle.y - self.bob.y)
            pos_error += dist
            heading_error += abs(self.bob.heading - particle.heading) % np.pi
        pos_error /= len(self.particles)
        heading_error /= len(self.particles)
        self.pos_errors.append(pos_error)
        self.heading_errors.append(heading_error)
        
    def plot(self):
        plt.figure(figsize=(10, 5))

        # Create the curve plot
        plt.subplot(121)
        plt.plot(self.pos_errors, label='position error', color='b', linestyle='-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Avg error')
        plt.title('Position Error')

        plt.subplot(122)
        plt.plot(self.heading_errors, label='heading error', color='r', linestyle='-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Avg error')
        plt.title('Heading Error')

        plt.tight_layout()

        OUTPUT_DIR = "./vis"
        pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        # plt.show()
        plt.savefig(os.path.join(OUTPUT_DIR, self.vis_prefix + ".png")) 

    
    def runFilter(self):
        """
        Description:
            Run PF localization
        """
        start_time = time.time()
        while True:
            ## TODO: (i) Implement Section 3.2.2. (ii) Display robot and particles on map. (iii) Compute and save position/heading error to plot. #####
            ## (i) Implement Section 3.2.2.

            if len(self.control) <= self.control_start_idx:
                continue

            # Compute and save position/heading error to plot.
            self.computeError()
            if self.is_vis and self.num_control >= 1000:
                self.plot()
                break

            print ("-------- num control: {} --------".format(self.num_control))

            self.readings_robot = self.bob.read_sensor()
            if self.readings_robot is None:
                continue
            
            self.particleMotionModel()

            self.updateWeight(self.readings_robot)
            pw = np.array([particle.weight for particle in self.particles])
            N_eff = 1.0 / (pw.dot(pw.T))  # Effective particle number
            print ("N_eff:", N_eff)
            if N_eff < self.num_particles / 1.5:
                print ("enter resample")
                self.resampleParticle()

            ## (ii) Display robot and particles on map.
            self.world.show_particles(particles = self.particles)
            self.world.show_robot(robot = self.bob)
            self.world.show_estimated_location(particles = self.particles)
            self.world.clear_objects()  
            
            # Avoid "ctrl + c" stucks occasionally
            if should_exit:
                exit(1)
        
        # Record the start time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nElapsed time: {elapsed_time:.2f} seconds")
        