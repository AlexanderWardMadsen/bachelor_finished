# social_force_model.py
import numpy as np
import math

eps = 1e-8


class SocialForceModel:
    def __init__(self, tau=0.5, v0=1.3, sigma=0.3, B=2.1, lambda_factor=0.5, personal_radius=0.3):
        self.tau = tau # relaxation time
        self.v0 = v0 # desired speed (m/s)
        self.sigma = sigma # range of repulsive force (m)
        self.B = B # repulsive force strength
        self.lambda_factor = lambda_factor # anisotropy factor
        self.personal_radius = personal_radius # personal space radius (m)



    def desired_force(self, pos, goal, vel):
        # compute the vector pointing from current position to the goal
        direction = (goal - pos) # 2D vector
        # compute the length (magnitude) of this vector (magnitude = importance)
        dist = np.linalg.norm(direction) # scalar
        # if the distance is extremely small, treat it as zero:
        # this means the agent is already at the goal -> return no desired force
        if dist < eps:
            return np.zeros_like(pos)
        # normalize the direction vector (make it length 1)
        e = direction / dist # unit vector = 2D vector / scale
        # compute the desired acceleration according to the social force model:
        # move toward goal with desired speed v0 and relax towards it over timescale tau
        # v0*e=fast/direction, (v0*e)-vel=how much need adjust, (v0*e-vel)/tau=acceleration to correct velocity
        return (self.v0 * e - vel) / self.tau



    def repulsive_force(self, pos, vel, others):
        acc = np.zeros(2) # zero acceleration
        cutoff_distance = 5.0 * self.sigma # optimization: ignore distant people
        
        for o in others:
            direction = pos - o # vector from other to self
            dist = np.linalg.norm(direction) # length of vector
            
            # skip self or coincident positions to avoid 0/0
            if dist < eps:
                continue
            # skip if too far away (negligible force)
            elif dist > cutoff_distance:
                continue

            e = direction / dist # unit vector away from other person
            
            # Anisotropic weight: angle between velocity e and vector to other person
            vmag = np.linalg.norm(vel)
            if vmag > eps:
                vel_unit = vel / vmag # unit velocity vector
                cos_phi = np.dot(vel_unit, e) # cosine of angle
            else:
                cos_phi = 0.0
            anisotropy = (self.lambda_factor + (1 - self.lambda_factor) * (1 + cos_phi) * 0.5)

            # comfortable distance = 2 * personal_radius (two people's bubbles)
            r_comfortable = 2.0 * self.personal_radius
            
            # exponential repulsion: stronger when closer than comfortable distance
            # Acceleration = B * exp((r_comfortable - dist) / sigma) * e
            # (assuming unit mass since we can't detect mass from camera)
            accel_magnitude = self.B * np.exp((r_comfortable - dist) / self.sigma)
            
            acc += anisotropy * accel_magnitude * e # add effect from this other person and put together with all others
        
        return acc



    # single person step
    def step(self, pos, vel, goal, others, dt):
        # 1. Calculate where you WANT to go
        desired_acc = self.desired_force(pos, goal, vel)
        # Example: [1.0, 0.5] (northeast toward goal)
        
        # 2. Calculate TOTAL push from all people
        repulsive_acc = self.repulsive_force(pos, vel, others)
        # Example: [-0.8, 0.2] (avoid person to your left)
        
        # 3. Combine both effects
        total_acc = desired_acc + repulsive_acc
        # = [1.0, 0.5] + [-0.8, 0.2] = [0.2, 0.7]
        # → You go northeast, but curve away from nearby person
        
        # 4. Update velocity and position
        new_vel = vel + total_acc * dt
        new_pos = pos + vel * dt + 0.5 * total_acc * dt*dt
        
        return new_pos, new_vel


