
import math
from geometry_msgs.msg import Point, Vector3

class SocialForceModel:
    def __init__(self, tau=0.5, v0=1.3, sigma=0.3, B=2.1, lambda_factor=0.5,
                 personal_radius=0.3):
        # Relaxation time, desired speed, and anisotropy factor
        self.tau = tau
        self.v0 = v0
        self.lambda_factor = lambda_factor
        # Social force parameters (pedestrian-pedestrian)
        self.sigma = sigma
        self.B = B
        self.personal_radius = personal_radius  # assumed identical for simplicity



    # Calculate desired force toward goal for person i
    def _compute_desired_force(self, pi, vi, goal):
        # Desired direction toward goal i
        dx = goal.x - pi.x
        dy = goal.y - pi.y
        dist_goal = math.hypot(dx, dy)
        if dist_goal > 0:
            ex = dx / dist_goal
            ey = dy / dist_goal
        else:
            ex = ey = 0.0

        # Desired velocity vector
        v_des_x = self.v0 * ex
        v_des_y = self.v0 * ey

        # Desired force F_des = (v_des - v) / tau
        fx = (v_des_x - vi.x) / self.tau
        fy = (v_des_y - vi.y) / self.tau
        
        return fx, fy



    # Calculate repulsive forces from other pedestrians
    def _compute_repulsive_force(self, pi, vi, ped_positions, i):
        fx = 0.0
        fy = 0.0
        n = len(ped_positions)
        cutoff_distance = 5.0 * self.sigma # optimization: ignore distant people
        
        # Repulsive force from other pedestrians
        for j in range(n):
            if j == i: 
                continue
            pj = ped_positions[j]
            # Vector from j to i
            dx = pi.x - pj.x
            dy = pi.y - pj.y
            dist = math.hypot(dx, dy)
            if dist <= 0.0:
                continue  # avoid division by zero
            # skip if too far away (negligible force)
            if dist > cutoff_distance:
                continue
            # Normalized direction
            nx = dx / dist
            ny = dy / dist

            # Anisotropic weight: angle between vi direction and vector to j
            vmag = math.hypot(vi.x, vi.y)
            if vmag > 0:
                vx_unit = vi.x / vmag
                vy_unit = vi.y / vmag
                cos_phi = vx_unit*nx + vy_unit*ny
            else:
                cos_phi = 0.0
            anisotropy = (self.lambda_factor + (1 - self.lambda_factor) * (1 + cos_phi) * 0.5)

            # Interpersonal distance including radii
            r_ij = self.personal_radius * 2.0
            # Repulsive force magnitude (exponential)
            f_ij = self.B * math.exp((r_ij - dist) / self.sigma)
            # Add to total forces
            fx += anisotropy * f_ij * nx
            fy += anisotropy * f_ij * ny
        
        return fx, fy


    
    #Compute next positions and velocities for all pedestrians.
    #param ped_positions: list of geometry_msgs.msg.Point (current positions)
    #param ped_velocities: list of geometry_msgs.msg.Vector3 (current velocities)
    #param goals: list of geometry_msgs.msg.Point (goal position for each pedestrian)
    #param obstacles: optional list of geometry_msgs.msg.Point (obstacle points)
    #param dt: time step (seconds)
    #return: (new_positions, new_velocities) as lists of Point and Vector3
    def update(self, ped_positions, ped_velocities, goals, obstacles=None, dt=0.1):
        n = len(ped_positions)
        # Prepare outputs
        new_positions = [Point() for _ in range(n)]
        new_velocities = [Vector3() for _ in range(n)]

        for i in range(n):
            # Current state of pedestrian i
            pi = ped_positions[i]
            vi = ped_velocities[i]
            goal = goals[i]

            # Calculate desired force toward goal
            fx_desired, fy_desired = self._compute_desired_force(pi, vi, goal)

            # Calculate repulsive forces from other people
            fx_repulsive, fy_repulsive = self._compute_repulsive_force(pi, vi, ped_positions, i)

            # Total force = desired + repulsive
            fx = fx_desired + fx_repulsive
            fy = fy_desired + fy_repulsive

            # Total acceleration on i is (fx, fy). Update velocity.
            new_vx = vi.x + fx * dt
            new_vy = vi.y + fy * dt

            # Update position
            new_x = pi.x + new_vx * dt
            new_y = pi.y + new_vy * dt

            # Assign to output lists
            new_positions[i].x = new_x
            new_positions[i].y = new_y
            new_positions[i].z = pi.z  # assuming 2D movement (z unchanged)
            new_velocities[i].x = new_vx
            new_velocities[i].y = new_vy
            new_velocities[i].z = 0.0

        return new_positions, new_velocities
