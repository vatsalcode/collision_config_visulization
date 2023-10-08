import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# Joint limits
joint_limits = np.array([
   [-2.8973, 2.8973],
   [-1.7628, 1.7628],
   [-2.8973, 2.8973],  
   [-3.0718, -0.0698],
   [-2.8973, 2.8973],
   [-0.0175, 3.7525],
   [-2.8973, 2.8973] 
])

# Robot class 
class Robot:

    def __init__(self):
        self.joint_limits = joint_limits
        self.ee_pose = np.zeros(2)

    def set_joint_positions(self, q):
        self.ee_pose = calculate_fk(q)[:2]

    def render(self, ax, is_collided=False):
        color = 'g' if is_collided else 'r'
        ax.scatter(self.ee_pose[0], self.ee_pose[1], c=color)
            
# Forward kinematics
def calculate_fk(q):
    return np.array([q[0], q[1], 0])
            
# Limit checking
@njit
def check_limits(vals, lims):
  for i in range(len(vals)):
    if vals[i] < lims[i,0] or vals[i] > lims[i,1]:
      return False
  return True
  
# Validity checker
class ValidityChecker:

    def __init__(self, robot):
        self.robot = robot
        self.limits = self.robot.joint_limits

    def check_joint_limits(self, q):
        return check_limits(q, self.limits) 

    def check_collision_free(self, q):
        # Check self collisions
        for i in range(len(q)):
            for j in range(i+1, len(q)):
                d = np.abs(q[i] - q[j])
                if d < 0.5:
                    return False
        return True
        
# Create instance        
robot = Robot()
checker = ValidityChecker(robot)

# Visualize configs 
fig, ax = plt.subplots()

for i in range(1000):
    q = [np.random.uniform(lim[0], lim[1]) for lim in joint_limits]
    
    if not checker.check_joint_limits(q):
        continue
        
    is_collided = not checker.check_collision_free(q)
    
    robot.set_joint_positions(q)
    if is_collided:
        robot.render(ax, True)
    else:
        robot.render(ax)
        
ax.set_xlim([-4,5])
ax.set_ylim([-4,5])
plt.show()


# fig, ax = plt.subplots(figsize=(6,6))
# ax.set_title("Robot Configurations") 
# ax.set_xlabel("X")
# ax.set_ylabel("Y")

# for i in range(10000):
#    q = sample_config()
#    collision = check_collision(q)

#    color = cm.jet(i/1000) 
#    if collision:
#        ax.scatter(x, y, s=30, c=color, alpha=1.0) 
#    else:
#        ax.scatter(x, y, s=30, c=color, alpha=0.3)
       
#    # Draw arm
#    ax.plot([0,x], [0,y], c=color)
       
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.axis('square')