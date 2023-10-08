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

    def __init__(self,payload=0):
        self.joint_limits = joint_limits
        self.ee_pose = np.zeros(2)
        self.payload=payload

    def set_joint_positions(self, q):
        self.ee_pose = calculate_fk(q,self.payload)[:2]

    def render(self, ax, is_collided=False):
        color = 'g' if is_collided else 'r'
        ax.scatter(self.ee_pose[0], self.ee_pose[1], c=color, s=20, alpha=0.7)
            
# Forward kinematics
def calculate_fk(q,payload):
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
        
# Visualization
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs = axs.ravel()
payloads = [1, 2, 3]

for idx, payload in enumerate(payloads):
    # Robot instance
    robot = Robot(payload)
    checker = ValidityChecker(robot)
    ax = axs[idx]

    for _ in range(1000):
        q = [np.random.uniform(lim[0], lim[1]) for lim in joint_limits]

        if not checker.check_joint_limits(q):
            continue

        is_collided = not checker.check_collision_free(q)
        robot.set_joint_positions(q)
        robot.render(ax, is_collided)

    # Axis settings
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_title(f'Payload: {payload}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

# Global title and show plot
plt.suptitle("End-Effector Positions for Different Payloads", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout
plt.show()