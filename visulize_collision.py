import numpy as np
import random
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from pyb_utils.collision import NamedCollisionObject, CollisionDetector
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pybullet as p
import pybullet_data

#p.connect(p.GUI)
#p.setAdditionalSearchPath(pybullet_data.getDataPath())

#p.setGracv=

class SimpleRobot:
    def get_joint_limits(self):
        class Limits:
            pass
        
        limits = Limits()
        limits.angle = np.array([
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ])  # [q_min, q_max] for each joint in rad
        
        limits.velocity = np.array([
            2.1750,
            2.1750,
            2.1750,
            2.1750,
            2.6100,
            2.6100,
            2.6100
        ])  # dq_max for each joint in rad/s
        
        limits.acceleration = np.array([
            15,
            7.5,
            10,
            12.5,
            15,
            20,
            20
        ])  # ddq_max for each joint in rad/s^2
        
        # I'm assuming effort corresponds to tau_max in Nm
        limits.effort = np.array([
            87,
            87,
            87,
            87,
            12,
            12,
            12
        ])  # tau_max for each joint
        
        return limits

    
    def is_self_collision_free(self, config):
        # Simplified: assuming any config is collision-free
        return True
    
    def get_manipulability(self, config):
        return 0.1



@njit
def check_lims(vals, lims):
    return np.all(vals >= lims[:, 0]) and np.all(vals <= lims[:, 1])


class RobotStateValidityChecker:
    def __init__(self, robot):
        self.robot = robot

        self.robot_limits = robot.get_joint_limits()
        self.robot_limits.velocity = np.concatenate([[-self.robot_limits.velocity / 1.25],
                                                     [self.robot_limits.velocity / 1.25]]).T
        self.robot_limits.effort = np.concatenate([[-self.robot_limits.effort],
                                                   [self.robot_limits.effort]]).T
        self.robot_limits.acceleration = np.concatenate([[-self.robot_limits.acceleration],
                                                         [self.robot_limits.acceleration]]).T

    def check_self_collision(self, q):
        if len(q.shape) == 1:
            q = q.reshape((1, -1))

        for config in q:
            if not self.robot.is_self_collision_free(config):
                return True
        return False

    def check_env_collision(self, q):
        # TODO: check collisions with obstacles
        return False

    def check_collision_free(self, q):
        if len(q.shape) == 1:
            q = q.reshape((1, -1))

        self_collision = self.check_self_collision(q)
        robot_obj_collision = self.check_env_collision(q)
        return not self_collision and not robot_obj_collision

    def check_manipulability(self, q):
        if len(q.shape) == 1:
            q = q.reshape((1, -1))

        for config in q:
            manipulability = self.robot.get_manipulability(config)
            if manipulability < 0.05:
                return False
        return True

    @staticmethod
    def __check_limits(vals, lims):
        # print(vals)
        return check_lims(vals, lims)

    def check_q_limits(self, q):
        # print(q)
        return self.__check_limits(q, self.robot_limits.angle)

    def check_dq_limits(self, dq):
        return self.__check_limits(dq, self.robot_limits.velocity)

    def check_ddq_limits(self, ddq):
        return self.__check_limits(ddq, self.robot_limits.acceleration)

    def check_tau_limits(self, tau):
        return self.__check_limits(tau, self.robot_limits.effort)
    
    #checking for config q is valid 
    def is_valid(self, q):
        return(
            self.check_q_limits(q) and
            not self.check_self_collision(q) and
            not self.check_env_collision(q)
        )  


robot= SimpleRobot()
checker= RobotStateValidityChecker(robot)

#Random Config
num_samples = 1000
joint_limits = robot.get_joint_limits()
q_samples = np.random.uniform(low=joint_limits.angle[:, 0], high=joint_limits.angle[:, 1], size=(num_samples, joint_limits.angle.shape[0]))

# Placeholder for Labels
labels = np.zeros(num_samples)

# Validate and Label Configurations
for idx, q in enumerate(q_samples):
    if checker.check_q_limits(q) and checker.check_collision_free(q):
        labels[idx] = 1  # Mark as collision-free

# PCA
pca = PCA(n_components=2)  # Reduce to 2D for visualization
q_pca = pca.fit_transform(q_samples)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(q_pca[labels==1, 0], q_pca[labels==1, 1], c='g', marker='o', label='Collision-Free')
plt.scatter(q_pca[labels==0, 0], q_pca[labels==0, 1], c='r', marker='x', label='Collision')
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
q_tsne = tsne.fit_transform(q_samples)

plt.subplot(1, 2, 2)
plt.scatter(q_tsne[labels==1, 0], q_tsne[labels==1, 1], c='g', marker='o', label='Collision-Free')
plt.scatter(q_tsne[labels==0, 0], q_tsne[labels==0, 1], c='r', marker='x', label='Collision')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend()

plt.tight_layout()
plt.show()