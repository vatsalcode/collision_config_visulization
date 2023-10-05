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

p.connect(p.DIRECT)
#p.setGravity(0,0,-9.81)

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


configs=np.random.rand(100,7)


# PCA reduction
pca = PCA(n_components=2)
reduced_data_pca = pca.fit_transform(configs)

# t-SNE reduction
reduced_data_tsne = TSNE(n_components=2).fit_transform(configs)

# Visualization using Matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(reduced_data_pca[:, 0], reduced_data_pca[:, 1])
plt.title('PCA Reduced Configurations')

plt.subplot(1, 2, 2)
plt.scatter(reduced_data_tsne[:, 0], reduced_data_tsne[:, 1])
plt.title('t-SNE Reduced Configurations')

plt.show()
