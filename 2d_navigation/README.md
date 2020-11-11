# 2d navigation with reinforcement learning

- 环境： OpenAI Gym
- 物理引擎： Gazebo7
- Agent： Turtlebot3 BURGER
- Sensors: 2d Lidar, gps(use gazebo to get the precise localization)
- Env: turtlebot3_stage_1(for pre-training), turtlebot3_stage_4(for training)

## DDPG version
DDPG的版本主要参考了[Motion planner using DDPG](https://github.com/Tortes/MotionPlannerUsingDDPG)这个仓库的代码，复现了[Virtual-to-real Deep Reinforcement Learning:
Continuous Control of Mobile Robots for Mapless Navigation](https://arxiv.org/pdf/1703.00420.pdf)中的网络和参数。  

### Modeling

- 采用MDP（State，Action，Transition Probability，Reward）建模，其中State由机器人感知的状态以及环境给出的状态构成，机器人感知的状态构成每步step中的observation，而环境给出的状态由reward function体现出来。具体而言，这里机器人的observation由2d lidar数据，机器人的朝向，机器人与目标位置的距离，机器人与目标位置矢量和机器人朝向的夹角组成。
- 当且仅当机器人 到达目标点 或 发生碰撞 时，结束当前navigation。
- 该任务中reward function设置为：
  -  到达：500
  -  碰撞：-450
  -  每step固定：-0.5
  -  每step固定：$500*\Delta d$

- 其中每step固定0.5为鼓励机器人尽早到达目标点，另外$500*\Delta d$为鼓励机器人接近目标点。  

- 对于机器人每个step的action，这里采用了连续值：

   - linear: 0~0.5m/s
   - angular: -1.0~1.0rad/s

## PPO version
PPO的版本主要使用了tianshou作为RL算法框架，目标为复现Pan Jia老师[Distributed multi-robot collision avoidance via deep reinforcement learning for navigation in complex scenarios](https://arxiv.org/abs/1808.03841)中单机器人场景的训练，目前使用pytorch复现文章中网络结构完成，但存在多步训练之后网络权重异常，仅输出角速度而线速度为0的情况。故目前使用LSTM作为替代policy网络进行训练。

### Modeling
- 考虑到DDPG版本中机器人有一定概率在无法抵达目标时会选择passive policy，（这里表现为主动撞墙结束当前navigation），故在PPO版本中添加了碰撞的惩罚，即当机器人与障碍物的距离小于一定阈值，则停止机器人并给予一定惩罚，同时使机器人回到安全的state继续执行任务。（这里使用的安全策略为原地转向，当且仅当机器人搭载的lidar前方180度内距离小于阈值判定为碰撞，故原地转向一定能找到一个安全的状态。）
- 当且仅当机器人 到达目标点 或 超时 时，结束当前navigation。
- 该任务中reward function设置为：
  - 到达：25
  - 碰撞（不结束当前navigation）：-5
  - 超时：-25
  - 每step固定：$-0.3*abs(ang_vel)$
  - 每step固定：$2.5*\Delta d$

- 其中每step固定$-0.3*abs(ang_vel)$为限制机器人过多的左右转，从而使其更快地到达目标位置。
- 对于机器人每个step的action，这里采用了离散值来加快计算：
  - linear: 0~0.5m/s(5等分)
  - angular:-1.0~1.0rad/s(11等分)

### Visualization
- 使用ROS中hector_slam包中的hector_trajectory_server记录机器人轨迹并在RViZ中显示，并在hector_slam包中添加删除轨迹的service，从而在机器人到达目标点或超时之后删除已有路径并重新记录路径点。