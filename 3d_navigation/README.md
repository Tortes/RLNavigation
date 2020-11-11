# 3d navigation with reinforcement learning

- 环境: OpenAI Gym
- 物理引擎: Gazebo7
- 算法框架: tianshou
- Agent: Husky
- Env: Random generated 3d rough terrain
- Sensors: 
  - 2d lidar:           get 360 distance information
  - intel realsense:    get the front depth information
  - wheel encoder:      replicated for 
  - gps:                get the current robot position and goal position
  - imu:                get the current robot roll and pitch

### Modeling
- 当且仅当机器人 到达目标点 或 发生碰撞 或 roll pitch角度大于阈值时，结束当前navigation。
- 对于机器人observation，首先使用intel realsense的深度信息作为输入的一部分，用于检测前方的地形。其次，使用2d lidar检测机器人四周的距离信息，用于检测是否发生碰撞。另外还使用了IMU作为机器人姿势的检测，确保自身roll pitch不发生剧烈变化。最后同2d navigation一样，加入了机器人的朝向，机器人与目标位置的距离，机器人与目标位置矢量和机器人朝向的夹角来作为导航的信息。
- 当前任务的policy network和critic network都使用PointNet+LSTM的结构作为网络的主体，之后分别使用不同的全连接层输出policy和critic。这里首先将intel realsense的深度信息转化为点云数据，后通过均匀降采样将其转化为稀疏点云输入给PointNet满足实时性，接着将PointNet提取出的特征与其他observation连接，输入给LSTM学习序列信息。
- 算法使用了tianshou中的PPO作为强化学习框架，使用pytorch作为网络框架。
- 该任务中reward function设置为：
  - arrival：120
  - done：-100
  - roll > 22.5: -1
  - pitch > 22.5: -1
  - 每step固定: -0.5
  - 每step固定: $100*\Delta d$
- 对于机器人每个step的action，这里采用了离散值来加快计算：
  - linear: 0~0.5m/s(5等分)
  - angular:-1.0~1.0rad/s(11等分)


### Problems
- 为了场景高度信息的多样性，三维地图相较于二维地图场地规格会更大，从而导致navigation的距离变长（目标点随机生成于地图不同位置）。长距离的navigation导致reinforcement learning的搜索空间变大，应使用全局路径规划算法为其生成短距离路径点缩小搜索空间。