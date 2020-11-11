#!/home/tortes/anaconda3/envs/ts/bin/python
import rospy
import gym
import gym_gazebo
import numpy as np
import tensorflow as tf
from ddpg import *
from environment import Env

exploration_decay_start_step = 5000
state_dim = 48008
action_dim = 2
action_linear_max = 5.  # m/s
action_angular_max = 2.  # rad/s
is_training = True


def main():
    rospy.init_node('ddpg_stage_1')
    env = Env(is_training)
    agent = DDPG(env, state_dim, action_dim)
    past_action = np.array([0., 0.])
    print('State Dimensions: ' + str(state_dim))
    print('Action Dimensions: ' + str(action_dim))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    if is_training:
        print('Training mode')
        avg_reward_his = []
        total_reward = 0
        var = 1.
        cnt = 0

        while True:
            state = env.reset()
            one_round_step = 0
            cnt = cnt + 1

            while True:
                a = agent.action(state)
                a[0] = np.clip(np.random.normal(a[0], var), 0., 1.)
                a[1] = np.clip(np.random.normal(a[1], var), -0.5, 0.5)

                state_, r, done, arrive = env.step(a, past_action)
                time_step = agent.perceive(state, a, r, state_, done)

                if arrive:
                    result = 'Success'
                else:
                    result = 'Fail'

                if time_step > 0:
                    total_reward += r

                if time_step % 10000 == 0 and time_step > 0:
                    print('---------------------------------------------------')
                    avg_reward = total_reward / 10000
                    print('Average_reward = ', avg_reward)
                    avg_reward_his.append(round(avg_reward, 2))
                    print('Average Reward:',avg_reward_his)
                    total_reward = 0

                if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                    var *= 0.9999

                past_action = a
                state = state_
                one_round_step += 1

                if arrive:
                    # print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)
                    print('----------No %i Arrive----------' % cnt)
                    print('Step:        %3i' % one_round_step)
                    print('Var:         %.2f' % var)
                    print('Time step:   %i' % time_step)
                    print('Finish time: %3is' % env.time)
                    print('Reward:      %3i' % total_reward)
                    one_round_step = 0
                    state = env.reset()

                if done or one_round_step >= 5000:
                    # print('Step: %3i' % one_round_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)
                    print('----------No %i Collision----------' % cnt)
                    print('Step:        %3i' % one_round_step)
                    print('Var:         %.2f' % var)
                    print('Time step:   %i' % time_step)
                    print('Finish time: %3is' % env.time)
                    print('Reward:      %3i' % total_reward)
                    break

    else:
        print('Testing mode')
        while True:
            state = env.reset()
            one_round_step = 0

            while True:
                a = agent.action(state)
                a[0] = np.clip(a[0], 0., 1.)
                a[1] = np.clip(a[1], -0.5, 0.5)
                state_, r, done, arrive = env.step(a, past_action)
                past_action = a
                state = state_
                one_round_step += 1


                if arrive:
                    print('----------Arrive----------')
                    print('Step:        %3i' % one_round_step)
                    print('Finish time: %3is' % env.time)
                    print('Reward:      %3i' % r)
                    one_round_step = 0
                    state = env.reset()

                if done:
                    print('----------Collision----------')
                    print('Step:        %3i' % one_round_step)
                    print('Finish time: %3is' % env.time)
                    print('Reward:      %3i' % r)
                    break


if __name__ == '__main__':
     main()
