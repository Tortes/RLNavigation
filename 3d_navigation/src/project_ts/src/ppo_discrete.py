'''
8.18
- short onpolicy sampling to 1 batch(trainer->onpolicy.py (110))

'''

import torch
import pprint
import argparse
import numpy as np
import gym
import gym_gazebo
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PPOPolicy
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils.net.common import Net
from tianshou.utils.net.common import Recurrent

from gymenv_discrete import Env
import rospy

# from atari import create_atari_environment, preprocess_fn

is_training = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--collect-per-step', type=int, default=100)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=8)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=1.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--max_episode_steps', type=int, default=2000)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(args=get_args()):
    env = Env(is_training)
    args.action_shape = env.action_dim
    args.state_shape = env.state_dim
    
    train_envs = Env(is_training)
    test_envs = Env(is_training)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    # net = Net(args.layer_num, args.state_shape, args.action_shape, device=args.device)
    net = Recurrent(args.layer_num, args.state_shape, args.action_shape, device=args.device)
    actor = Actor(net, args.action_shape, hidden_layer_size=args.action_shape).to(args.device)
    critic = Critic(net, hidden_layer_size=args.action_shape).to(args.device)
    optim = torch.optim.Adam(list(
        actor.parameters()) + list(critic.parameters()), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor, critic, optim, dist, args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        action_range=None,
        reward_normalization=False)
    # collector
    train_collector = Collector(
        policy, train_envs, ReplayBuffer(args.buffer_size, ignore_obs_next=True))
    test_collector = Collector(policy, test_envs)
    # log
    writer = SummaryWriter(args.logdir + '/' + 'ppo')

    def stop_fn(x):
        return x >= 6000

    policy.load_state_dict(torch.load('ppo_discrete_5000.pth'))

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, stop_fn=stop_fn, writer=writer)
    
    # Saver
    torch.save(policy.state_dict(), 'ppo_discrete.pth')

    train_collector.close()
    test_collector.close()
    
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        # env = create_atari_environment(args.task)
        env = Env(is_training)
        collector = Collector(policy, env)
        result = collector.collect(n_step=2000, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    rospy.init_node('ppo')
    test_ppo()
