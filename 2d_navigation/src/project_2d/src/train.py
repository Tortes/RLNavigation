import os
import torch
import pprint
import argparse
import datetime
import numpy as np
import gym
import gym_gazebo
from torch.utils.tensorboard import SummaryWriter

from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic
# from tianshou.utils.net.common import Net
from tianshou.utils.net.common import Recurrent

from environment_list import Env
# from net import Net
import rospy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_training', type=bool, default=True)
    parser.add_argument('--linear_velocity', type=float, default=0.5)
    parser.add_argument('--angular_velocity', type=float, default=1.0)
    parser.add_argument('--realsense_max_point', type=int, default=5000)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    #
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--collect-per-step', type=int, default=100)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=128)
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

def train(args=get_args()):

    # Get Env
    env = Env(args)
    train_env = Env(args)
    test_env = Env(args)

    # Get args
    action_shape = env.action_dim
    state_shape = env.state_dim

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_env.seed(args.seed)
    test_env.seed(args.seed)

    # Def net
    # net = Net(args.layer_num, 
    #           state_shape, 
    #           action_shape, 
    #           device=args.device)
    net = Recurrent(args.layer_num, 
              state_shape, 
              action_shape, 
              device=args.device)

    actor = Actor(net, action_shape, hidden_layer_size=action_shape).to(args.device)
    critic = Critic(net, hidden_layer_size=action_shape).to(args.device)
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor, critic, optim, dist, args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        action_range=None,
        reward_normalization=False)
    
    # Def Collector
    train_collector = Collector(policy, 
                                train_env, 
                                ReplayBuffer(args.buffer_size, ignore_obs_next=True))
    test_collector = Collector(policy, test_env)

    # Def logger
    writer = SummaryWriter(args.logdir + '/' + 'ppo')

    # Load Model
    if args.load_model:
        policy.load_state_dict(torch.load(args.load_model))
    
    # Set stop fn
    def stop_fn(x):
        return x >= 10

    # Trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, stop_fn=stop_fn, writer=writer)

    # Saver
    if args.save_path:
        filename = os.path.join(args.save_path, datetime.date.today(), 'ppo_discrete.pth')
        torch.save(policy.state_dict(), filename)
    else:
        torch.save(policy.state_dict(), 'ppo_discrete.pth')
    
    train_collector.close()
    test_collector.close()

    if __name__ == '__main__':
        pprint.pprint(result)
        env = Env(args.is_training)
        collector = Collector(policy, env)
        result = collector.collect(n_step=2000, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()

if __name__ == '__main__':
    rospy.init_node('ppo')
    train()
