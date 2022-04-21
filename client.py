#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import argparse
import glob
import gym
import os
import pickle
import socket
import sys
import time
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.utils import set_random_seed
from utils import AGENTS


class Client():
    def __init__(self, args):
        self.ctrl = {'verbose': args.verbose,
                     'interval': args.interval * 1e6,
                     'tolerance': args.tolerance * 1e6,
                     'supported_envs': [],
                     'env_name': args.env,
                     'env': None,
                     'agent_name': args.agent,
                     'agent': None,
                     'training': args.training,
                     'timesteps': args.timesteps}

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((args.address, args.port))
        env_list, env_detail = self.get_supported_envs()
        self.ctrl['supported_envs'] = env_detail

        if args.list_envs:
            print(env_list)
            sys.exit(0)

        if self.ctrl['env_name'] not in self.ctrl['supported_envs']:
            print('Requested environment not supported by both endpoints')
            sys.exit(1)

        self.ctrl['env'] = gym.make(self.ctrl['env_name'])

        if self.ctrl['training'] == 'offline':
            self.ctrl['agent'] = self.load_trained_agent()
        elif self.ctrl['training'] == 'online':
            print("Training...")
            self.ctrl['agent'] = self.train_an_agent(self.ctrl['timesteps'])
        else:
            print('Training type not supported')
            sys.exit(1)

        try:
            self.run()
        except KeyboardInterrupt:
            self.sock.send(pickle.dumps({'finish': None}))
            self.sock.close()
            sys.exit(0)

    def run(self):
        reply, _ = self.request({'init': {'env_name': self.ctrl['env_name'],
                                          'agent_name': self.ctrl['agent_name']}})
        if reply.get('init', '') != 'OK':
            print('Failed to initialize remote environment')
            sys.exit(0)
        
        tic = time.time() #counter init
        episode_times = [] #vector to calculate the mean time and standard deviation of the first 100 episodes
        episode_lenghts = [] #vector to calculate the mean reward and standard deviation of the first 100 episodes
        timeouts_n = 0 #number of episodes that were not completed
        ep_len = 0 #variable to count number of actions at each episodes
        ep_num = 1 #episode number
        reply, _ = self.request({'reset': None})
        
        while True:
            try:
                # Busy waiting, taking tolerance out of deadline
                deadline = time.time_ns() + self.ctrl['interval']
                while deadline - self.ctrl['tolerance'] > time.time_ns():
                    pass

                action, _state = self.ctrl['agent'].predict(reply['obs'], deterministic=False)
                ep_len -= 1 #number of actions (reward)

                # Send action and get reply
                self.sock.settimeout(0.5) #prevents the client from waiting forever for a response from the server if any packets are lost

                reply, _ = self.request({'action': action})

                # Reset when done
                if reply['done']:
                    toc = time.time()
                    elapsed_time = toc-tic
                    episode_times.append(elapsed_time)
                    episode_lenghts.append(ep_len)
                    elapsed_time = "{:.6f}".format(toc-tic)
                    print("------------------------")
                    print("Episode: ", ep_num)
                    ep_num += 1
                    print("Episode Time (s): ", elapsed_time)
                    print("Episode Reward: ", ep_len)
                    ep_len  = 1
                    print("Mean time: {:.2f} +/- {:.2f}".format(np.mean(episode_times), np.std(episode_times)))
                    print("Mean lenght: {:.2f} +/- {:.2f}".format(np.mean(episode_lenghts), np.std(episode_lenghts)))
                    tic = time.time()
                    reply, _ = self.request({'reset': None})

            except Exception: #if some packet is lost and there is a timeout in the request
                if ep_num == 101: #last episode
                    self.sock.close()
                    sys.exit(0)
                    break
                ep_len -= 1
                print("Timeout Exception")

    def train(self, model, timesteps, callback=None, **kwargs):
        model.learn(total_timesteps=timesteps, callback=callback, **kwargs)
        return model

    def load_trained_agent(self):
        if self.ctrl['agent_name'] == 'auto':
            for key, _ in AGENTS.items():
                if key.upper() in self.ctrl['supported_envs'].get(self.ctrl['env_name'], []):
                    self.ctrl['agent_name'] = key
                    break

        if self.ctrl['agent_name'] not in AGENTS.keys():
            print('Unable to find a trained agent for the {} environment'.format(self.ctrl['env_name']))
            sys.exit(1)

        agent_env_path = 'rl-trained-agents/{}/{}_*/{}.zip'.format(self.ctrl['agent_name'],
                                                                   self.ctrl['env_name'],
                                                                   self.ctrl['env_name'])
        agent_path = glob.glob(agent_env_path)[0]
        return AGENTS[self.ctrl['agent_name']].load(agent_path, env=self.ctrl['env'])

    def train_an_agent(self, timesteps):
        if self.ctrl['agent_name'] == 'auto':
            for key, _ in AGENTS.items():
                if key.upper() in self.ctrl['supported_envs'].get(self.ctrl['env_name'], []):
                    self.ctrl['agent_name'] = key
                    break

        if self.ctrl['agent_name'] not in AGENTS.keys():
            print('Unable to find a trained agent for the {} environment'.format(self.ctrl['env_name']))
            sys.exit(1)

        model = AGENTS[self.ctrl['agent_name']]('MlpPolicy', self.ctrl['env'], verbose=0)
        model = self.train(model, timesteps)
        return model

    def request(self, msg):
        self.sock.send(pickle.dumps(msg))
        reply, from_addr = self.sock.recvfrom(1500)
        return pickle.loads(reply), from_addr

    def get_supported_envs(self):
        gym_envs = [e.id for e in gym.envs.registry.all()]
        trained_agents = glob.glob('rl-trained-agents/*/*/*.zip')
        supported_envs = {}
        for env_agent in trained_agents:
            env_name = os.path.basename(env_agent).replace('.zip', '')
            agent_name = os.path.basename(os.path.dirname(os.path.dirname(env_agent))).upper()
            if env_name in supported_envs:
                supported_envs[env_name].append(agent_name)
            else:
                supported_envs[env_name] = [agent_name]

        del supported_envs['best_model']

        # Get what exists on server node
        reply, _ = self.request({'get_supported_envs': None})
        intersection = set(reply['supported_envs']).intersection(set(supported_envs.keys()))

        return sorted(list(intersection)), supported_envs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Client Simple')
    parser.add_argument('-a', '--address', type=str, default='192.168.1.15', help='Server address')
    parser.add_argument('-p', '--port', type=int, default=9000, help='Server port')
    parser.add_argument('-i', '--interval', type=int, default=10, help='Max interval between actions (ms)')
    parser.add_argument('-t', '--tolerance', type=int, default=2, help='Tolerance on interval')
    parser.add_argument('-e', '--env', type=str, default='Acrobot-v1',
                        help='OpenAI Gym Environment. Use -l to see all environments available')
    parser.add_argument('-g', '--agent', type=str, default='auto',
                        help='Agent in [a2c ddpg dqn her ppo qrdqn sac td3 tqc]')
    parser.add_argument('-l', '--list-envs', action='store_true',
                        help='List supported environments (server must be running)')
    parser.add_argument('-v', '--verbose', type=int, default=0, help='Verbosity level')
    parser.add_argument('-r', '--training', type=str, default='offline', help='Choose between online and offline training')
    parser.add_argument('-s', '--timesteps', type=int, default=20000, help='Number of timesteps to train online an agent')
    args = parser.parse_args()

    Client(args)
