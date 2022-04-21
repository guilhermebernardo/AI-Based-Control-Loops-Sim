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

#from stable_baselines3 import A2C
from utils import AGENTS


class Server():
    def __init__(self, args):
        self.ctrl = {'interval': args.interval * 1e6,
                     'tolerance': args.tolerance * 1e6,
                     'env_name': None,
                     'env': None,
                     'agent_name': None,
                     'agent': None,
                     'verbose': args.verbose,
                     'training': args.training,
                     'timesteps': args.timesteps}

        self.state = {'obs': None,
                      'reward': None,
                      'done': False,
                      'info': None}

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((args.address, args.port))

        try:
            self.run()
        except KeyboardInterrupt:
            sys.exit(0)

    def run(self):
        while True:
            request, from_addr = self.receive()
            if 'init' in request:
                # Get and initialize environment
                self.ctrl['env_name'] = request.get('init', {}).get('env_name')
                self.ctrl['env'] = gym.make(self.ctrl['env_name'])
                self.state['obs'] = self.ctrl['env'].reset()

                # Get and initialize local agent
                self.ctrl['agent_name'] = request.get('init', {}).get('agent_name')
                if self.ctrl['training'] == 'offline':
                    self.ctrl['agent'] = self.load_local_agent()
                elif self.ctrl['training'] == 'online':
                    self.ctrl['agent'] = self.train_an_agent(self.ctrl['timesteps'])
                else:
                    print('Training type not supported')
                    sys.exit(1)

                # Reply to client and proceed to control loop
                self.reply({'init': 'OK'}, from_addr)
                break
            elif 'get_supported_envs' in request:
                supported_envs = self.get_supported_envs()
                self.reply({'supported_envs': supported_envs}, from_addr)
            else:
                print('Initialization failed')
                sys.exit(0)
            time.sleep(0.01)

        # After initialization, disable blocking
        self.sock.setblocking(False)
        
        episode_lengths = []
        ep_len = 0
        finish_game = 0
        first_time = True #begins with reset and reward = 0

        deadline = time.time_ns() + self.ctrl['interval']
        self.sock.settimeout(2)
        while True:
            try:
                request, from_addr = self.receive()

                if finish_game == 100: #after 100 episode executions
                    if len(episode_lengths) > 0:
                        print("Mean length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))
                    self.sock.close()
                    sys.exit(0)
                    break

                if 'reset' in request:
                    self.state['obs'] = self.ctrl['env'].reset()
                    if not first_time:
                        episode_lengths.append(ep_len) #after 1 episode execution, store the rewards
                        print("Episode Lenght:", ep_len)
                        finish_game += 1 #variable to stop the game execution after 100 rewards
                        ep_len = 0
                    first_time = False
                    #time.sleep(1)
                    reply = {'obs': self.state['obs']}
                elif 'action' in request:
                    self.state['obs'], self.state['reward'], self.state['done'], self.state['info'] = self.ctrl['env'].step(request['action']) # noqa
                    ep_len += self.state['reward'] #the reward is based on the number of actions performed. the greater the number of actions, the lower the reward (10 actions = -10 reward points)
                    reply = self.state
                elif 'finish' in request:
                    self.sock.close()
                    sys.exit(0)
                    break

                self.reply(reply, from_addr)
                deadline = time.time_ns() + self.ctrl['interval']

            except Exception:
                if time.time_ns() < deadline:
                    pass
                else:
                    deadline = deadline + self.ctrl['interval']
                    # Take random action provided by our local untrained model
                    action, _ = self.ctrl['agent'].predict(self.state['obs'], deterministic=False)
                    self.state['obs'], self.state['reward'], self.state['done'], self.state['info'] = self.ctrl['env'].step(action)  # noqa
                    ep_len += self.state['reward'] 
                    print('.', end='')
                    sys.stdout.flush()

            self.ctrl['env'].render(mode='human')

    def train(self, model, timesteps, callback=None, **kwargs):
        model.learn(total_timesteps=timesteps, callback=callback, **kwargs)
        return model

    def load_local_agent(self):
        m = AGENTS[self.ctrl['agent_name']]('MlpPolicy', self.ctrl['env'], verbose=0)
        return m

    def train_an_agent(self, timesteps):
        m = AGENTS[self.ctrl['agent_name']]('MlpPolicy', self.ctrl['env'], verbose=0)
        print('Training...')
        m = self.train(m, timesteps)
        return m

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
        return sorted(supported_envs.keys())

    def receive(self):
        request, from_addr = self.sock.recvfrom(1500)
        return pickle.loads(request), from_addr

    def reply(self, msg, addr):
        self.sock.sendto(pickle.dumps(msg), addr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Client Simple')
    parser.add_argument('-a', '--address', type=str, default='0.0.0.0', help='Server address')
    parser.add_argument('-p', '--port', type=int, default=9000, help='Server port')
    parser.add_argument('-i', '--interval', type=int, default=10, help='Max interval between actions (ms)')
    parser.add_argument('-t', '--tolerance', type=int, default=2, help='Tolerance on interval')
    parser.add_argument('-v', '--verbose', type=int, default=0, help='Verbosity level')
    parser.add_argument('-r', '--training', type=str, default='offline', help='Choose between online and offline training')
    parser.add_argument('-s', '--timesteps', type=int, default=20000, help='Number of timesteps to traing an agent')
    args = parser.parse_args()

    Server(args)
