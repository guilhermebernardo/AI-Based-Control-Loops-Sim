#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import argparse
import glob
import gym
import ipaddress
import os
import pickle
import socket
import sys
import time
from stable_baselines3 import A2C
from stable_baselines3.common.utils import set_random_seed
from utils import AGENTS


class Client():
    def __init__(self, args):
        self.ctrl = {'verbose': args['verbose'],
                     'interval': args['interval'] * 1e6,
                     'tolerance': args['tolerance'] * 1e6,
                     'supported_envs': [],
                     'env_name': args['env'],
                     'env': None,
                     'agent_name': args['agent'],
                     'agent': None}

        self.sock = {'ctrl': None, 'data': None}

        self.sock = {'ctrl': socket.socket(socket.AF_INET, socket.SOCK_STREAM),
                     'data': socket.socket(socket.AF_INET, socket.SOCK_DGRAM)}

        try:
            if isinstance(ipaddress.ip_address(args['address']), ipaddress.IPv4Address):
                self.sock['ctrl'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock['data'] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            elif isinstance(ipaddress.ip_address(args['address']), ipaddress.IPv6Address):
                self.sock['ctrl'] = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
                self.sock['data'] = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        except ValueError:
            print('Error: invalid IP address {}'.format(args['address']))
            sys.exit(1)

        self.sock['ctrl'].connect((args['address'], args['control_port']))
        self.sock['data'].connect((args['address'], args['port']))

        env_list, env_detail = self.get_supported_envs()
        self.ctrl['supported_envs'] = env_detail

        if args['list_envs']:
            print(env_list)
            sys.exit(0)

        if self.ctrl['env_name'] not in self.ctrl['supported_envs']:
            print('Requested environment not supported by both endpoints')
            sys.exit(1)

        self.ctrl['env'] = gym.make(self.ctrl['env_name'])
        self.ctrl['agent'] = self.load_trained_agent()

        try:
            self.run()
        except KeyboardInterrupt:
            result = self.ctrl_request({'finish': None})
            print(result)
            self.sock['ctrl'].close()
            self.sock['data'].close()
            sys.exit(0)

    def run(self):
        reply = self.ctrl_request({'init': {'env_name': self.ctrl['env_name'],
                                   'agent_name': self.ctrl['agent_name']}})

        if reply.get('init', '') != 'OK':
            print('Failed to initialize remote environment')
            sys.exit(0)

        reply = self.ctrl_request({'reset': None})
        while True:
            # Busy waiting, taking tolerance out of deadline
            deadline = time.time_ns() + self.ctrl['interval']
            while deadline - self.ctrl['tolerance'] > time.time_ns():
                pass

            action, _state = self.ctrl['agent'].predict(reply['obs'], deterministic=False)

            # Send action and get reply
            reply, _ = self.request({'action': action})
            print(action, '/', reply)

            # Reset when done
            if reply['done']:
                reply = self.ctrl_request({'reset': None})

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

    def request(self, msg):
        self.sock['data'].send(pickle.dumps(msg))
        reply, from_addr = self.sock['data'].recvfrom(1500)
        return pickle.loads(reply), from_addr

    def ctrl_request(self, msg):
        self.sock['ctrl'].send(pickle.dumps(msg))
        reply = self.sock['ctrl'].recv(1500)
        return pickle.loads(reply)

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
        reply = self.ctrl_request({'get_supported_envs': None})
        print(reply)
        intersection = set(reply['supported_envs']).intersection(set(supported_envs.keys()))

        return sorted(list(intersection)), supported_envs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Client Simple')
    parser.add_argument('-a', '--address', type=str, default='192.168.122.74', help='Server address')
    parser.add_argument('-p', '--port', type=int, default=9000, help='Server port')
    parser.add_argument('-c', '--control-port', type=int, default=9001, help='Control port')
    parser.add_argument('-i', '--interval', type=int, default=10, help='Max interval between actions')
    parser.add_argument('-t', '--tolerance', type=int, default=2, help='Tolerance on interval')
    parser.add_argument('-e', '--env', type=str, default='CartPole-v1',
                        help='OpenAI Gym Environment. Use -l to see all environments available')
    parser.add_argument('-g', '--agent', type=str, default='auto',
                        help='Agent in [a2c ddpg dqn her ppo qrdqn sac td3 tqc]')
    parser.add_argument('-l', '--list-envs', action='store_true',
                        help='List supported environments (server must be running)')
    parser.add_argument('-v', '--verbose', type=int, default=0, help='Verbosity level')
    args = vars(parser.parse_args())

    Client(args)
