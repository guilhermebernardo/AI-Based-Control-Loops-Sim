#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import argparse
import glob
import gym
import ipaddress
import os
import pickle
import select
import socket
import sys
import time

from stable_baselines3 import A2C
from utils import AGENTS


class Server():
    def __init__(self, args):
        self.ctrl = {'interval': args['interval'] * 1e6,
                     'tolerance': args['tolerance'] * 1e6,
                     'env_name': None,
                     'env': None,
                     'agent_name': None,
                     'local_agent': None,
                     'verbose': args['verbose']}

        self.state = {'obs': None,
                      'reward': None,
                      'done': False,
                      'info': None}

        self.sock = {'ctrl': None, 'data': None, 'ctrl_conn': None}

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

        self.sock['ctrl'].bind((args['address'], args['control_port']))
        self.sock['ctrl'].listen()
        self.sock['data'].bind((args['address'], args['port']))
        self.sock['data'].setblocking(False)

        try:
            self.run()
        except KeyboardInterrupt:
            self.sock['ctrl'].close()
            self.sock['data'].close()
            sys.exit(0)

    def run(self):
        conn, from_addr = self.sock['ctrl'].accept()
        self.sock['ctrl_conn'] = conn

        while True:
            request = self.ctrl_receive()

            if 'get_supported_envs' in request:
                supported_envs = self.get_supported_envs()
                self.ctrl_reply({'supported_envs': supported_envs})
            elif 'init' in request:
                # Get and initialize environment
                self.ctrl['env_name'] = request.get('init', {}).get('env_name')
                self.ctrl['env'] = gym.make(self.ctrl['env_name'])
                self.state['obs'] = self.ctrl['env'].reset()

                # Get and initialize local agent
                self.ctrl['agent_name'] = request.get('init', {}).get('agent_name')
                self.ctrl['local_agent'] = self.load_local_agent()

                # Reply to client and proceed to control loop
                self.ctrl_reply({'init': 'OK'})
                self.sock['ctrl_conn'].setblocking(False)
                break
            else:
                print('Unknown request:', request)

        deadline = time.time_ns() + self.ctrl['interval']
        stats = {'episode_reward': 0,
                 'episode_steps': 0,
                 'deadlines_missed': 0,
                 'episode': 0}
        done_on_deadline = False
        while True:
            # First check control messages
            try:
                request = self.ctrl_receive()
                print('CTRL request:', request)
                if 'reset' in request:
                    self.state['obs'] = self.ctrl['env'].reset()
                    reply = {'obs': self.state['obs'],
                             'stats': stats}
                    self.ctrl_reply(reply)
                    stats['episode_reward'] = 0
                    stats['episode_steps'] = 0
                    stats['deadlines_missed'] = 0
                    stats['episode'] += 1
                elif 'finish' in request:
                    self.ctrl_reply({'finish': 'OK', 'stats': stats})
                    self.sock['ctrl'].close()
                    self.sock['data'].close()
                    sys.exit(0)
            except Exception:
                pass

            # Now check data messages
            try:
                request, from_addr = self.receive()

                if done_on_deadline:
                    self.state['obs'], self.state['reward'], self.state['done'], self.state['info'] = end_data
                    done_on_deadline = False
                else:
                    if 'action' in request:
                        self.state['obs'], self.state['reward'], self.state['done'], self.state['info'] = self.ctrl['env'].step(request['action'])  # noqa
                        reply = self.state
                        deadline += self.ctrl['interval']
                        stats['episode_reward'] += self.state['reward']
                        stats['episode_steps'] += 1

                print(reply)
                self.reply(reply, from_addr)

            except Exception:
                if time.time_ns() > deadline:
                    deadline = deadline + self.ctrl['interval']
                    action, _ = self.ctrl['local_agent'].predict(self.state['obs'], deterministic=False)
                    self.state['obs'], self.state['reward'], self.state['done'], self.state['info'] = self.ctrl['env'].step(action)  # noqa
                    print('.', end='')
                    sys.stdout.flush()
                    stats['episode_reward'] += self.state['reward']
                    stats['episode_steps'] += 1
                    stats['deadlines_missed'] += 1

                    if self.state['done']:
                        end_data = self.ctrl['env'].reset()
                        done_on_deadline = True

            self.ctrl['env'].render(mode='human')

    def load_local_agent(self):
        return AGENTS[self.ctrl['agent_name']]('MlpPolicy', self.ctrl['env'], verbose=0)

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
        request, from_addr = self.sock['data'].recvfrom(1500)
        return pickle.loads(request), from_addr

    def ctrl_receive(self):
        request = self.sock['ctrl_conn'].recv(1500)
        return pickle.loads(request)

    def reply(self, msg, addr):
        self.sock['data'].sendto(pickle.dumps(msg), addr)

    def ctrl_reply(self, msg):
        self.sock['ctrl_conn'].send(pickle.dumps(msg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Client Simple')
    parser.add_argument('-a', '--address', type=str, default='0.0.0.0', help='Server address')
    parser.add_argument('-p', '--port', type=int, default=9000, help='Server port')
    parser.add_argument('-c', '--control-port', type=int, default=9001, help='Control port')
    parser.add_argument('-i', '--interval', type=int, default=10, help='Max interval between actions (ms)')
    parser.add_argument('-t', '--tolerance', type=int, default=2, help='Tolerance on interval')
    parser.add_argument('-v', '--verbose', type=int, default=0, help='Verbosity level')
    args = vars(parser.parse_args())

    Server(args)
