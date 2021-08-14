import os
import time

from rl_trainer import RL_Trainer
from rnd_agent import RNDAgent
from dqn_utils import get_env_kwargs, PiecewiseSchedule, ConstantSchedule

import logging
import json


class Q_Trainer(object):

    def __init__(self, params, train_logger):
        self.params = params

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
        }

        env_args = get_env_kwargs(params['env_name'], params['num_timesteps'])

        self.agent_params = {**train_args, **env_args, **params}

        self.params['agent_class'] = RNDAgent
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['batch_size']
        self.params['env_wrappers'] = self.agent_params['env_wrappers']
        
        self.rl_trainer = RL_Trainer(self.params, train_logger)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.agent_params['num_timesteps'],
            collect_policy = self.rl_trainer.agent.exploration_critic,#actor,
            eval_policy = self.rl_trainer.agent.exploration_critic,
            )

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_name',
        default='PointmassHard-v0',
        choices=('PointmassEasy-v0', 'PointmassMedium-v0', 'PointmassHard-v0', 'PointmassVeryHard-v0', 'PointmassMediumRandom-v0', 'PointmassHardRandom-v0')
    )
    parser.add_argument('--my_exploration',  action='store_true')
    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--use_rnd', action='store_true')
    parser.add_argument('--num_exploration_steps', type=int, default=10000)#10000)
    parser.add_argument('--unsupervised_exploration', action='store_true')

    parser.add_argument('--exploit_rew_shift', type=float, default=0.0)
    parser.add_argument('--exploit_rew_scale', type=float, default=1.0)

    parser.add_argument('--rnd_output_size', type=int, default=5)
    parser.add_argument('--rnd_n_layers', type=int, default=2)
    parser.add_argument('--rnd_size', type=int, default=400)

    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--no_gpu', '-ngpu', default=True, action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e3)) # logging freq
    parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    params['student_id'] = '2020712940'
    params['student_name'] = '김승현'

    params['double_q'] = True
    params['num_agent_train_steps_per_iter'] = 1
    params['num_critic_updates_per_agent_update'] = 1
    params['num_timesteps'] = 50000 #50000
    params['learning_starts'] = 2000
    params['eps'] = 0.2
    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if params['env_name']=='PointmassEasy-v0':
        params['ep_len']=50
    if params['env_name']=='PointmassMedium-v0':
        params['ep_len']=150
    if params['env_name']=='PointmassHard-v0' or params['env_name']=='PointmassHardRandom-v0' :
        params['ep_len']=100
    if params['env_name']=='PointmassVeryHard-v0':
        params['ep_len']=200
    if  params['env_name']=='PointmassMediumRandom-v0':
        params['ep_len']=150
        
    
    params['exploit_weight_schedule'] = ConstantSchedule(1.0)
    if params['use_rnd']:
        params['explore_weight_schedule'] = PiecewiseSchedule([(0,1), (params['num_exploration_steps'], 0)], outside_value=0.0)
    else:
        params['explore_weight_schedule'] = ConstantSchedule(0.0)

    # Exploration reward 만 사용할 경우
    if params['unsupervised_exploration']:
        params['explore_weight_schedule'] = ConstantSchedule(1.0)
        params['exploit_weight_schedule'] = ConstantSchedule(0.0)
        
        if not params['use_rnd']:
            params['learning_starts'] = params['num_exploration_steps']
    

    logdir_prefix = params['student_id'] + '_' + params['student_name'] # keep for autograder
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), './data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = logdir_prefix + args.exp_name + '_' + args.env_name[:-2] + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    training_log_path = os.path.join(logdir, './log/')
    if not (os.path.exists(training_log_path)):
        os.makedirs(training_log_path)
    
    log_file_name = params['student_id'] + '_' + params['student_name'] + '_' + 'train_log'
    train_logger = make_logger(log_file = log_file_name, log_dir = training_log_path)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")
    print("params : ", params)

    trainer = Q_Trainer(params, train_logger)
    trainer.run_training_loop()


def make_logger(log_file, log_dir, name='rnd'):
    
    logger = logging.getLogger(name)
    #log level의 가장 낮은 단계 DEBUG,  -> INFO -> WARNING ....
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_dir+log_file+".log")
    
    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    main()
