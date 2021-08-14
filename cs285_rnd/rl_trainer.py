from collections import OrderedDict
import pickle
import os
import sys
import time
import pdb
import os

import gym
from gym import wrappers
import numpy as np
import torch
import pytorch_util as ptu
import utils
from rnd_agent import RNDAgent
from dqn_utils import (
        get_wrapper_by_name,
        register_custom_envs,
)
from dqn_utils import PiecewiseSchedule, ConstantSchedule
#register all of our envs
import pointmass

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below
device = None

class RL_Trainer(object):
    def __init__(self, params, train_logger):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        print(self.params)
        self.train_logger = train_logger
        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )
        self.print_params()
        #############
        ## ENV
        #############
        
        # Make the gym environment
        register_custom_envs()
        self.env = gym.make(self.params['env_name'])
        start_state = self.env._sample_empty_state()
        goal_state = self.env.fixed_goal
        self.train_logger.info('Start state: {} Goal state: {}'.format(start_state, goal_state))

        self.eval_env = gym.make(self.params['env_name'])
        if not ('pointmass' in self.params['env_name']):
            import matplotlib
            matplotlib.use('Agg')
            self.env.set_logdir(self.params['logdir'] + '/expl_')
            self.eval_env.set_logdir(self.params['logdir'] + '/eval_')
            
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.Monitor(self.env, os.path.join(self.params['logdir'], "gym"), force=True)
            self.eval_env = wrappers.Monitor(self.eval_env, os.path.join(self.params['logdir'], "gym"), force=True)
            #self.env = params['env_wrappers'](self.env)
            #self.eval_env = params['env_wrappers'](self.eval_env)
            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')

        self.env.seed(seed)
        self.eval_env.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        
        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        
        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1000 if isinstance(self.agent, RNDAgent) else 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                self.train_logger.info("********** Iteration %i ************"%itr)
                
            
            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            self.agent.step_env()
            envsteps_this_batch = 1
            train_video_paths = None
            paths = None
            

            #if (not self.agent.offline_exploitation) or (self.agent.t <= self.agent.num_exploration_steps):
            self.total_envsteps += envsteps_this_batch

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                self.train_logger.info("Training agent...")
            all_logs = self.train_agent()

            # Log densities and output trajectories
            if itr % print_period == 0:
                self.dump_density_graphs(itr)

            # log/save
            if self.logmetrics:
                # perform logging
                self.train_logger.info('Beginning logging procedure...')
                if isinstance(self.agent, RNDAgent):
                    self.perform_dqn_logging(all_logs)

                if self.params['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logdir'], itr))


    def train_agent(self):
        # TODO: get this from Piazza
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['train_batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs


    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        self.train_logger.info("Timestep %d" % (self.agent.t,))

        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        self.train_logger.info("mean reward (100 episodes) %f" % self.mean_episode_reward)

        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        self.train_logger.info("best mean reward %f" % self.best_mean_episode_reward)
        
        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            self.train_logger.info("running time %f" % time_since_start)
            #logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)
        
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.eval_env, self.agent.exploration_critic, self.params['eval_batch_size'], self.params['ep_len'])
        
        eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
        eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        logs["Eval_MaxReturn"] = np.max(eval_returns)
        logs["Eval_MinReturn"] = np.min(eval_returns)
        logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)
        
        logs['Buffer size'] = self.agent.replay_buffer.num_in_buffer

        sys.stdout.flush()

        for key, value in logs.items():
            self.train_logger.info('{} : {}'.format(key, value))
        self.train_logger.info('Done logging...\n\n')

   
    def dump_density_graphs(self, itr):
        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        filepath = lambda name: self.params['logdir']+'/curr_{}.png'.format(name)

        num_states = self.agent.replay_buffer.num_in_buffer - 2
        states = self.agent.replay_buffer.obs[:num_states]
        if num_states <= 0: return
        
        H, xedges, yedges = np.histogram2d(states[:,0], states[:,1], range=[[0., 1.], [0., 1.]], density=True)
        plt.imshow(np.rot90(H), interpolation='bicubic')
        plt.colorbar()
        plt.title('State Density')
        self.fig.savefig(filepath('state_density'), bbox_inches='tight')

        '''
        plt.clf()
        ii, jj = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
        obs = np.stack([ii.flatten(), jj.flatten()], axis=1)
        density = self.agent.exploration_model.forward_np(obs)
        density = density.reshape(ii.shape)
        plt.imshow(density[::-1])
        plt.colorbar()
        plt.title('RND Value')
        self.fig.savefig(filepath('rnd_value'), bbox_inches='tight')
        
        plt.clf()
        exploration_values = self.agent.exploration_critic.qa_values(obs).mean(-1)
        exploration_values = exploration_values.reshape(ii.shape)
        plt.imshow(exploration_values[::-1])
        plt.colorbar()
        plt.title('Predicted Exploration Value')
        self.fig.savefig(filepath('exploration_value'), bbox_inches='tight')
        '''
    
    def print_params(self):
        self.train_logger.info("student_name : {}\tstudent_id : {}".format(self.params['student_name'], self.params['student_id']))
        self.train_logger.info("env_name : {}\tuse_rnd : {}\t my_exploration : {}".format(self.params['env_name'], self.params['use_rnd'], self.params['my_exploration']))
        self.train_logger.info("num_timesteps : {}\tepisode_max_len : {}".format(self.params['num_timesteps'], self.params['ep_len']))
        exploit_weight = 1
        if self.params['use_rnd']:
            explore_weight = 'schedule'
        else:
            explore_weight = 0
        
        if self.params['unsupervised_exploration']:
            exploit_weight, explore_weight = 0, 1
            
        self.train_logger.info("exploit_weight : {}\texplore_weight : {}".format(exploit_weight, explore_weight))
        