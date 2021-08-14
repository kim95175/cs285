from collections import OrderedDict

from dqn_critic import DQNCritic
from replay_buffer import ReplayBuffer
from utils import *
from dqn_utils import MemoryOptimizedReplayBuffer
from rnd_model import RNDModel, MyExplorationModel
import numpy as np
import pdb


class RNDAgent(object):
    def __init__(self, env, agent_params):
        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        # import ipdb; ipdb.set_trace()
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.t = 0
        self.num_param_updates = 0

        self.replay_buffer = MemoryOptimizedReplayBuffer(100000, 1, float_obs=True)
        self.num_exploration_steps = agent_params['num_exploration_steps']
        
        #self.exploitation_critic = DQNCritic(agent_params, self.optimizer_spec) # estimates policy return under actual reward
        self.exploration_critic = DQNCritic(agent_params, self.optimizer_spec) # estimates policy return under reward with exploration bonus

        if agent_params['my_exploration']:
            self.exploration_model = MyExplorationModel(agent_params, self.batch_size)
            print('using my exploration model')
        else:
            self.exploration_model = RNDModel(agent_params, self.optimizer_spec)
            print('using RND model')

        self.explore_weight_schedule = agent_params['explore_weight_schedule']
        self.exploit_weight_schedule = agent_params['exploit_weight_schedule']
        #print("explore {}, exploit {}".format(self.explore_weight_schedule._v, self.exploit_weight_schedule._v))
        
        self.exploit_rew_shift = agent_params['exploit_rew_shift']
        self.exploit_rew_scale = agent_params['exploit_rew_scale']
        self.eps = agent_params['eps']

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        '''
        if self.t > self.num_exploration_steps:
            # TODO: After exploration is over, set the actor to optimize the extrinsic critic
            #HINT: Look at method ArgMaxPolicy.set_critic
            #self.actor.set_critic(self.exploitation_critic)
            #self.exploration_critic.set_critic(self.exploration_critic)
            pass
        '''

        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):
            # Get Reward Weights
            # TODO: Get the current explore reward weight and exploit reward weight
            #       using the schedule's passed in (see __init__)
            # COMMENT: Until part 3, explore_weight = 1, and exploit_weight = 0
            explore_weight = self.explore_weight_schedule.value(self.t)
            exploit_weight = self.exploit_weight_schedule.value(self.t)

            # Run Exploration Model #
            # TODO: Evaluate the exploration model on s' to get the exploration bonus
            # HINT: Normalize the exploration bonus, as RND values vary highly in magnitude
            error = self.exploration_model.forward_np(next_ob_no)
            expl_bonus = normalize(error, error.mean(), error.std())

            # Reward Calculations #
            # TODO: Calculate mixed rewards, which will be passed into the exploration critic
            # HINT: See hw5 pdf for definition of mixed_reward
            mixed_reward = explore_weight*expl_bonus + exploit_weight*re_n

            # TODO: Calculate the environment reward
            # HINT: For part 1, env_reward is just 're_n'
            #       After this, env_reward is 're_n' shifted by self.exploit_rew_shift,
            #       and scaled by self.exploit_rew_scale
            env_reward = (re_n + self.exploit_rew_shift)*self.exploit_rew_scale

            # Update Critics And Exploration Model #

            # TODO 1): Update the exploration model (based off s')
            # TODO 2): Update the exploration critic (based off mixed_reward)
            # TODO 3): Update the exploitation critic (based off env_reward)
            expl_model_loss = self.exploration_model.update(next_ob_no)
            exploration_critic_loss = self.exploration_critic.update(ob_no, ac_na, next_ob_no, mixed_reward, terminal_n)
            #exploitation_critic_loss = self.exploitation_critic.update(ob_no, ac_na, next_ob_no, env_reward, terminal_n)

            # Target Networks #
            if self.num_param_updates % self.target_update_freq == 0:
                # TODO: Update the exploitation and exploration target networks
                self.exploration_critic.update_target_network()
            # Logging #
            log['Exploration Critic Loss'] = exploration_critic_loss['Training Loss']
            log['Exploration Model Loss'] = expl_model_loss
            log['Exploration weight'] = explore_weight


            self.num_param_updates += 1

        self.t += 1
        return log


    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

        perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            processed = self.replay_buffer.encode_recent_observation()
            action = self.exploration_critic.get_action(processed)
        
        next_obs, reward, done, info = self.env.step(action)
        #if reward != -1:
        #    print("next_obs {}, reward {}, done {}".format(next_obs, reward, done))
        self.last_obs = next_obs.copy()

        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        if done:
            self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]