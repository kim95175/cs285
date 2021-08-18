import numpy as np

from dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from dqn import DQN


class DQNAgent(object):
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

        self.dqn = DQN(agent_params, self.optimizer_spec)
        #self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander, n_step=agent_params['n_step'])
        self.t = 0
        self.num_param_updates = 0
        self.num_episodes = 0

        self.num_grounded = 0
        self.num_at_site = 0

        self.n_step = agent_params['n_step']
        self.gamma = agent_params['gamma']

    #def add_to_replay_buffer(self, paths):
    #    pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        

        # TODO store the latest observation ("frame") into the replay buffer
        # HINT: the replay buffer used here is `MemoryOptimizedReplayBuffer`
            # in dqn_utils.py
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        
        #print(self.replay_buffer_idx)
        eps = self.exploration.value(self.t)

        # TODO use epsilon greedy exploration when selecting action
        perform_random_action = np.random.random() < eps or self.t < self.learning_starts
        if perform_random_action:
            # HINT: take random action 
                # with probability eps (see np.random.random())
                # OR if your current step number (see self.t) is less that self.learning_starts
            action = np.random.randint(self.num_actions)
        else:
            # HINT: Your actor will take in multiple previous observations ("frames") in order
                # to deal with the partial observability of the environment. Get the most recent 
                # `frame_history_len` observations using functionality from the replay buffer,
                # and then use those observations as input to your actor. 
            enc_last_obs = self.replay_buffer.encode_recent_observation()
            enc_last_obs = enc_last_obs[None, :] 

            # TODO query the policy with enc_last_obs to select action
            action = self.dqn.get_action(enc_last_obs.astype(np.float32))
            action = action[0]
        
        # TODO take a step in the environment using the action from the policy
        # HINT1: remember that self.last_obs must always point to the newest/latest observation
        # HINT2: remember the following useful function that you've seen before:
            #obs, reward, done, info = env.step(action)
        obs, reward, done, info = self.env.step(action)
        self.last_obs=obs

        # TODO store the result of taking this action into the replay buffer
        # HINT1: see your replay buffer's `store_effect` function
        # HINT2: one of the arguments you'll need to pass in is self.replay_buffer_idx from above
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # TODO if taking this step resulted in done, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()
            self.num_episodes += 1
            if info['at_site']:
                self.num_at_site += 1
            if info['grounded']:
                self.num_grounded += 1
            #print(info)

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, obs, action, reward, next_obs, done):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            #print(f"obs {obs.shape}, action {action.shape}, reward {reward.shape}, next_obs {next_obs.shape}, done {done.shape}")
            # TODO fill in the call to the update function using the appropriate tensors
            #print("agent/train ", obs.shape, next_obs.shape)
            if self.n_step > 1:
                obs, action, reward, next_obs, done = self.calc_n_step_return(obs, action, reward, next_obs, done)
            
            log = self.dqn.update(
                obs, action, next_obs, reward, done
            )
            # TODO update the target network periodically 
            # HINT: your critic already has this functionality implemented
            if self.num_param_updates % self.target_update_freq == 0:
                self.dqn.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
    

    def calc_n_step_return(self, obs_n, action_n, reward_n, next_obs_n, done_n):
        ##is_done = np.ones(done[0].shape)
        #print("is_done", is_done)
        done_step = np.zeros(done_n[0].shape)

        next_obs_stack = np.stack(next_obs_n, axis=1)
        done_stack = np.stack(done_n, axis=1)
        #print(next_obs.shape)

        obs = obs_n[0]
        action = action_n[0]
        n_step_reward = reward_n[0]
        is_done = (1 - done_n[0])
    
        #print("0", n_step_reward)
        for i in range(1, self.n_step):
            n_step_reward += (self.gamma**i) * ( reward_n[i] * (is_done))
            #print(i, n_step_reward)
            done_step += is_done
            is_done *= (1 - done_n[i])
            #print(done[i])
            #print("is_done", is_done)
        
        #print("done step", done_step)
        
        next_obs = np.zeros(obs.shape)
        done = np.zeros(done_n[0].shape)
        for idx, step in enumerate(done_step):
            next_obs[idx] = next_obs_stack[idx][int(step)]
            done[idx] = done_stack[idx][int(step)]
        '''
        if 0 in done_step:
            #print(obs_n)
            print("================N-step_return=========")
            print(next_obs_n)
            print(reward_n)
            print(done_n)

            print(next_obs)
            print(n_step_reward)
            print(done)

            print(is_done)
            print(done_step)
        #print(done)
        '''
        
        return obs, action, n_step_reward, next_obs, done
