#python run_dqn.py --env_name LunarLander-v3 --exp_name dqn_n1_s1 --seed 1 
#python run_dqn.py --env_name LunarLander-v3 --exp_name dqn_n1_s2 --seed 2 
#python run_dqn.py --env_name LunarLander-v3 --exp_name dqn_n1_s3 --seed 3 

#python run_dqn.py --env_name LunarLander-v3 --exp_name ddqn_n1_s1 --seed 1 --double_q
#python run_dqn.py --env_name LunarLander-v3 --exp_name ddqn_n1_s2 --seed 2 --double_q
#python run_dqn.py --env_name LunarLander-v3 --exp_name ddqn_n1_s3 --seed 3 --double_q

#python run_dqn.py --env_name LunarLander-v3 --exp_name dqn_n3_s1 --seed 1 --n_step 1 
#python run_dqn.py --env_name LunarLander-v3 --exp_name dqn_n3_s2 --seed 2 --n_step 2 
#python run_dqn.py --env_name LunarLander-v3 --exp_name dqn_n3_s3 --seed 3 --n_step 3 


python run_dqn.py --env_name LunarLander-v3 --exp_name dqn_n1_s5 --seed 5 --num_timesteps 500000 --batch_size 32 --double_q
#python run_dqn.py --env_name LunarLander-v3 --exp_name ddqn_n1_s4 --seed 4 --double_q
#python run_dqn.py --env_name LunarLander-v3 --exp_name dqn_n_step3_s4 --seed 4 --n_step 4
