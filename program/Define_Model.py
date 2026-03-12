from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

import torch

class DefineModel:
    def __init__(
            self,
            device_,
            retrieve_training_,
            policy_type_,
            working_library_,
            vec_env_,
            learning_rate_,
            buffer_size_,
            learning_start_,
            batch_size_,
            gradient_steps_,
            tau_=0.005,
            gamma_=0.99
    ):
        self.device = device_

        self.retrieve_training = retrieve_training_
        self.policy_type = policy_type_
        self.working_library = working_library_
        self.vec_env = vec_env_
        self.elapsed_num_steps = 0

        self.learning_rate = learning_rate_
        self.buffer_size = buffer_size_
        self.learning_start = learning_start_
        self.batch_size = batch_size_
        self.gradient_steps = gradient_steps_

        self.tau = tau_
        self.gamma = gamma_

        self.model = None

    def define_model(self):
        if self.policy_type == 'SAC':
            self.sac()
            return self.model, self.elapsed_num_steps
        elif self.policy_type == 'TD3':
            self.td3()
            return self.model, self.elapsed_num_steps
        elif self.policy_type == 'PPO':
            self.ppo()
            return self.model, self.elapsed_num_steps

    def td3(self):
        if self.retrieve_training:
            self.model = TD3.load(self.working_library + '\\model\\final_model', env=self.vec_env)
            self.model.load_replay_buffer(self.working_library + '\\model\\final_buffer')

            self.elapsed_num_steps = int(self.model.replay_buffer.size())

            print(f'elapsed_num_steps = {self.elapsed_num_steps}')

        else:
            self.model = TD3(
                policy='MlpPolicy',
                env=self.vec_env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                learning_starts=self.learning_start,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                gradient_steps=self.gradient_steps,
                verbose=1
            )

    def sac(self):
        if self.retrieve_training:
            self.model = SAC.load(self.working_library + '\\model\\final_model', env=self.vec_env, device=self.device)
            self.model.load_replay_buffer(self.working_library + '\\model\\final_buffer')

            self.elapsed_num_steps = int(self.model.replay_buffer.size())

            print(f'elapsed_num_steps = {self.elapsed_num_steps}')

        else:
            self.model = SAC(
                policy='MlpPolicy',
                env=self.vec_env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                learning_starts=self.learning_start,
                batch_size=self.batch_size,
                tau=self.tau,
                gamma=self.gamma,
                gradient_steps=self.gradient_steps,
                verbose=1,
                device=self.device,
                policy_kwargs=dict(hidden_sizes=(512, 512))
            )

    def ppo(self):
        if self.retrieve_training:
            self.model = PPO.load(self.working_library + '\\model\\final_model', env=self.vec_env)

        else:
            policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[512, 512], vf=[512, 512]))
            self.model = PPO(
                policy='MlpPolicy',
                env=self.vec_env,
                learning_rate=self.learning_rate,
                n_steps=64,
                batch_size=32,
                gamma=self.gamma,
                verbose=1,
                policy_kwargs=policy_kwargs
            )