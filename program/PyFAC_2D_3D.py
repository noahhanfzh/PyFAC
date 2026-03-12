from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from PyFAC_Env import Env

from Plot_Output import PlotOutput

from Define_Model import DefineModel

from Callback import ModelBufferSaveCallback

class PyFAC:
    def __init__(
            self,
            action_mode,
            working_library,
            processor_number,
            project_name,
            frequency_range,
            amplitude_range,
            reward_type,
            retrieve_training,
            policy_type,
            learning_rate,
            solver_dimension,
            parallel,
            use_fst,
            use_lstm,
            device
    ):
        self.device = device

        self.action_mode = action_mode
        self.working_library = working_library
        self.processor_number = processor_number
        self.project_name = project_name
        self.frequency_range = frequency_range
        self.amplitude_range = amplitude_range
        self.reward_type = reward_type
        self.learning_rate = learning_rate

        self.retrieve_training = retrieve_training
        self.policy_type = policy_type

        self.buffer_size = int(1e5)
        self.learning_start = 1000
        self.batch_size = 128
        self.gradient_steps = 1

        self.tau = 0.005
        self.gamma = 0.99

        self.solver_dimension = solver_dimension
        self.parallel = parallel

        self.use_fst = use_fst
        self.use_lstm = use_lstm

    def make_env(self, rank):
        def _init():
            return Env(
                action_mode = self.action_mode,
                working_library = self.working_library,
                processor_number = self.processor_number,
                solver_dimension = self.solver_dimension,
                project_name = self.project_name,
                frequency_range = self.frequency_range,
                amplitude_range = self.amplitude_range,
                reward_type = self.reward_type,
                env_num = rank,
                use_fst = self.use_fst,
                use_lstm = self.use_lstm,
                device = self.device
            )
        return _init

    def model_learn(self):
        envs = [self.make_env(i+1) for i in range(self.parallel)]
        vec_env = SubprocVecEnv(envs)

        model, elapsed_num_steps = DefineModel(
            device_=self.device,
            retrieve_training_=self.retrieve_training,
            policy_type_=self.policy_type,
            working_library_=self.working_library,
            vec_env_=vec_env,
            learning_rate_=self.learning_rate,
            buffer_size_=self.buffer_size,
            learning_start_=self.learning_start,
            batch_size_=self.batch_size,
            gradient_steps_=self.gradient_steps,
            tau_=self.tau,
            gamma_=self.gamma
        ).define_model()

        print(f'elapsed_num_steps = {elapsed_num_steps}')

        plotter = PlotOutput(working_library_=self.working_library)
        callback = ModelBufferSaveCallback(
            save_freq=200,
            save_path=self.working_library + '\\model',
            num_steps=elapsed_num_steps,
            plotter=plotter
        )

        model.learn(total_timesteps=100_000, callback=callback)





