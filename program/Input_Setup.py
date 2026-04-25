import datetime
import os
import re

class InputSetup:
    def __init__(self, library_dir):
        self.library_dir = library_dir

        self.processor_number = 1
        self.parallel = 1
        self.time_stamp = 1
        self.model_type = 'RAE2822'
        self.policy_type = 'SAC'
        self.action_mode = 'frequency'
        self.reward_type = 'lift'
        self.learning_rate = 1e-4
        self.solver_dimension = '2D'
        self.retrieve_training = False
        self.use_fst = False
        self.use_lstm = False

    def input_setup(self):
        self.processor_number = input('Input PROCESSOR NUMBER (integer): ')
        self.processor_number = int(self.processor_number)
        print('PROCESSOR NUMBER = ' + str(self.processor_number))
        print('\n')

        self.parallel = input('Input PARALLEL NUMBER (integer): ')
        self.parallel = int(self.parallel)
        print('PARALLEL NUMBER = ' + str(self.parallel))
        print('\n')

        while True:
            self.retrieve_training = input('Retrieve Previous Training?(Y/N)')

            if self.retrieve_training == 'Y' or self.retrieve_training == 'y':
                self.retrieve_training = True
                print('Retrieve Previous Training')
                print('\n')

                self.retrieve_training_configuration()

                break

            elif self.retrieve_training == 'N' or self.retrieve_training == 'n':
                self.retrieve_training = False
                print('Do not Retrieve Previous Training')
                print('\n')

                self.training_configuration()

                break
            else:
                print('Check Input')
                print('\n')
                continue

        while True:
            self.use_fst = input('Apply FST?(Y/N)')

            if self.use_fst == 'Y' or self.use_fst == 'y':
                self.use_fst = True
                print('Apply FST')
                print('\n')
                break

            elif self.use_fst == 'N' or self.use_fst == 'n':
                self.use_fst = False
                print('Do not Apply FST')
                print('\n')
                break

            else:
                print('Check Input')
                print('\n')
                continue

        while True:
            self.use_lstm = input('Apply FST?(Y/N)')

            if self.use_lstm == 'Y' or self.use_lstm == 'y':
                self.use_lstm = True
                print('Apply LSTM')
                print('\n')
                break

            elif self.use_lstm == 'N' or self.use_lstm == 'n':
                self.use_lstm = False
                print('Do not Apply LSTM')
                print('\n')
                break

            else:
                print('Check Input')
                print('\n')
                continue

        return (self.processor_number, self.parallel, self.time_stamp, self.model_type, self.solver_dimension,
                self.policy_type, self.action_mode, self.reward_type, self.learning_rate, self.retrieve_training,
                self.use_fst, self.use_lstm)


    def training_configuration(self):

        curr_time_ = datetime.datetime.now()
        system_time_ = datetime.datetime.strftime(curr_time_, '%Y-%m-%d %H:%M:%S')
        print("SYSTEM TIME: " + str(system_time_))
        self.time_stamp = datetime.datetime.strftime(curr_time_, '%Y%m%d%H%M%S')
        print("TIME STAMP: " + str(self.time_stamp))
        print('\n')

        while True:
            print('POLICY TYPE:')
            print('A2C, DDPG, PPO, SAC, TD3')
            self.policy_type = input('Choose Policy Type: ')

            if self.policy_type in ('A2C', 'DDPG', 'PPO', 'SAC', 'TD3'):
                print('Policy: ' + self.policy_type)
                print('\n')
                break
            else:
                print('Check Input.')
                print('\n')
                continue

        while True:
            print('MODEL TYPE:')
            print('1.RAE2822, 2.NACA0012')
            self.model_type = input('Choose number: ')

            if self.model_type == '1':
                self.model_type = 'RAE2822'
                print(f'SOLVER DIMENSION: {self.model_type}')
                print('\n')
                break
            elif self.model_type == '2':
                self.model_type = 'NACA0012'
                print(f'SOLVER DIMENSION: {self.model_type}')
                print('\n')
                break
            else:
                print('INVALID MODEL TYPE ! ! ')
                print('\n')
                continue

        while True:
            print('SOLVER DIMENSION:')
            print('2.2D, 3.3D')
            self.solver_dimension = input('Choose number: ')

            if self.solver_dimension == '2':
                self.solver_dimension = 2
                print(f'SOLVER DIMENSION: {self.solver_dimension}D')
                print('\n')
                break
            elif self.solver_dimension == '3':
                self.solver_dimension = 3
                print(f'SOLVER DIMENSION: {self.solver_dimension}D')
                print('\n')
                break
            else:
                print('INVALID SOLVER DIMENSION ! ! ')
                print('\n')
                continue

        while True:  # 1.frequency     2.amplitude     3.amplitude_frequency
            print('ACTION MODE: 1.frequency     2.amplitude     3.amplitude_frequency')
            self.action_mode = input('Choose the NUMBER: ')
            if self.action_mode == '1':
                self.action_mode = 'frequency'
                print('ACTION MODE = ' + self.action_mode)
                print('\n')
                break
            elif self.action_mode == '2':
                self.action_mode = 'amplitude'
                print('ACTION MODE = ' + self.action_mode)
                print('\n')
                break
            elif self.action_mode == '3':
                self.action_mode = 'amplitude_frequency'
                print('ACTION MODE = ' + self.action_mode)
                print('\n')
                break
            else:
                print('ACTION MODE FALSE ! ! ')
                print('\n')
                continue

        while True:
            print('REWARD TYPE: ')
            print('1.max lift: lift  2.min drag: drag  3.max lift to drag ratio: lift_drag')
            print('4.max lift & less oscillation: lift_oscillation')
            print('5.max lift & min drag & less oscillation: lift_drag_oscillation')
            self.reward_type = input('Choose the NUMBER: ')
            if self.reward_type == '1':
                self.reward_type = 'lift'
                print('REWARD TYPE = ' + self.reward_type)
                print('\n')
                break
            elif self.reward_type == '2':
                self.reward_type = 'drag'
                print('REWARD TYPE = ' + self.reward_type)
                print('\n')
                break
            elif self.reward_type == '3':
                self.reward_type = 'lift_drag'
                print('REWARD TYPE = ' + self.reward_type)
                print('\n')
                break
            elif self.reward_type == '4':
                self.reward_type = 'lift_oscillation'
                print('REWARD TYPE = ' + self.reward_type)
                print('\n')
                break
            elif self.reward_type == '5':
                self.reward_type = 'lift_drag_oscillation'
                print('REWARD TYPE = ' + self.reward_type)
                print('\n')
                break
            else:
                print('REWARD TYPE FALSE ! ! ')
                print('\n')
                continue

        self.learning_rate = input('LEARNING RATE (float): ')
        self.learning_rate = float(self.learning_rate)
        print('Input LEARNING RATE = ' + str(self.learning_rate))
        print('\n')

    def retrieve_training_configuration(self):
        while True:
            working_folder_name_ = input('Input Working Folder Name:')

            if os.path.exists(self.library_dir + "\\" + working_folder_name_):
                print('Working Folder Exist')
                print('\n')

                pattern_ = r'(?<![eE])-'  # 负向前瞻
                split_name_ = re.split(pattern_, working_folder_name_)

                self.time_stamp = str(split_name_[0])
                self.model_type = str(split_name_[1])
                self.solver_dimension = str(split_name_[2])
                self.policy_type = str(split_name_[3])
                self.action_mode = str(split_name_[4])
                self.reward_type = str(split_name_[5])
                self.learning_rate = float(split_name_[6])

                if self.solver_dimension == '2D':
                    self.solver_dimension = 2
                else:
                    self.solver_dimension = 3

                print(f'Time Stamp: {self.time_stamp}')
                print(f'Modle Type: {self.model_type}')
                print(f'Solver Dimension: {self.solver_dimension}D')
                print(f'Policy Type: {self.policy_type}')
                print(f'Action Mode: {self.action_mode}')
                print(f'Reward Type: {self.reward_type}')
                print(f'Learning Rate: {str(self.learning_rate)}')
                print('\n')
                break

            else:
                print('Working Folder not Exist')
                print('\n')
                continue




