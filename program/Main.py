import os
import shutil

from PyFAC_2D_3D import PyFAC
from Input_Setup import InputSetup

import torch


def copy_working_files(cwd_, working_library_, project_name_):
    for file_name_ in os.listdir(cwd_):
        if file_name_.endswith(('.py', '.c')):
            source_file_ = cwd_ + '\\' + file_name_
            shutil.copy2(source_file_, working_library_)
            copied_file_ = os.path.join(working_library_, source_file_)
            while not os.path.exists(copied_file_):
                pass
        elif file_name_.endswith('.h5'):
            if file_name_.startswith(project_name_):
                source_file_ = cwd_ + '\\' + file_name_
                shutil.copy2(source_file_, working_library_)
                copied_file_ = os.path.join(working_library_, source_file_)
                while not os.path.exists(copied_file_):
                    pass
    print('All Working Files Copied.\n')

if __name__ == "__main__":
    cwd = os.getcwd()
    parent_dir = os.path.dirname(cwd)
    library_dir = parent_dir + '\\Library'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(library_dir):
        os.makedirs(library_dir)

    (processor_number, parallel, time_stamp, model_type, solver_dimension, policy_type,
     action_mode, reward_type, learning_rate, retrieve_training, use_fst, use_lstm) = InputSetup(library_dir).input_setup()

    frequency_range = [50, 400]
    amplitude_range = [1, 50]

    working_folder = (f'{library_dir}\\{time_stamp}-{model_type}-{solver_dimension}D-{policy_type}-'
                      f'{action_mode}-{reward_type}-{learning_rate}')
    if not os.path.exists(working_folder):
        os.makedirs(working_folder)

    if solver_dimension == 2:
        if model_type == 'RAE2822':
            project_name = 'RAE2822-2D-AFC-48k-0.01-0.6-0.65-FST'
        else:
            project_name = 'NACA0012-2D-AFC-11k-0.005-0.6-0.65'
    else:
        if model_type == 'RAE2822':
            project_name = 'RAE2822-2D-AFC-553k-0.005-0.6-0.65'
        else:
            project_name = 'NACA0012-3D-AFC-349k-0.005-0.6-0.65'

    if action_mode == 'frequency' or action_mode == 'amplitude' or action_mode == 'action':  # 动作参数数量
        action_dim = 1
    else:
        action_dim = 2

    copy_working_files(cwd, working_folder, project_name)

    PyFAC(
        action_mode,
        working_folder,
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
    ).model_learn()


