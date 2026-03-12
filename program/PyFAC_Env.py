import numpy as np

import os

import datetime

import time

import shutil

import ansys.fluent.core as pyfluent
from ansys.fluent.core import ScalarFieldDataRequest, VectorFieldDataRequest

import sys

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType

from typing import SupportsFloat, Any

import csv

from LSTMenv import trainLSTM, predictLSTM

import torch


class Env(gym.Env):
    def __init__(
            self,
            action_mode,
            working_library,
            processor_number,
            solver_dimension,
            project_name,
            frequency_range,
            amplitude_range,
            reward_type,
            env_num,
            use_fst,
            use_lstm,
            device
    ):

        self.device = device

        self.use_fst = use_fst
        self.use_lstm = use_lstm

        self.env_num = env_num

        self.max_flowtime = 3.

        self.increment_time = 0.02

        self.iteration_per_step = 20
        self.step_size = 0.001
        self.num_time_step = int(self.increment_time / self.step_size)
        self.cycle_resolution = 10

        self.working_library = working_library
        self.working_folder = None

        self.processor_number = processor_number

        self.project_name = project_name

        self.action_mode = action_mode

        if self.action_mode == 'frequency' or self.action_mode == 'amplitude':  # 动作参数数量
            self.action_dim = 1
        else:
            self.action_dim = 2

        self.env_time_stamp = None
        self.folder_name = None

        self.iteration = 1
        self.next_state = []

        self.cl_dic = {}
        self.cd_dic = {}
        self.cl_cd_dic = {}

        self.cl_avg = None
        self.cd_avg = None
        self.cl_cd_avg = None

        self.initial_cl = None
        self.initial_cd = None
        self.initial_cl_cd = None

        self.reward = None
        self.episode_reward = None
        self.terminated = False
        self.truncated = False

        self.frequency = None
        self.amplitude = None

        self.frequency_range = frequency_range
        self.amplitude_range = amplitude_range

        self.initial_flowtime = None
        self.flowtime = None

        self.reward_type = reward_type

        self.episode_cd = 0.
        self.episode_cl = 0.
        self.episode_cl_cd = 0.

        self.session = None

        self.solver_dimension = solver_dimension

        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        self.wp_group = []
        self.nw_group = []
        self.fw_group = []

        self.initial_wp = []
        self.initial_fp = []
        self.initial_fu = []
        self.initial_fv = []
        self.initial_fw = []
        self.wp = []
        self.fp = []
        self.fu = []
        self.fv = []
        self.fw = []

        self.first_cycle = True
        self.num_episode = 0

        self.start_fluent()

        self.copy_udf_file()

        self.state_dim = len(self.wp_group) + 3 * (len(self.nw_group) + len(self.fw_group)) + 1

        print(f'state dimension: {str(self.state_dim)}')

        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)

        self.episode_time_counter = None
        self.step_time_counter = None
        self.agent_time_counter = None

        self.lstm_in_dim = None
        self.lstm_out_dim = None
        self.lstm_input = []
        self.lstm_output = None

        self.lstm_model = None

        self.step_mse = []
        self.episode_mse = None

        self.lstm_training_loss = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        torch.cuda.empty_cache()

        curr_time = datetime.datetime.now()
        self.env_time_stamp = datetime.datetime.strftime(curr_time, '%Y%m%d%H%M%S')

        self.folder_name = self.env_time_stamp + '-' + str(self.env_num)
        self.working_folder = self.working_library + '\\' + self.folder_name

        os.makedirs(self.working_folder)

        print(f'start reset timestamp: {self.folder_name}')

        self.iteration = 1

        self.cl_dic = {}
        self.cd_dic = {}
        self.cl_cd_dic = {}

        self.terminated = False
        self.truncated = False

        self.episode_cd = 0.
        self.episode_cl = 0.
        self.episode_cl_cd = 0.

        self.step_size = 0.001
        self.num_time_step = int(self.increment_time / self.step_size)

        self.num_episode += 1

        self.init_simulation()

        self.export_cl_cd()

        self.export_initial_state()

        self.initial_flowtime = self.flowtime
        self.initial_cl = self.cl_avg
        self.initial_cd = self.cd_avg
        self.initial_cl_cd = self.cl_cd_avg

        self.lstm_in_dim = self.lstm_out_dim = len(self.initial_wp + self.initial_fp + self.initial_fu + self.initial_fv + self.initial_fw) + 1
        self.lstm_input = []

        self.reset_simulation()

        if self.use_lstm:
            if self.num_episode > 50 and self.num_episode % 10 == 0:
                self.lstm_model, self.lstm_training_loss = trainLSTM(self.working_library, self.device)

                torch.save(self.lstm_model, f'{self.working_folder}\\LSTM_model_{self.num_episode}.pth')

                with open(f'{self.working_folder}\\lstm_training_loss_{self.num_episode}.csv', 'w') as file:
                    csv_writer = csv.writer(file)
                    for row in self.lstm_training_loss:
                        csv_writer.writerow([row])

        self.step_mse = []

        return np.zeros(self.state_dim, dtype=np.float32), {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        if self.action_mode == 'frequency':
            self.frequency = action[0] * (self.frequency_range[1] - self.frequency_range[0]) + self.frequency_range[0]
            self.amplitude = self.amplitude_range[1]
        elif self.action_mode == 'amplitude':
            self.frequency = self.frequency_range[1]
            self.amplitude = action[0] * (self.amplitude_range[1] - self.amplitude_range[0]) + self.amplitude_range[0]
        else:
            self.frequency = action[0] * (self.frequency_range[1] - self.frequency_range[0]) + self.frequency_range[0]
            self.amplitude = action[1] * (self.amplitude_range[1] - self.amplitude_range[0]) + self.amplitude_range[0]

        if self.use_lstm:
            self.lstm_input.append(self.next_state + [self.frequency / 400.])
            if self.num_episode <= 50 or (self.num_episode > 50 and self.num_episode % 10 == 0):
                with open(f'{self.working_library}\\LSTM_input.csv', mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(self.next_state + [self.frequency / 400.])

        with open(
                self.working_folder + '\\' + self.folder_name + '-Action.csv', mode='a', newline=''
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([str(self.iteration).zfill(5), self.amplitude, self.frequency])

        if (self.num_episode <= 50 or (self.num_episode > 50 and self.num_episode % 10 == 0)) or not self.use_lstm:

            self.step_size = 1 / self.frequency / self.cycle_resolution

            self.udf()

            self.one_step_simulation()

            self.export_state()

            self.export_cl_cd()

            self.assemble_next_state()

            if self.flowtime >= self.max_flowtime:
                self.terminated = True

            self.cal_step_reward()

            if self.use_lstm:
                with open(f'{self.working_library}\\LSTM_output.csv', mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(self.next_state + [(self.cl_avg - self.initial_cl) * 100.])

            if self.num_episode > 50 and self.num_episode % 10 == 0 and self.use_lstm:
                self.lstm_mse()

        if self.num_episode > 50 and self.num_episode % 10 != 0 and self.use_lstm:
            self.lstm_surrogate()

        self.iteration += 1

        return self.next_state, self.reward, self.terminated, self.truncated, {}

    def cal_step_reward(self):
        if self.reward_type == 'lift':
            self.reward = (self.cl_avg - self.initial_cl) * 25.

        elif self.reward_type == 'lift_drag':
            self.reward = (self.cl_cd_avg - self.initial_cl_cd) * 5.

        with open(
                self.working_folder + '\\' + self.folder_name + '-StepReward.csv', mode='a', newline=''
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.iteration, self.reward])

        self.episode_cl = self.episode_cl + self.cl_avg * (self.num_time_step * self.step_size)
        self.episode_cd = self.episode_cd + self.cd_avg * (self.num_time_step * self.step_size)
        self.episode_cl_cd = self.episode_cl_cd + self.cl_cd_avg * (self.num_time_step * self.step_size)

        if self.terminated:
            self.episode_cl = self.episode_cl / (self.flowtime - self.initial_flowtime)
            self.episode_cd = self.episode_cd / (self.flowtime - self.initial_flowtime)
            self.episode_cl_cd = self.episode_cl_cd / (self.flowtime - self.initial_flowtime)
            self.cal_episode_reward()

    def cal_episode_reward(self):

        if self.reward_type == 'lift':
            self.episode_reward = (self.episode_cl - self.initial_cl) * 25.

        elif self.reward_type == 'lift_drag':
            self.episode_reward = (self.episode_cl_cd - self.initial_cl_cd) * 5.

        if not os.path.exists(self.working_library + '\\log-episode_reward.csv'):
            with open(
                    self.working_library + '\\log-episode_reward.csv', mode='a', newline=''
            ) as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    ['folder_name', 'episode_reward', 'episode_cl', 'episode_cd', 'episode_cl_cd']
                )

        with open(
                self.working_library + '\\log-episode_reward.csv', mode='a', newline=''
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    self.folder_name,
                    str(self.episode_reward),
                    str(self.episode_cl),
                    str(self.episode_cd),
                    str(self.episode_cl_cd)
                ]
            )

    def assemble_next_state(self):
        self.next_state = []
        self.next_state += [(x - y) / y * 5e2 for x, y in zip(self.wp, self.initial_wp)]
        self.next_state += [(x - y) / y * 5e2 for x, y in zip(self.fp, self.initial_fp)]
        self.next_state += [(x - y) / y * 1. for x, y in zip(self.fu, self.initial_fu)]
        self.next_state += [(x - y) / y * 4e-1 for x, y in zip(self.fv, self.initial_fv)]
        self.next_state += [(x - y) / y * 1e-4 for x, y in zip(self.fw, self.initial_fw)]
        # self.next_state += [self.flowtime - self.initial_flowtime]

        with open(
                self.working_folder + '\\' + self.folder_name + '-next_state.csv', mode='a', newline=''
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.iteration] + self.next_state)

        self.next_state = np.array(self.next_state)

    def init_simulation(self):

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        self.session.settings.file.read_case_data(file_name = self.working_library + "\\" + self.project_name + ".cas.h5")
        self.session.settings.solution.run_calculation.transient_controls.time_step_count = self.num_time_step
        self.session.settings.solution.run_calculation.transient_controls.max_iter_per_time_step = self.iteration_per_step
        self.session.settings.solution.run_calculation.transient_controls.time_step_size = self.step_size

        self.define_report_file()
        self.session.settings.solution.run_calculation.calculate()

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def reset_simulation(self):

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        if self.first_cycle:
            self.session.tui.define.user_defined.compiled_functions('compile', f'"libudf_{str(self.env_num).zfill(2)}"', 'yes', f'"udf_{str(self.env_num).zfill(2)}.c"', f'"DEFINE_SOURCE_{str(self.env_num).zfill(2)}.c"', '""', '""')
            self.first_cycle = False

        self.session.tui.define.user_defined.compiled_functions('load', f'"libudf_{str(self.env_num).zfill(2)}"')

        if self.use_fst:
            self.session.settings.setup.cell_zone_conditions.fluid['unspecified'] = {"source_terms" : {"source_terms" : {"energy" : [], "omega" : [], "k" : [], "x-momentum" : [{"udf" : f"x_mom_source::libudf_{str(self.env_num).zfill(2)}", "option" : "udf"}], "y-momentum" : [{"udf" : f"y_mom_source::libudf_{str(self.env_num).zfill(2)}", "option" : "udf"}], "z-momentum" : [{"udf" : f"z_mom_source::libudf_{str(self.env_num).zfill(2)}", "option" : "udf"}], "mass" : []}, "sources" : True}}
            self.session.settings.solution.run_calculation.transient_controls.time_step_count = 100
            self.session.settings.solution.run_calculation.calculate()

        if self.use_lstm:
            self.export_state()
            self.export_cl_cd()
            self.assemble_next_state()
            for _ in range(9):
                self.lstm_input.append(self.next_state + [0])
                if self.num_episode <= 50 or (self.num_episode > 50 and self.num_episode % 10 == 0):
                    with open(f'{self.working_library}\\LSTM_input.csv', mode='a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(self.next_state + [0])
                self.session.settings.solution.run_calculation.calculate()
                self.export_state()
                self.export_cl_cd()
                self.assemble_next_state()
                if self.num_episode <= 50 or (self.num_episode > 50 and self.num_episode % 10 == 0):
                    with open(f'{self.working_library}\\LSTM_output.csv', mode='a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(self.next_state + [(self.cl_avg - self.initial_cl) * 100.])

        self.session.settings.setup.boundary_conditions.velocity_inlet['slot1'] = {
            "momentum": {"flow_direction": [0.97437, 0.224951], "velocity": {"udf": f"slot_velocity_1::libudf_{str(self.env_num).zfill(2)}", "option": "udf"},
                         "velocity_specification_method": "Magnitude and Direction"}}
        self.session.settings.setup.boundary_conditions.velocity_inlet['slot2'] = {
            "momentum": {"flow_direction": [0.97437, 0.224951], "velocity": {"udf": f"slot_velocity_2::libudf_{str(self.env_num).zfill(2)}", "option": "udf"},
                         "velocity_specification_method": "Magnitude and Direction"}}

        self.session.settings.solution.run_calculation.transient_controls.udf_hook = f"adjust_step_size::libudf_{str(self.env_num).zfill(2)}"

        time.sleep(5)

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def export_initial_state(self):
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        field_data = self.session.fields.field_data

        wall_pressure = field_data.get_field_data(
            ScalarFieldDataRequest(field_name="pressure", surfaces=self.wp_group)
        )
        self.initial_wp = []
        for element in self.wp_group:
            self.initial_wp += wall_pressure[element].tolist()

        field_pressure = field_data.get_field_data(
            ScalarFieldDataRequest(field_name="pressure", surfaces=self.nw_group + self.fw_group)
        )
        self.initial_fp = []
        for element in self.nw_group:
            self.initial_fp += field_pressure[element].tolist()
        for element in self.fw_group:
            self.initial_fp += field_pressure[element].tolist()

        field_velocity = field_data.get_field_data(
            VectorFieldDataRequest(field_name="velocity", surfaces=self.nw_group + self.fw_group)
        )

        self.initial_fu = []
        for element in self.nw_group:
            self.initial_fu.append(field_velocity[element].tolist()[0][0])
        for element in self.fw_group:
            self.initial_fu.append(field_velocity[element].tolist()[0][0])

        self.initial_fv = []
        for element in self.nw_group:
            self.initial_fv.append(field_velocity[element].tolist()[0][1])
        for element in self.fw_group:
            self.initial_fv.append(field_velocity[element].tolist()[0][1])

        self.initial_fw = []
        for element in self.nw_group:
            self.initial_fw.append(field_velocity[element].tolist()[0][2])
        for element in self.fw_group:
            self.initial_fw.append(field_velocity[element].tolist()[0][2])

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def one_step_simulation(self):

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        self.session.tui.define.user_defined.compiled_functions(
            'compile', f'"libudf_{str(self.env_num).zfill(2)}"', 'yes', '"y"', f'"udf_{str(self.env_num).zfill(2)}.c"', f'"DEFINE_SOURCE_{str(self.env_num).zfill(2)}.c"', '""', '""'
        )

        num_cycle = self.increment_time * self.frequency
        if num_cycle <= 1.:
            num_cycle = 1
        else:
            num_cycle = round(num_cycle)
        self.num_time_step = self.cycle_resolution * num_cycle

        self.session.settings.solution.run_calculation.transient_controls.time_step_count = self.num_time_step
        self.session.settings.solution.run_calculation.calculate()

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def export_state(self):

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        field_data = self.session.fields.field_data

        wall_pressure = field_data.get_field_data(
            ScalarFieldDataRequest(field_name="pressure", surfaces=self.wp_group)
        )
        self.wp = []
        for element in self.wp_group:
            self.wp += wall_pressure[element].tolist()

        field_pressure = field_data.get_field_data(
            ScalarFieldDataRequest(field_name="pressure", surfaces=self.nw_group + self.fw_group)
        )
        self.fp = []
        for element in self.nw_group:
            self.fp += field_pressure[element].tolist()
        for element in self.fw_group:
            self.fp += field_pressure[element].tolist()

        field_velocity = field_data.get_field_data(
            VectorFieldDataRequest(field_name="velocity", surfaces=self.nw_group + self.fw_group)
        )

        self.fu = []
        for element in self.nw_group:
            self.fu.append(field_velocity[element].tolist()[0][0])
        for element in self.fw_group:
            self.fu.append(field_velocity[element].tolist()[0][0])

        self.fv = []
        for element in self.nw_group:
            self.fv.append(field_velocity[element].tolist()[0][1])
        for element in self.fw_group:
            self.fv.append(field_velocity[element].tolist()[0][1])

        self.fw = []
        for element in self.nw_group:
            self.fw.append(field_velocity[element].tolist()[0][2])
        for element in self.fw_group:
            self.fw.append(field_velocity[element].tolist()[0][2])

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def export_cl_cd(self):
        cl_ = []
        cd_ = []
        cl_cd_ = []

        with open(self.working_folder + "\\" + self.folder_name + "-monitor-lift-drag.out", 'r') as file_:
            lines_ = file_.readlines()

            self.flowtime = float(lines_[-1].split()[1])

            for line_ in lines_[-10:]:
                parts = line_.split()

                cl_.append(float(parts[2]))
                cd_.append(float(parts[3]))
                cl_cd_.append(float(parts[2]) / float(parts[3]))

        self.cl_avg = np.mean(cl_)
        self.cd_avg = np.mean(cd_)
        self.cl_cd_avg = np.mean(cl_cd_)

    def copy_working_files(self):
        source_file = self.working_library + '\\' + self.project_name + '.cas.h5'
        target_folder = self.working_folder

        shutil.copy2(source_file, target_folder)

        copied_file = os.path.join(target_folder, self.project_name + '.cas.h5')
        while not os.path.exists(copied_file):
            pass

        source_file = self.working_library + '\\' + self.project_name + '.dat.h5'
        target_folder = self.working_folder

        shutil.copy2(source_file, target_folder)

        copied_file = os.path.join(target_folder, self.project_name + '.dat.h5')
        while not os.path.exists(copied_file):
            pass

    def delete_sim_files(self):
        time.sleep(10)

        for filename in os.listdir(self.working_folder):
            if filename.endswith('.cas.h5') or filename.endswith('.dat.h5'):
                full_file_path = os.path.join(self.working_folder, filename)

                try:
                    os.remove(full_file_path)
                except FileNotFoundError:
                    pass

    def udf(self):
        with open(self.working_library + f'\\udf_{str(self.env_num).zfill(2)}.c', 'w') as file_:
            file_.write('#include "udf.h"\n')
            file_.write('\n')
            file_.write('#define AMPLITUDE ' + str(self.amplitude) + '\n')
            file_.write('#define FREQUENCY ' + str(self.frequency) + '\n')
            file_.write('\n')
            file_.write('DEFINE_PROFILE(slot_velocity_1,t,i)\n')
            file_.write('{\n')
            file_.write('face_t f;\n')
            file_.write('real time;\n')
            file_.write('time=CURRENT_TIME;\n')
            file_.write('begin_f_loop(f,t)\n')
            file_.write('{\n')
            file_.write('F_PROFILE(f,t,i)=AMPLITUDE*sin(2.*FREQUENCY*M_PI*(time-' + str(self.flowtime) + '));\n')
            file_.write('}\n')
            file_.write('end_f_loop(f,t)\n')
            file_.write('}\n')
            file_.write('\n')
            file_.write('DEFINE_PROFILE(slot_velocity_2,a,b)\n')
            file_.write('{\n')
            file_.write('face_t c;\n')
            file_.write('real time;\n')
            file_.write('time=CURRENT_TIME;\n')
            file_.write('begin_c_loop(c,a)\n')
            file_.write('{\n')
            file_.write('F_PROFILE(c,a,b)=AMPLITUDE*sin(2.*FREQUENCY*M_PI*(time-' + str(self.flowtime) + ')+M_PI);\n')
            file_.write('}\n')
            file_.write('end_c_loop(c,a)\n')
            file_.write('}\n')
            file_.write('\n')
            file_.write('DEFINE_DELTAT(adjust_step_size, domain)\n')
            file_.write('{\n')
            file_.write('real step_size;\n')
            file_.write('step_size = ' + str(self.step_size) + ';\n')
            file_.write('return step_size;\n')
            file_.write('}\n')

    def start_fluent(self):

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        self.session = pyfluent.launch_fluent(product_version="24.1", dimension=self.solver_dimension, precision="double",
                                              processor_count=self.processor_number, ui_mode="no_gui_or_graphics",
                                              py=True, cwd=self.working_library)

        self.session.settings.file.read_case_data(file_name = self.working_library + "\\" + self.project_name + ".cas.h5")

        field_info = self.session.fields.field_info
        surface_info = field_info.get_surfaces_info()

        self.wp_group = [key for key in surface_info if key.startswith('wp_')]
        self.wp_group.sort(key=lambda x: int(x[-2:]))

        self.nw_group = [key for key in surface_info if key.startswith('nw_')]
        self.nw_group.sort(key=lambda x: int(x[-2:]))

        self.fw_group = [key for key in surface_info if key.startswith('fw_')]
        self.fw_group.sort(key=lambda x: int(x[-2:]))

        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    def define_report_file(self):

        self.session.settings.solution.monitor.report_files['monitor-lift-drag'] = {}
        self.session.settings.solution.monitor.report_files['monitor-lift-drag'] = {
            "report_defs": ["flow-time", "report-cl", "report-cd"],
            "file_name": self.working_folder + "\\" + self.folder_name + "-monitor-lift-drag.out"
        }

        self.session.settings.solution.monitor.report_files['monitor-slots'] = {}
        self.session.settings.solution.monitor.report_files['monitor-slots'] = {
            "report_defs": ["flow-time", "report-slot1", "report-slot2"],
            "file_name": self.working_folder + "\\" + self.folder_name + "-monitor-slots.out"
        }

    def copy_udf_file(self):

        source_file = self.working_library + '\\udf.c'
        destination_file = self.working_library + f'\\udf_{str(self.env_num).zfill(2)}.c'
        shutil.copy(source_file, destination_file)

        source_file = self.working_library + '\\DEFINE_SOURCE.c'
        destination_file = self.working_library + f'\\DEFINE_SOURCE_{str(self.env_num).zfill(2)}.c'
        shutil.copy(source_file, destination_file)

    def lstm_mse(self):
        self.lstm_output = predictLSTM(self.lstm_model, self.lstm_input[-10:], self.device)
        self.step_mse.append(np.mean((np.array(self.lstm_output) - np.array(
            self.next_state + [(self.cl_avg - self.initial_cl) * 100.])) ** 2))
        with open(f'{self.working_folder}\\LSTM_step_mse_{self.num_episode}.csv', mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.num_episode, self.iteration, self.step_mse])

        if self.terminated:
            self.episode_mse = np.mean(np.array(self.step_mse))
            with open(f'{self.working_library}\\LSTM_episode_mse.csv', mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([self.num_episode, self.episode_mse])

    def lstm_surrogate(self):
        self.lstm_output = predictLSTM(self.lstm_model, self.lstm_input[-10:], self.device)
        self.next_state = self.lstm_output[:-1]
        self.reward = self.lstm_output[-1]

        num_cycle = self.increment_time * self.frequency
        if num_cycle <= 1.:
            num_cycle = 1
        else:
            num_cycle = round(num_cycle)
        self.flowtime += num_cycle / self.frequency

        if self.flowtime >= self.max_flowtime:
            self.terminated = True

        with open(
                self.working_folder + '\\' + self.folder_name + '-StepReward.csv', mode='a', newline=''
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([self.iteration, self.reward])

        self.cl_avg = self.reward / 25. + self.initial_cl

        self.episode_cl = self.episode_cl + self.cl_avg * (num_cycle / self.frequency)

        if self.terminated:
            self.episode_cl = self.episode_cl / (self.flowtime - self.initial_flowtime)
            self.cal_episode_reward()


