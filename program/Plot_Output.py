import os

import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class PlotOutput:
    def __init__(
            self,
            working_library_
    ):
        self.working_library = working_library_
        self.plot_folder = self.working_library + '\\plot_folder'

        self.time_stamp_log = []

        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def plot_output(self):

        if not os.path.exists(self.working_library + '\\log-episode_reward.csv'):
            return

        with open(self.working_library + '\\log-episode_reward.csv', 'r') as file_:
            reader_ = csv.reader(file_)
            next(reader_)
            for row_ in reader_:
                if row_ == []:
                    continue

                if row_[0] not in self.time_stamp_log:
                    folder_name = row_[0]
                    self.time_stamp_log.append(row_[0])

                    self.plot_episode_result(folder_name)

        self.plot_episode_reward()

    def plot_episode_result(self, folder_name):

        if os.path.exists(self.working_library + '\\' + folder_name + '\\' + folder_name + '-monitor-lift-drag.out'):

            timestep_ = []
            flowtime_ = []
            cd_ = []
            cl_ = []

            with open(self.working_library + '\\' + folder_name + '\\' + folder_name + '-monitor-lift-drag.out', 'r') as file_:
                lines_ = file_.readlines()

            for line_ in lines_[3:]:
                parts_ = line_.split()
                timestep_.append(int(parts_[0]))
                flowtime_.append(float(parts_[1]))
                cl_.append(float(parts_[2]))
                cd_.append(float(parts_[3]))

            self.plot_line(self.plot_folder, folder_name + '-Cd.png', flowtime_, cd_, 'Flowtime', 'Cd')
            self.plot_line(self.plot_folder, folder_name + '-Cl.png', flowtime_, cl_, 'Flowtime', 'Cl')

        if os.path.exists(self.working_library + '\\' + folder_name + '\\' + folder_name + '-monitor-slots.out'):

            timestep_ = []
            flowtime_ = []
            slot1_ = []
            slot2_ = []

            with open(self.working_library + '\\' + folder_name + '\\' + folder_name + '-monitor-slots.out', 'r') as file_:
                lines_ = file_.readlines()

            for line_ in lines_[3:]:
                parts_ = line_.split()
                timestep_.append(int(parts_[0]))
                flowtime_.append(float(parts_[1]))
                slot1_.append(float(parts_[2]))
                slot2_.append(float(parts_[3]))

            self.plot_line(self.plot_folder, folder_name + '-slot1.png', flowtime_, slot1_, 'Flowtime', 'slot1')
            self.plot_line(self.plot_folder, folder_name + '-slot2.png', flowtime_, slot2_, 'Flowtime', 'slot2')

        if os.path.exists(self.working_library + '\\' + folder_name + '\\' + folder_name + '-report-clcd-ft.out'):

            timestep_ = []
            ft_ = []
            cd_ = []
            cl_ = []

            with open(self.working_library + '\\' + folder_name + '\\' + folder_name + '-report-clcd-ft.out', 'r') as file_:
                lines_ = file_.readlines()

            for line_ in lines_[3:]:
                parts_ = line_.split()
                timestep_.append(int(parts_[0]))
                ft_.append(float(parts_[1]))
                cd_.append(float(parts_[2]))
                cl_.append(float(parts_[3]))

            self.plot_line(self.plot_folder, folder_name + '-Cd.png', ft_, cd_, 'Flowtime', 'Cd')
            self.plot_line(self.plot_folder, folder_name + '-Cl.png', ft_, cl_, 'Flowtime', 'Cl')

        if os.path.exists(self.working_library + '\\' + folder_name + '\\' + folder_name + '-Action.csv'):

            iteration_ = []
            amplitude_ = []
            frequency_ = []

            with open(self.working_library + '\\' + folder_name + '\\' + folder_name + '-Action.csv', 'r') as file_:
                reader_ = csv.reader(file_)
                for row_ in reader_:
                    if row_ == []:
                        continue

                    iteration_.append(int(row_[0]))
                    amplitude_.append(float(row_[1]))
                    frequency_.append(float(row_[2]))

            self.plot_scatter(
                self.plot_folder,
                folder_name + '-Frequency.png',
                iteration_,
                frequency_,
                'Iteration',
                'Frequency'
            )
            self.plot_scatter(
                self.plot_folder,
                folder_name + '-Amplitude.png',
                iteration_,
                amplitude_,
                'Iteration',
                'Amplitude'
            )

        if os.path.exists(self.working_library + '\\' + folder_name + '\\' + folder_name + '-next_state.csv'):

            x_ = []
            states_ = []
            with open(self.working_library + '\\' + folder_name + '\\' + folder_name + '-next_state.csv', 'r') as file_:
                reader_ = csv.reader(file_)

                for row_ in reader_:
                    if row_ == []:
                        continue

                    for part_ in row_[1:]:
                        x_.append(int(row_[0]))
                        states_.append(float(part_[1:]))

            self.plot_scatter(
                self.plot_folder,
                folder_name + '-Training_data.png',
                x_,
                states_,
                'value',
                'scaled_states'
            )

        if os.path.exists(self.working_library + '\\' + folder_name + '\\' + folder_name + '-StepReward.csv'):

            iteration_ = []
            step_reward_ = []
            with open(self.working_library + '\\' + folder_name + '\\' + folder_name + '-StepReward.csv', 'r') as file_:
                reader_ = csv.reader(file_)

                for row_ in reader_:
                    if row_ == []:
                        continue

                    iteration_.append(int(row_[0]))
                    step_reward_.append(float(row_[1]))

            self.plot_scatter(
                self.plot_folder,
                folder_name + '-Step Reward.png',
                iteration_,
                step_reward_,
                'Iteration',
                'Step Reward')

    def plot_episode_reward(self):

        episode_ = []
        episode_reward_ = []
        episode_cl_ = []
        episode_cd_ = []
        episode_cl_cd_ = []
        cl_max_ = []
        cl_min_ = []
        steady_time_ = []

        count_ = 0
        time_stamp_ = ''

        if not os.path.exists(self.working_library + '\\log-episode_reward.csv'):
            return

        with open(self.working_library + '\\log-episode_reward.csv', 'r') as file_:
            reader_ = csv.reader(file_)
            next(reader_)
            for row_ in reader_:
                if row_ == []:
                    continue

                if time_stamp_ != row_[0].split('-')[0]:
                    count_ += 1
                    time_stamp_ = row_[0].split('-')[0]

                episode_.append(count_)
                episode_reward_.append(float(row_[1]))
                episode_cl_.append(float(row_[2]))
                episode_cd_.append(float(row_[3]))
                episode_cl_cd_.append(float(row_[4]))

        self.plot_line(
            self.plot_folder,
            'Episode-Reward.png',
            episode_,
            episode_reward_,
            'Episode',
            'Episode Reward'
        )
        self.plot_line(
            self.plot_folder,
            'Episode-Cl.png',
            episode_,
            episode_cl_,
            'Episode',
            'Episode Cl'
        )
        self.plot_line(
            self.plot_folder,
            'Episode-Cd.png',
            episode_,
            episode_cd_,
            'Episode',
            'Episode Cd'
        )

        self.plot_line(
            self.plot_folder,
            'Episode-Cl_Cd.png',
            episode_,
            episode_cl_cd_,
            'Episode',
            'Episode Cl/Cd'
        )

    @staticmethod
    def plot_line(plot_folder_, file_name_, x_group_, y_group_, x_label_, y_label_):
        # 创建图形
        fig = plt.figure(figsize=(16, 9), dpi=100)
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

        ax.plot(x_group_, y_group_, label='Agent', color='#000066')

        # 设置坐标轴标签和标题
        ax.set_xlabel(x_label_, fontsize=10)
        ax.set_ylabel(y_label_, fontsize=10)

        if y_label_ == 'Cd':
            # 设置坐标轴范围
            ax.set_xlim(1., 3.)  # 设置 X 轴范围
            # ax.set_ylim(0., 0.1)  # 设置 Y 轴范围
            # 设置 X 轴刻度间隔为 10
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
            # 设置 Y 轴刻度间隔为 40
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

        elif y_label_ == 'Cl':
            # 设置坐标轴范围
            ax.set_xlim(1., 3.)  # 设置 X 轴范围
            # ax.set_ylim(1.2, 1.35)  # 设置 Y 轴范围
            # 设置 X 轴刻度间隔为 10
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
            # 设置 Y 轴刻度间隔为 40
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

        plt.grid()

        # 保存图形
        plt.savefig(plot_folder_ + '\\' + file_name_, dpi=300)  # 保存为300 DPI的PNG格式

        plt.close()

    @staticmethod
    def plot_scatter(plot_folder_, file_name_, x_group_, y_group_, x_label_, y_label_):
        # 创建图形
        fig = plt.figure(figsize=(16, 9), dpi=100)
        ax = fig.add_axes([0.1, 0.1, 0.85, 0.85])

        ax.scatter(x_group_, y_group_, label='Agent', color='#000066')

        # 设置坐标轴标签和标题
        ax.set_xlabel(x_label_, fontsize=10)
        ax.set_ylabel(y_label_, fontsize=10)

        if y_label_ == 'Frequency':
            # 设置坐标轴范围
            ax.set_ylim(0, 400)  # 设置 Y 轴范围
            # 设置 X 轴刻度间隔为 10
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            # 设置 Y 轴刻度间隔为 40
            ax.yaxis.set_major_locator(ticker.MultipleLocator(40))

        elif y_label_ == 'Amplitude':
            # 设置坐标轴范围
            ax.set_ylim(0, 50)  # 设置 Y 轴范围
            # 设置 X 轴刻度间隔为 10
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            # 设置 Y 轴刻度间隔为 40
            ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

        elif y_label_ == 'Action':
            # 设置坐标轴范围
            ax.set_ylim(-0.1, 0.1)  # 设置 Y 轴范围
            # 设置 X 轴刻度间隔为 10
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            # 设置 Y 轴刻度间隔为 40
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))

        plt.grid()

        # 保存图形
        plt.savefig(plot_folder_ + '\\' + file_name_, dpi=300)  # 保存为300 DPI的PNG格式

        plt.close()