from stable_baselines3.common.callbacks import BaseCallback

import os

class ModelBufferSaveCallback(BaseCallback):
    def __init__(self, save_freq, save_path, num_steps, verbose=0, plotter=None):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.num_steps = num_steps
        self.model_version = self.num_steps // self.save_freq
        self.plotter = plotter

    def _on_step(self):

        self.num_steps += 1
        print(f'num_steps: {self.num_steps}')

        if self.num_steps >= self.save_freq * self.model_version:

            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            self.model.save(self.save_path + '\\final_model')
            self.model.save(self.save_path + '\\' + str(self.model_version).zfill(5) + '-model')

            if hasattr(self.model, 'replay_buffer'):
                self.model.save_replay_buffer(self.save_path + '\\final_buffer')

            self.plotter.plot_output()

            self.model_version += 1

        return True