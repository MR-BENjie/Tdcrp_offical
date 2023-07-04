import warnings

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
import gym
import rlbench.gym
import hydra
import numpy as np
import torch
from dm_env import specs
import argparse
import utils
from logger import Logger
from video import TrainVideoRecorder, VideoRecorder
import pickle
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import PlugChargerInPowerSupply, UnplugCharger, ScrewNail, TakeUsbOutOfComputer, InsertUsbInComputer

def make_agent(obs_spec, action_spec, cfg):
	print(obs_spec.shape)
	cfg.obs_shape = obs_spec.shape
	cfg.action_shape = action_spec.shape

	return hydra.utils.instantiate(cfg)


class Workspace:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		#utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		#self.agent = make_agent(self.train_env.observation_spec(),
		#						self.train_env.action_spec(), cfg.agent)

		"""self._action_spec = specs.BoundedArray(self.train_env.action_space.shape,
											   np.float32,
											   self.train_env.action_space.low,
											   self.train_env.action_space.high,
											   'action')
		self._obs_spec = {}
		self._obs_spec['pixels'] = specs.BoundedArray(shape=self.train_env.observation_space["left_shoulder_rgb"].shape,
													  dtype=np.uint8,
													  minimum=self.train_env.observation_space["left_shoulder_rgb"].low,
													  maximum=self.train_env.observation_space["left_shoulder_rgb"].high,
													  name='observation')
		self._obs_spec['features'] = specs.Array(shape=self.train_env.observation_space["state"].shape,
												 dtype=np.float32,
												 name='observation')
		self.agent = make_agent(self._obs_spec['features'],
								self._action_spec, cfg.agent)"""

		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0

	def setup(self):
		# create envs
		task = self.cfg.task+"-vision-v0"
		print(task)

		#self.eval_env = gym.make(task, render_mode="human")
		#self.train_env = self.eval_env

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(
			self.work_dir if self.cfg.save_train_video else None)

		self.live_demos = True
		DATASET = '' if self.live_demos else 'PATH/TO/YOUR/DATASET'

		obs_config = ObservationConfig()
		obs_config.set_all(True)

		self.env = Environment(
			action_mode=MoveArmThenGripper(
				arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
			obs_config=ObservationConfig(),
			headless=False)
		self.env.launch()

		# task = env.get_task(ReachTarget)


		# An example of using the demos to 'train' using behaviour cloning loss.

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.action_repeat

	def generate_eval(self):
		step, episode, total_reward = 0, 0, 0
		generate_until_episode = utils.Until(self.cfg.num_demos)
		observations_list = list()
		states_list = list()
		actions_list = list()
		rewards_list = list()
		save_every = 1
		batch_size = 64
		task = self.env.get_task(PlugChargerInPowerSupply)
		#UnplugCharger, ScrewNail, TakeUsbOutOfComputer, InsertUsbInComputer

		demos = task.get_demos(self.cfg.num_demos, live_demos=self.live_demos)  # -> List[List[Observation]]
		demos = np.array(demos).flatten()
		while generate_until_episode(episode):
			print("one episode")

			batch = np.random.choice(demos, replace=False)
			batch_images = [obs.left_shoulder_rgb.T for obs in batch]
			batch_state = []
			for obs in batch:
				low_dim_data = [] if obs.gripper_open is None else [[obs.gripper_open]]
				for data in [obs.joint_velocities, obs.joint_positions,
							 obs.joint_forces,
							 obs.gripper_pose, obs.gripper_joint_positions,
							 obs.gripper_touch_forces, obs.task_low_dim_state]:
					if data is not None:
						low_dim_data.append(data)

				obs_low_dim_state = np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
				batch_state.append(obs_low_dim_state)
			ground_truth_actions = [obs.joint_velocities for obs in batch]
			rewards = [1.0 for _ in batch]
			episode += 1

			#rewards_list.append(rewards)
			#observations_list.append(np.stack(batch_images, 0))
			#states_list.append(np.stack(batch_state, 0))
			#actions_list.append(np.stack(ground_truth_actions, 0))
			rewards_list.extend(rewards)
			observations_list.extend(batch_images)
			states_list.extend(batch_state)
			actions_list.extend(ground_truth_actions)

			# Make np arrays
		observations_list_temp = list()
		states_list_temp = list()
		actions_list_temp = list()
		rewards_list_temp = list()

		x = (int)(len(rewards_list)/batch_size)
		for i in range(x):
			rewards_list_temp.append(np.stack(rewards_list[i*batch_size:i*batch_size+batch_size], 0 ))
			states_list_temp.append(np.stack(states_list[i * batch_size:i * batch_size + batch_size], 0))
			actions_list_temp.append(np.stack(actions_list[i * batch_size:i * batch_size + batch_size], 0))
			observations_list_temp.append(np.stack(observations_list[i * batch_size:i * batch_size + batch_size], 0))
		# Save demo in pickle file


		save_dir = Path(self.work_dir)
		save_dir.mkdir(parents=True, exist_ok=True)
		snapshot_path = save_dir / 'expert_demos.pkl'
		payload = [
			observations_list_temp, states_list_temp, actions_list_temp, rewards_list_temp
		]


		with open(str(snapshot_path), 'wb') as f:
			pickle.dump(payload, f)

	"""def generate_eval(self):
		step, episode, total_reward = 0, 0, 0
		generate_until_episode = utils.Until(self.cfg.num_demos)
		observations_list = list()
		states_list = list()
		actions_list = list()
		rewards_list = list()
		save_every = 1
		while generate_until_episode(episode):
			print("one episode")
			observations = list()
			states = list()
			actions = list()
			rewards = list()
			episode_reward = 0
			i = 0
			#time_step = self.eval_env.reset()
			observation= self.eval_env.reset()
			done = False

			self.video_recorder.init(self.eval_env)
			while not done:
				#if i % save_every == 0:
				#	observations.append(time_step.observation['pixels'])
				#	states.append(time_step.observation['features'])
				#	rewards.append(time_step.reward)
				with torch.no_grad(), utils.eval_mode(self.agent):
					action = self.agent.act(
						observation["state"],
						self.global_step,
						eval_mode=True)

				observation, reward, done, _ = self.eval_env.step(action)

				if i % save_every == 0:
					observations.append(observation["left_shoulder_rgb"])
					states.append(observation["state"])
					rewards.append(reward)

				if i % save_every == 0:
					actions.append(action)
					self.video_recorder.record(self.eval_env)
				total_reward += reward
				episode_reward += reward
				step += 1
				i = i + 1
			print(episode, episode_reward)
			if episode_reward > 100:
				episode += 1
				self.video_recorder.save(f'{episode}_eval.mp4')
				rewards_list.append(np.array(rewards))
				observations_list.append(np.stack(observations, 0))
				states_list.append(np.stack(states, 0))
				actions_list.append(np.stack(actions, 0))
				
		# Make np arrays
		observations_list = np.array(observations_list)
		states_list = np.array(states_list)
		actions_list = np.array(actions_list)
		rewards_list = np.array(rewards_list)

		# Save demo in pickle file
		save_dir = Path(self.work_dir)
		save_dir.mkdir(parents=True, exist_ok=True)
		snapshot_path = save_dir / 'expert_demos.pkl'
		payload = [
			observations_list, states_list, actions_list, rewards_list
		]

		with open(str(snapshot_path), 'wb') as f:
			pickle.dump(payload, f)"""

	def load_snapshot(self, snapshot):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot(agent_payload)


@hydra.main(config_path='cfgs', config_name='config_generate')
def main(cfg):
	from generate import Workspace as W
	root_dir = Path.cwd()
	workspace = W(cfg)

	# Load weights	
	snapshot = Path(cfg.weight)
	if snapshot.exists():
		print(f'resuming: {snapshot}')
		workspace.load_snapshot(snapshot)
	
	workspace.generate_eval()


if __name__ == '__main__':
	main()
