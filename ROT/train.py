#!/usr/bin/env python3

import warnings
import os
from gym import spaces
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
import gym
import rlbench.gym
import hydra
import numpy as np
import torch
from dm_env import specs
from tqdm import tqdm

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_expert_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle

warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
	cfg.obs_shape = obs_spec[cfg.obs_type].shape
	cfg.action_shape = action_spec.shape

	return hydra.utils.instantiate(cfg)

class WorkspaceIL:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(self._obs_spec,
								self._action_spec, cfg.agent)

		if repr(self.agent) == 'drqv2':
			self.cfg.suite.num_train_frames = self.cfg.num_train_frames_drq
		if repr(self.agent) == 'bc':
			self.cfg.suite.num_train_frames = self.cfg.num_train_frames_bc
			self.cfg.suite.num_seed_frames = 0

		self.expert_replay_loader = make_expert_replay_loader(
			self.cfg.expert_dataset, self.cfg.batch_size // 2, self.cfg.num_demos, self.cfg.obs_type)
		self.expert_replay_iter = iter(self.expert_replay_loader)
			
		self.timer = utils.Timer()
		self._global_step = 1
		self._global_episode = 0

		with open(self.cfg.expert_dataset, 'rb') as f:
			if self.cfg.obs_type == 'pixels':
				self.expert_demo, _, _, self.expert_reward = pickle.load(f)
			elif self.cfg.obs_type == 'features':
				_, self.expert_demo, _, self.expert_reward = pickle.load(f)

		self.expert_demo = self.expert_demo[:self.cfg.num_demos]
		self.expert_reward = np.mean(self.expert_reward[:self.cfg.num_demos])
		
	def setup(self):
		# create logger
		self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
		# create envs
		#self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		#self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		task = self.cfg.task + "-vision-v0"
		print(task)

		self.eval_env = gym.make(task, observation_mode ='vision')
		self.train_env = self.eval_env

		self._action_spec = specs.BoundedArray(self.train_env.action_space.shape,
											   np.float32,
											   self.train_env.action_space.low,
											   self.train_env.action_space.high,
											   'action')

		self._obs_spec = {}
		left_shoulder_rgb_obs = spaces.Box(
			low=0, high=1, shape=(3,128,128))
		self._obs_spec['pixels'] = specs.BoundedArray(shape= left_shoulder_rgb_obs.shape,
													  dtype=np.uint8,
													  minimum=left_shoulder_rgb_obs.low,
													  maximum=left_shoulder_rgb_obs.high,
													  name='observation')
		self._obs_spec['features'] = specs.Array(shape=self.train_env.observation_space["state"].shape,
												 dtype=np.float64,
												 name='observation')

		self.swap_obs_type = {"pixels":"left_shoulder_rgb", "features":"state"}

		# create replay buffer
		data_specs = [
			self._obs_spec[self.cfg.obs_type],
			self._action_spec,
			specs.Array((1, ), np.float32, 'reward'),
			specs.Array((1, ), np.float32, 'discount')
		]

		self.replay_storage = ReplayBufferStorage(data_specs,
												  self.work_dir / 'buffer')

		self.replay_loader = make_replay_loader(
			self.work_dir / 'buffer', self.cfg.replay_buffer_size,
			self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
			self.cfg.suite.save_snapshot, self.cfg.nstep, self.cfg.suite.discount)

		self._replay_iter = None
		self.expert_replay_iter = None

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(
			self.work_dir if self.cfg.save_train_video else None)

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.suite.action_repeat

	@property
	def replay_iter(self):
		if self._replay_iter is None:
			self._replay_iter = iter(self.replay_loader)
		return self._replay_iter

	def eval(self):
		step, episode, total_reward = 1, 0, 0
		eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)

		eval_step = 1
		while eval_until_episode(episode):
			observation = self.eval_env.reset()
			done = False
			#self.video_recorder.init(self.eval_env, enabled=(episode == 0))
			while not done and step%self.cfg.max_horizen!=0:
				with torch.no_grad(), utils.eval_mode(self.agent):
					action = self.agent.act(observation[self.swap_obs_type[self.cfg.obs_type]],
											self.global_step,
											eval_mode=True)
				observation, reward, done, _  = self.eval_env.step(action)

				#self.video_recorder.record(self.eval_env)
				total_reward += reward
				step += 1

			episode += 1
			#self.video_recorder.save(f'{self.global_frame}.mp4')

		
		with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
			log('episode_reward', total_reward / episode)
			log('episode_length', step * self.cfg.suite.action_repeat / episode)
			log('episode', self.global_episode)
			log('step', self.global_step)
			if repr(self.agent) != 'drqv2':
				log('expert_reward', self.expert_reward)


	def train_il(self):
		# predicates
		train_until_step = utils.Until(self.cfg.suite.num_train_frames,
									   self.cfg.suite.action_repeat)
		seed_until_step = utils.Until(self.cfg.suite.num_seed_frames,
									  self.cfg.suite.action_repeat)
		eval_every_step = utils.Every(self.cfg.suite.eval_every_frames,
									  self.cfg.suite.action_repeat)

		episode_step, episode_reward = 0, 0

		#time_steps = list()
		observations = list()
		actions = list()
		rewards = list()
		dones = list()

		observation= self.train_env.reset()
		#time_steps.append(time_step)
		done = False
		if repr(self.agent) == 'potil':
			if self.agent.auto_rew_scale:
				self.agent.sinkhorn_rew_scale = 1.  # Set after first episode

		#self.train_video_recorder.init(observation[self.swap_obs_type[self.cfg.obs_type]].T)
		self.train_video_recorder.init(observation[self.swap_obs_type[self.cfg.obs_type]])

		metrics = None
		for _ in tqdm(range(self.cfg.suite.num_train_frames)):
			if done or self._global_step%self.cfg.max_horizen==0:
				dones[-1] = True
				self._global_episode += 1
				if self._global_episode % 10 == 0:
					self.train_video_recorder.save(f'{self.global_frame}.mp4')
				# wait until all the metrics schema is populated
				observations = np.stack(observations, 0)
				actions = np.stack(actions, 0)
				if repr(self.agent) == 'potil':
					new_rewards = self.agent.ot_rewarder(
						observations, self.expert_demo, self.global_step)
					new_rewards_sum = np.sum(new_rewards)
				elif repr(self.agent) == 'dac':
					new_rewards = self.agent.dac_rewarder(observations, actions)
					new_rewards_sum = np.sum(new_rewards)
				
				if repr(self.agent) == 'potil':
					if self.agent.auto_rew_scale: 
						if self._global_episode == 1:
							self.agent.sinkhorn_rew_scale = self.agent.sinkhorn_rew_scale * self.agent.auto_rew_scale_factor / float(
								np.abs(new_rewards_sum))
							new_rewards = self.agent.ot_rewarder(
								observations, self.expert_demo, self.global_step)
							new_rewards_sum = np.sum(new_rewards)
                                    
				for i, elt in enumerate(observations):
					if repr(self.agent) == 'potil' or repr(self.agent) == 'dac':
							rewards[i]= new_rewards[i]

					dict_episode = {'observation':elt, 'action':actions[i], 'reward': rewards[i], 'discount':self.cfg.suite.discount, 'done' : dones[i]}
					self.replay_storage.add(dict_episode)

				if self._global_episode % 2 == 0 and len(self.replay_storage) > 0:
					# Update
					metrics = self.agent.update(self.replay_iter, self.expert_replay_iter, self.global_step,
												self.cfg.bc_regularize)
					self.logger.log_metrics(metrics, self.global_frame, ty='train')

				if metrics is not None:
					elapsed_time, total_time = self.timer.reset()
					episode_frame = episode_step * self.cfg.suite.action_repeat
					with self.logger.log_and_dump_ctx(self.global_frame,
													  ty='train') as log:
						log('fps', episode_frame / elapsed_time)
						log('total_time', total_time)
						log('episode_reward', episode_reward)
						log('episode_length', episode_frame)
						log('episode', self.global_episode)
						log('buffer_size', len(self.replay_storage))
						log('step', self.global_step)
						if repr(self.agent) == 'potil' or repr(self.agent) == 'dac':
								log('expert_reward', self.expert_reward)
								log('imitation_reward', new_rewards_sum)

				# eval
				if self._global_episode % 50 == 0:
					self.logger.log('eval_total_time', self.timer.total_time(),
									self.global_frame)
					self.eval()
				# reset env

				observations = list()
				actions = list()
				rewards = list()
				dones = list()

				observation = self.train_env.reset()

				#observations.append(time_step.observation[self.cfg.obs_type])
				#actions.append(time_step.action)

				#self.train_video_recorder.init(observation[self.swap_obs_type[self.cfg.obs_type]].T)
				self.train_video_recorder.init(observation[self.swap_obs_type[self.cfg.obs_type]])

				# try to save snapshot
				if self.cfg.suite.save_snapshot:
					self.save_snapshot()
				episode_step = 0
				episode_reward = 0

			# try to evaluate
			#observations.append(observation[self.swap_obs_type[self.cfg.obs_type]].T)
			observations.append(observation[self.swap_obs_type[self.cfg.obs_type]])

			# sample action
			with torch.no_grad(), utils.eval_mode(self.agent):
				action = self.agent.act(observation[self.swap_obs_type[self.cfg.obs_type]],
										self.global_step,
										eval_mode=False)
				#action = self.agent.act(observation[self.swap_obs_type[self.cfg.obs_type]].T,
				#						self.global_step,
				#						eval_mode=False)

			# take env step
			observation, reward, done, _ = self.train_env.step(action)
			episode_reward += reward

			rewards.append(reward)
			actions.append(action)
			dones.append(done)

			self.train_video_recorder.record(observation[self.swap_obs_type[self.cfg.obs_type]])
			episode_step += 1
			self._global_step += 1

	def save_snapshot(self):
		snapshot = self.work_dir / 'snapshot.pt'
		keys_to_save = ['timer', '_global_step', '_global_episode']
		payload = {k: self.__dict__[k] for k in keys_to_save}
		payload.update(self.agent.save_snapshot())
		with snapshot.open('wb') as f:
			torch.save(payload, f)

	def load_snapshot(self, snapshot):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot(agent_payload)

@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
	from train import WorkspaceIL as W
	root_dir = Path.cwd()
	workspace = W(cfg)
	
	# Load weights
	if cfg.load_bc:
		snapshot = Path(cfg.bc_weight)
		if snapshot.exists():
			print(f'resuming bc: {snapshot}')
			workspace.load_snapshot(snapshot)

	workspace.train_il()


if __name__ == '__main__':
	main()
