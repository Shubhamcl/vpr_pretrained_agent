from asyncio import futures
import copy
import logging
from pathlib import Path
from time import time

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T

import sys
sys.path.insert(0, '/home/shubham/Desktop/git/carla-roach/')

import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util
import carla
from augmentation import hard

logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)


class CilrsDataset(Dataset):
    def __init__(self, list_expert_h5, im_augmenter=None, 
        wide_image=False, lb_mode=False):

        self._im_augmenter = im_augmenter
        self._batch_read_number = 0
        self._im_stack_idx = [-1] # env_wrapper.im_stack_idx
        self.lb_mode = lb_mode

        # Normalizing image function
        im_mean = [0.485, 0.456, 0.406]
        im_std = [0.229, 0.224, 0.225]
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=im_mean, std=im_std)])

        self.image_key = ['small_central_rgb']
        if wide_image:
            self.image_key = ['wide_small_rgb']

        self._obs_keys_to_load = ['speed', 'gnss'] + self.image_key

        # Some setting(s)
        self.speed_factor = 12.0

        # Data holders
        self._obs_list = []
        self._supervision_list = []

        # log.info(f'Loading H5 Files.')
        self.expert_frames = self._load_h5(list_expert_h5)
        log.info(f" Loaded {len(list_expert_h5)} files.")

    def _load_h5(self, list_h5):
        n_frames = 0

        for h5_path in list_h5:

            with h5py.File(h5_path, 'r', libver='latest', swmr=True) as hf:
                total_steps = len(hf.keys())
                for step_str, group_step in hf.items():
                    
                    if group_step.attrs['critical']:
                        current_step = int(step_str.split('_')[-1])
                        im_stack_idx_list = [max(0, current_step+i+1) for i in self._im_stack_idx]# [current_step]

                        # Load the obs as per list
                        obs_dict = {}
                        for obs_key in self._obs_keys_to_load:
                            # For image, attach the address and other attributes, rest attach numpy value
                            if 'rgb' in obs_key:
                                obs_dict[obs_key] = [self.read_group_to_dict(group_step['obs'][obs_key],
                                        [h5_path, f'step_{i}', 'obs', obs_key]) for i in im_stack_idx_list]
                                # group_step['obs'][obs_key]
                            else:
                                obs_dict[obs_key] = self.read_group_to_dict(group_step['obs'][obs_key],
                                                                            [h5_path, step_str, 'obs', obs_key])

                        supervision_dict = self.read_group_to_dict(group_step['supervision'],
                                                                   [h5_path, step_str, 'supervision'])
                        self._obs_list.append(obs_dict)
                        self._supervision_list.append(self.process_supervision(supervision_dict))
                        n_frames += 1
        return n_frames

    @staticmethod
    def read_group_to_dict(group, list_keys):
        data_dict = {}
        for k, v in group.items():
            if v.size > 5000:
                data_dict[k] = list_keys
            else:
                data_dict[k] = np.array(v)
        return data_dict

    def __len__(self):
        return len(self._obs_list)

    def __getitem__(self, idx):
        # Note: Future is loaded in else case, and by default

        obs = copy.deepcopy(self._obs_list[idx])
        # load images/large dataset here
        for obs_key, obs_dict in obs.items():
            if 'rgb' in obs_key:
                for i in range(len(obs_dict)):
                    for k, v in obs_dict[i].items():
                        if type(v) is list:
                            with h5py.File(v[0], 'r', libver='latest', swmr=True) as hf:
                                group = hf
                                for key in v[1:]:
                                    group = group[key]
                                obs[obs_key][i][k] = np.array(group[k])
            elif 'pretrain_action' in obs_key or 'mid_fusion' in obs_key:
                obs[obs_key] = np.array(obs[obs_key])
            else:
                for k, v in obs_dict.items():
                    if type(v) is list:
                        with h5py.File(v[0], 'r', libver='latest', swmr=True) as hf:
                            group = hf
                            for key in v[1:]:
                                group = group[key]
                            if k == 'future':
                                obs[obs_key] = np.array(group)
                            else:
                                obs[obs_key][k] = np.array(group[k])

        supervision = copy.deepcopy(self._supervision_list[idx])

        if self._im_augmenter is not None:
            for obs_key, obs_dict in obs.items():
                if 'rgb' in obs_key:
                    for i in range(len(obs_dict)):
                        obs[obs_key][i]['data'] = self._im_augmenter(
                            self._batch_read_number).augment_image(obs[obs_key][i]['data'])

        policy_input, command = self.process_obs(obs)

        self._batch_read_number += 1

        return command, policy_input, supervision

    def process_obs(self, obs):
        # Prepare Command
        # VOID = -1
        # LEFT = 1
        # RIGHT = 2
        # STRAIGHT = 3
        # LANEFOLLOW = 4
        # CHANGELANELEFT = 5
        # CHANGELANERIGHT = 6

        command = obs['gnss']['command'][0]
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]

        # Make state [Speed only, as default]
        state_list = []
        state_list.append(obs['speed']['forward_speed'][0]/self.speed_factor)

        # Leaderboard needs more states
        if self.lb_mode:
            ev_gps = obs['gnss']['gnss']
            # imu nan bug
            compass = 0.0 if np.isnan(obs['gnss']['imu'][-1]) else obs['gnss']['imu'][-1]

            gps_point = obs['gnss']['target_gps']
            target_vec_in_global = gps_util.gps_to_location(gps_point) - gps_util.gps_to_location(ev_gps)
            ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
            loc_in_ev = trans_utils.vec_global_to_ref(target_vec_in_global, ref_rot_in_global)

            # Append vec
            state_list.append(loc_in_ev.x)
            state_list.append(loc_in_ev.y)
            # Append cmd
            cmd_one_hot = [0] * 6
            cmd_one_hot[command] = 1
            state_list += cmd_one_hot

        # Make a list for image
        im_list = []
        im = self._im_transform(obs[self.image_key[0]][0]['data'])
        im_list.append(im)

        policy_input = {
            'im': torch.squeeze(torch.stack(im_list, dim=1)),
            'state': torch.tensor(state_list, dtype=torch.float32)
        }
            
        return policy_input, torch.tensor([command], dtype=torch.int8)

    def process_supervision(self, supervision):
        '''
        supervision['speed']: in m/s
        supervision['action']: throttle, steer, brake in [0,1]
        '''
        processed_supervision = {}

        # [[ Action ]]
        # deterministic action [ Made true due to settings in ROACH ]
        if True:
            throttle, steer, brake = supervision['action']
            if brake > 0:
                acc = -1.0 * brake
            else:
                acc = throttle
            action = np.array([acc, steer], dtype=np.float32)
        else:
            # Continuous action?
            action = supervision['action']
        processed_supervision['action'] = torch.tensor(action)

        # [[ States (or just speed) ]]
        processed_supervision['speed'] = torch.tensor(supervision['speed'])/self.speed_factor

        return processed_supervision


def get_dataloader(dataset_dir, dagger_dirs=[], batch_size=32, 
wide_image=False, lb_mode=False, sequence_penalty=False,
                   aug=False, num_workers=14):
    now = time()

    def make_dataset(list_expert_h5, aug=False):
        augmenter = None
        if aug:
            augmenter = hard

        # Deleted augmentation flag, hence is_train gone to waste
        dataset = CilrsDataset(list_expert_h5, im_augmenter=augmenter, wide_image=wide_image, lb_mode=lb_mode)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                shuffle=True, drop_last=True, pin_memory=False)

        return dataloader

    dataset_path = Path(dataset_dir)

    # Glob, sort
    list_expert_h5 = list(dataset_path.glob('*.h5'))
    list_expert_h5 = sorted(list_expert_h5, key=lambda x: int(x.name.split('.')[0]))

    # Train, Val
    list_expert_h5_train = [x for i, x in enumerate(list_expert_h5) if i % 10 != 0]
    list_expert_h5_val = [x for i, x in enumerate(list_expert_h5) if i % 10 == 0]

    if len(dagger_dirs) > 0:
        for dagger_dir in dagger_dirs:

            dagger_dir = Path(dagger_dir)
            # Glob, sort
            list_dagger_h5 = list(dagger_dir.glob('*.h5'))
            list_dagger_h5 = sorted(list_dagger_h5, key=lambda x: int(x.name.split('.')[0]))

            # Train, Val
            list_dagger_train = [x for i, x in enumerate(list_dagger_h5) if i % 10 != 0]
            list_dagger_val = [x for i, x in enumerate(list_dagger_h5) if i % 10 == 0]

            list_expert_h5_train += list_dagger_train
            list_expert_h5_val += list_dagger_val

    log.info(f'Loading training dataset')
    train = make_dataset(list_expert_h5_train, is_train=True, sequence_penalty=sequence_penalty, aug=aug)
    log.info(f'Loading validation dataset')
    val = make_dataset(list_expert_h5_val, False, False)

    log.info(f"Dataset loaded in {time()-now:.2f} seconds.")
    return train, val
