import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import random

# Pose86K layout:
# RH: 0:21, LH: 21:42, LIPS: 42:42+19, BODY: rest
NUM_LIPS = 19


class PoseDatasetV2(Dataset):

    def temporal_drop(self, pose_data: np.ndarray, max_dp=0.15):
        """
        Drop ONE contiguous segment with length in [0, max_dp*T].
        pose_data: (T, J, D)
        """
        T = pose_data.shape[0]
        if T <= 2:
            return pose_data

        dp_len = int(T * max_dp * np.random.random())
        if dp_len <= 0:
            return pose_data

        # keep at least 1 frame
        dp_len = min(dp_len, T - 1)
        start = np.random.randint(0, T - dp_len + 1)
        end = start + dp_len
        idx = np.concatenate([np.arange(0, start), np.arange(end, T)], axis=0)
        return pose_data[idx]

    def jitter(self, x: np.ndarray, std=0.01):
        """Gaussian jitter on normalized coordinates (same scale as ablation)."""
        if std <= 0:
            return x
        noise = np.random.normal(0.0, std, size=x.shape).astype(np.float32)
        return (x + noise).astype(np.float32)

    def maybe_flip(self, rh, lh, lips, body, p=0.0):
        """
        Optional horizontal flip + swap hands.
        Default p=0.0 (OFF) because left/right can be semantic in CSLR.
        Flip assumes normalized x around 0 (your normalize outputs ~[-0.5,0.5]).
        """
        if np.random.random() >= p:
            return rh, lh, lips, body

        rh_f = rh.copy()
        lh_f = lh.copy()
        lips_f = lips.copy()
        body_f = body.copy()

        # flip x
        rh_f[..., 0] *= -1
        lh_f[..., 0] *= -1
        lips_f[..., 0] *= -1
        body_f[..., 0] *= -1

        # swap hands
        return lh_f, rh_f, lips_f, body_f

    # -----------------------------
    # Normalization (keep your style)
    # -----------------------------
    def _normalize_part(self, pose: np.ndarray):
        """
        pose: (K, 2)
        output ~ [-0.5, 0.5]
        """
        pose = pose.astype(np.float32)

        # center by joint0
        pose = pose - pose[0:1]

        # shift to positive
        pose = pose - pose.min(axis=0, keepdims=True)

        # scale to 0..1 box
        max_vals = pose.max(axis=0)
        denom = float(max(max_vals[0], max_vals[1], 1e-6))
        pose = pose / denom

        # zero-mean + absmax scale
        pose = pose - pose.mean()
        absmax = float(np.max(np.abs(pose)) + 1e-6)
        pose = (pose / absmax) * 0.5
        return pose.astype(np.float32)

    # -----------------------------
    # Init / load
    # -----------------------------
    def __init__(
        self,
        dataset_name2,
        label_csv,
        split_type,
        target_enc_df,
        transform=None,
        augmentations=True,
        augmentations_prob=0.5,
        additional_joints=True,
        mode="SI",
        max_frames=1000,
        # ablation params
        jitter_std=0.01,
        temporal_drop_max=0.15,
        p_jitter=1.0,          # apply jitter whenever aug=True
        p_temporal_drop=1.0,   # apply temporal drop whenever aug=True
        p_flip=0.0,            # OFF by default
    ):
        self.dataset_name = dataset_name2
        self.split_type = split_type
        self.transform = transform

        self.augmentations = augmentations and (split_type == "train")
        self.augmentations_prob = augmentations_prob

        self.additional_joints = additional_joints
        self.mode = mode
        self.max_frames = max_frames

        # ablation-inspired aug configs
        self.jitter_std = float(jitter_std)
        self.temporal_drop_max = float(temporal_drop_max)
        self.p_jitter = float(p_jitter)
        self.p_temporal_drop = float(p_temporal_drop)
        self.p_flip = float(p_flip)

        # load pose pkl
        pkl_path = f"./data/pose_data_isharah2000_hands_lips_body_phase2_{mode}.pkl"
        assert os.path.exists(pkl_path), f"Pose data file not found: {pkl_path}"
        with open(pkl_path, "rb") as f:
            self.pose_dict = pickle.load(f)

        self.files = []
        self.labels = []

        self.all_data = pd.read_csv(label_csv, delimiter="|")
        if "isharah" in self.dataset_name:
            self.all_data = self.all_data[self.all_data["id"].notna()]
            self.all_data = self.all_data[self.all_data["gloss"].notna()]

        for _, row in self.all_data.iterrows():
            sample_id = str(row["id"])
            enc_label = target_enc_df[target_enc_df["id"] == sample_id]["enc"]
            if (not enc_label.empty) and (sample_id in self.pose_dict):
                self.files.append(sample_id)
                self.labels.append(enc_label.iloc[0])

        print(f"Loaded {len(self.files)} samples for split: {split_type} | aug={self.augmentations}")

    def __len__(self):
        return len(self.files)

    def get_file_path(self, idx):
        return self.files[idx]

    # -----------------------------
    # Main readPose
    # -----------------------------
    def readPose(self, sample_id):
        pose_data = self.pose_dict[sample_id]["keypoints"]
        if pose_data is None or pose_data.shape[0] == 0:
            raise ValueError(f"Empty pose for {sample_id}")

        # crop for speed
        pose_data = pose_data[: self.max_frames]
        T, J, D = pose_data.shape

        # clip-level augmentation switch
        do_aug = self.augmentations and (np.random.random() < self.augmentations_prob)

        if do_aug and (np.random.random() < self.p_temporal_drop):
            pose_data = self.temporal_drop(pose_data, max_dp=self.temporal_drop_max)
            T = pose_data.shape[0]

        # split parts (use xy only)
        pose_xy = pose_data[:, :, :2] if pose_data.shape[-1] > 2 else pose_data
        right_hand = pose_xy[:, 0:21, :]
        left_hand = pose_xy[:, 21:42, :]
        lips = pose_xy[:, 42:42 + NUM_LIPS, :]
        body = pose_xy[:, 42 + NUM_LIPS:, :]  # whatever remains

        # per-frame normalization + fill missing
        right_joints, left_joints, face_joints, body_joints = [], [], [], []
        for t in range(T):
            rh = right_hand[t]
            lh = left_hand[t]
            fc = lips[t]
            bd = body[t]

            # fill missing (all zeros)
            if rh.sum() == 0:
                rh = right_joints[-1] if t > 0 else np.zeros((21, 2), np.float32)
            else:
                rh = self._normalize_part(rh)

            if lh.sum() == 0:
                lh = left_joints[-1] if t > 0 else np.zeros((21, 2), np.float32)
            else:
                lh = self._normalize_part(lh)

            if fc.sum() == 0:
                fc = face_joints[-1] if t > 0 else np.zeros((NUM_LIPS, 2), np.float32)
            else:
                fc = self._normalize_part(fc)

            if bd.sum() == 0:
                bd = body_joints[-1] if t > 0 else np.zeros((len(bd), 2), np.float32)
            else:
                bd = self._normalize_part(bd)

            right_joints.append(rh)
            left_joints.append(lh)
            face_joints.append(fc)
            body_joints.append(bd)

        # backward fill for early missing
        for t in range(len(left_joints) - 2, -1, -1):
            if np.sum(left_joints[t]) == 0:
                left_joints[t] = left_joints[t + 1].copy()
        for t in range(len(right_joints) - 2, -1, -1):
            if np.sum(right_joints[t]) == 0:
                right_joints[t] = right_joints[t + 1].copy()

        rh = np.stack(right_joints, axis=0)   # (T,21,2)
        lh = np.stack(left_joints, axis=0)    # (T,21,2)
        fc = np.stack(face_joints, axis=0)    # (T,19,2)
        bd = np.stack(body_joints, axis=0)    # (T,?,2)

        # optional flip
        if do_aug and self.p_flip > 0:
            rh, lh, fc, bd = self.maybe_flip(rh, lh, fc, bd, p=self.p_flip)

        # jitter after normalize (ablation)
        if do_aug and (np.random.random() < self.p_jitter):
            rh = self.jitter(rh, std=self.jitter_std)
            lh = self.jitter(lh, std=self.jitter_std)
            fc = self.jitter(fc, std=self.jitter_std)
            bd = self.jitter(bd, std=self.jitter_std)

        # concat
        out = np.concatenate([rh, lh], axis=1)  # (T,42,2)
        if self.additional_joints:
            out = np.concatenate([out, fc, bd], axis=1)  # (T,86,2) if bd=25 joints

        return out.astype(np.float32)

    def pad_or_crop_sequence(self, sequence, min_len=32, max_len=1000):
        T, J, D = sequence.shape
        if T < min_len:
            pad = np.zeros((min_len - T, J, D), dtype=sequence.dtype)
            sequence = np.concatenate([sequence, pad], axis=0)
        if sequence.shape[0] > max_len:
            sequence = sequence[:max_len]
        return sequence

    def __getitem__(self, idx):
        sample_id = self.files[idx]
        pose = self.readPose(sample_id)
        pose = self.pad_or_crop_sequence(pose, min_len=32, max_len=self.max_frames)
        pose = torch.from_numpy(pose).float()

        if self.transform:
            pose = self.transform(pose)

        label = torch.as_tensor(self.labels[idx])
        return sample_id, pose, label