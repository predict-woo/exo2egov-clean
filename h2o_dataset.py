"""
H2O Dataset for Exo2Ego training.
Supports both single exocentric view (replicated 4 times) and multi-view format (4 actual views).
"""
import os
import json
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class H2ODataset(Dataset):
    """
    H2O Dataset for Exo2Ego training.
    
    Supports two formats:
    1. Single exo view: {"exo": "path/to/exo.png", "ego": "path/to/ego.png"}
       - The single exo view is replicated 4 times
    2. Multi exo views: {"exo": ["path/cam0.png", "path/cam1.png", "path/cam2.png", "path/cam3.png"], "ego": "path/to/ego.png"}
       - Each exo camera has its own intrinsics and extrinsics
    """
    
    def __init__(self, args, sample_frames=1):
        """
        Args:
            args: Argument namespace containing dataset configuration
            sample_frames: Number of frames to sample per sequence
        """
        self.root = args.h2o_root if hasattr(args, 'h2o_root') else "/cluster/project/cvg/data/H2O"
        self.sample_frames = sample_frames
        self.exo_width = args.exo_width
        self.exo_height = args.exo_height
        self.ego_width = args.ego_width
        self.ego_height = args.ego_height
        
        # Load split file (JSONL format, despite .json extension)
        split_path = args.train_dict
        with open(split_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        
        # Detect format: multi-view if exo is a list
        self.multi_view = isinstance(self.samples[0]['exo'], list) if self.samples else False
        
        # Group samples by sequence for multi-frame sampling
        self.sequences = self._group_sequences()
        
        # Cache for intrinsics (loaded once per camera)
        self._intrinsics_cache = {}
        
        format_str = "multi-view (4 cameras)" if self.multi_view else "single-view (replicated)"
        print(f"H2ODataset initialized with {len(self.samples)} samples in {len(self.sequences)} sequences [{format_str}]")
    
    def _group_sequences(self):
        """
        Group samples by sequence (subject/action/take).
        Returns list of lists, where each inner list contains sample indices.
        """
        sequences = {}
        for idx, sample in enumerate(self.samples):
            # Extract sequence key: subject1/h1/0
            parts = sample['ego'].split('/')
            seq_key = '/'.join(parts[:3])
            if seq_key not in sequences:
                sequences[seq_key] = []
            sequences[seq_key].append(idx)
        
        # Sort indices within each sequence by frame number
        for seq_key in sequences:
            sequences[seq_key].sort(key=lambda i: int(self.samples[i]['ego'].split('/')[-1].replace('.png', '')))
        
        return list(sequences.values())
    
    def _load_intrinsics(self, cam_dir):
        """
        Load camera intrinsics from cam_intrinsics.txt.
        Uses caching to avoid repeated file reads.
        """
        if cam_dir in self._intrinsics_cache:
            return self._intrinsics_cache[cam_dir]
        
        intrinsics_path = os.path.join(cam_dir, "cam_intrinsics.txt")
        with open(intrinsics_path, 'r') as f:
            values = f.read().strip().split()
        
        fx, fy, cx, cy, width, height = map(float, values)
        intrinsics = {
            'focal': torch.tensor([fx, fy], dtype=torch.float32),
            'c': torch.tensor([cx, cy], dtype=torch.float32),
            'width': int(width),
            'height': int(height)
        }
        
        self._intrinsics_cache[cam_dir] = intrinsics
        return intrinsics
    
    def _load_extrinsics(self, cam_dir, frame_id):
        """
        Load 4x4 camera-to-world extrinsic matrix.
        """
        extrinsics_path = os.path.join(cam_dir, "cam_pose", f"{frame_id}.txt")
        with open(extrinsics_path, 'r') as f:
            values = list(map(float, f.read().strip().split()))
        return torch.tensor(values, dtype=torch.float32).reshape(4, 4)
    
    def _parse_image_path(self, rel_path):
        """
        Extract camera directory and frame ID from relative image path.
        
        Args:
            rel_path: e.g., "subject1/h1/0/cam2/rgb/000000.png"
        
        Returns:
            cam_dir: Full path to camera directory
            frame_id: Frame identifier string (e.g., "000000")
        """
        parts = rel_path.split('/')
        cam_dir = os.path.join(self.root, *parts[:-2])  # subject1/h1/0/cam2
        frame_id = parts[-1].replace('.png', '')  # 000000
        return cam_dir, frame_id
    
    def _load_image(self, path, width, height):
        """
        Load image, resize, and normalize to [-1, 1].
        """
        with Image.open(path) as img:
            img_resized = img.resize((width, height), Image.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img_resized)).float()
            # Normalize: [0, 255] -> [-1, 1]
            img_normalized = img_tensor / 127.5 - 1
            return img_normalized.permute(2, 0, 1)  # (3, H, W)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns dictionary matching EgoExo4Ddataset format:
            - pixel_values: (F, 3, ego_H, ego_W) target ego images
            - ego_images: (F, 3, ego_H//8, ego_W//8) downsampled ego images
            - condition_values: list of 4 tensors, each (F, 3, exo_H, exo_W)
            - ego_poses: (F, 4, 4) ego camera poses
            - exo_poses: (4, 4, 4) exo camera poses (one per camera)
            - focal: (2,) focal lengths [fx, fy]
            - c: (2,) principal point [cx, cy]
            - rel_poses: (F, 4, 4, 4) relative poses
            - scene: str sequence identifier
            - pixel_files, condition_files, ego_files: file paths
            - idx: sample index
        """
        # Get sequence and select frames
        seq_indices = self.sequences[idx % len(self.sequences)]
        
        # Select consecutive frames
        if len(seq_indices) < self.sample_frames:
            # Pad by repeating last frame if sequence too short
            selected_indices = seq_indices.copy()
            while len(selected_indices) < self.sample_frames:
                selected_indices.append(selected_indices[-1])
        else:
            start_idx = random.randint(0, len(seq_indices) - self.sample_frames)
            selected_indices = seq_indices[start_idx:start_idx + self.sample_frames]
        
        # Initialize tensors
        pixel_values = torch.empty((self.sample_frames, 3, self.ego_height, self.ego_width))
        ego_images = torch.empty((self.sample_frames, 3, self.ego_height // 8, self.ego_width // 8))
        condition_values = [
            torch.empty((self.sample_frames, 3, self.exo_height, self.exo_width)) 
            for _ in range(4)
        ]
        ego_poses = torch.empty((self.sample_frames, 4, 4))
        exo_poses_all = torch.empty((self.sample_frames, 4, 4, 4))  # (F, 4 cams, 4, 4)
        
        pixel_files = []
        condition_files = []
        
        for i, sample_idx in enumerate(selected_indices):
            sample = self.samples[sample_idx]
            
            # Load ego image (target)
            ego_path = os.path.join(self.root, sample['ego'])
            pixel_values[i] = self._load_image(ego_path, self.ego_width, self.ego_height)
            ego_images[i] = self._load_image(ego_path, self.ego_width // 8, self.ego_height // 8)
            pixel_files.append(ego_path)
            
            # Load ego pose
            ego_cam_dir, ego_frame = self._parse_image_path(sample['ego'])
            ego_poses[i] = self._load_extrinsics(ego_cam_dir, ego_frame)
            
            if self.multi_view:
                # Multi-view: load 4 different exo images with their own poses
                exo_paths = sample['exo']  # List of 4 paths
                frame_exo_files = []
                for j, exo_rel_path in enumerate(exo_paths):
                    exo_path = os.path.join(self.root, exo_rel_path)
                    exo_img = self._load_image(exo_path, self.exo_width, self.exo_height)
                    condition_values[j][i] = exo_img
                    frame_exo_files.append(exo_path)
                    
                    # Load extrinsics for this specific exo camera
                    exo_cam_dir, exo_frame = self._parse_image_path(exo_rel_path)
                    exo_poses_all[i, j] = self._load_extrinsics(exo_cam_dir, exo_frame)
                
                condition_files.append(frame_exo_files)
            else:
                # Single-view: replicate to 4 cameras
                exo_path = os.path.join(self.root, sample['exo'])
                exo_img = self._load_image(exo_path, self.exo_width, self.exo_height)
                
                # Replicate exo image to all 4 "cameras"
                for j in range(4):
                    condition_values[j][i] = exo_img
                condition_files.append(exo_path)
                
                # Load exo pose and replicate
                exo_cam_dir, exo_frame = self._parse_image_path(sample['exo'])
                exo_pose = self._load_extrinsics(exo_cam_dir, exo_frame)
                for j in range(4):
                    exo_poses_all[i, j] = exo_pose
        
        # Get intrinsics from first frame's first exo camera
        first_sample = self.samples[selected_indices[0]]
        if self.multi_view:
            exo_cam_dir, _ = self._parse_image_path(first_sample['exo'][0])
        else:
            exo_cam_dir, _ = self._parse_image_path(first_sample['exo'])
        intrinsics = self._load_intrinsics(exo_cam_dir)
        
        # Scale intrinsics for resized images
        scale_x = self.exo_width / intrinsics['width']
        scale_y = self.exo_height / intrinsics['height']
        focal = intrinsics['focal'].clone()
        focal[0] *= scale_x
        focal[1] *= scale_y
        c = intrinsics['c'].clone()
        c[0] *= scale_x
        c[1] *= scale_y
        
        # Use first frame's exo poses for the "static" exo cameras
        exo_poses = exo_poses_all[0]  # (4, 4, 4)
        
        # Apply coordinate convention conversion (to NeRF convention)
        # Original: x-right, y-down, z-forward
        # NeRF: x-right, y-up, z-back
        ego_poses_conv = ego_poses.clone()
        exo_poses_conv = exo_poses.clone()
        ego_poses_conv[:, :, 1:3] *= -1
        exo_poses_conv[:, :, 1:3] *= -1
        
        # Compute relative poses: inv(ego) @ exo for each frame
        # Result: (F, 4, 4, 4) - one relative pose per exo camera per frame
        ego_poses_inv = torch.linalg.inv(ego_poses_conv)  # (F, 4, 4)
        rel_poses = torch.matmul(
            ego_poses_inv.unsqueeze(1),  # (F, 1, 4, 4)
            exo_poses_conv.unsqueeze(0)  # (1, 4, 4, 4)
        )  # Result: (F, 4, 4, 4)
        
        # Extract scene name for compatibility
        scene = '/'.join(first_sample['ego'].split('/')[:3])
        
        return {
            'pixel_values': pixel_values,           # (F, 3, ego_H, ego_W)
            'pixel_files': pixel_files,
            'condition_values': condition_values,   # list of 4 tensors, each (F, 3, exo_H, exo_W)
            'condition_files': condition_files,
            'ego_poses': ego_poses_conv,            # (F, 4, 4) - with coord conversion
            'exo_poses': exo_poses_conv,            # (4, 4, 4) - with coord conversion
            'scene': scene,
            'focal': focal,                         # (2,)
            'c': c,                                 # (2,)
            'ego_images': ego_images,               # (F, 3, ego_H//8, ego_W//8)
            'ego_files': pixel_files,
            'idx': idx,
            'rel_poses': rel_poses.squeeze(),       # (4, 4, 4) for F=1, or (F, 4, 4, 4) for F>1
        }
    
    def get_validation(self, args, accelerator):
        """
        Return validation data in expected format for train_stage1.py.
        
        Returns a dictionary structured for the validation loop.
        """
        # Pick a random sequence for validation
        seq_idx = random.randint(0, len(self.sequences) - 1)
        data = self.__getitem__(seq_idx)
        
        # Structure for validation: list format expected by train_stage1.py
        all_validation_values_ego = [
            data['pixel_values'].unsqueeze(0).to(accelerator.device)
        ]
        
        all_validation_values_exo = [
            [cv.unsqueeze(0).to(accelerator.device) for cv in data['condition_values']]
        ]
        
        all_validation_values = {
            'ego': all_validation_values_ego,
            'exo': all_validation_values_exo,
        }
        
        return {
            'all_validation_values': all_validation_values,
            'validation_files': data['pixel_files'],
            'ego_poses': data['ego_poses'].unsqueeze(0),       # (1, F, 4, 4)
            'exo_poses': data['exo_poses'],                    # (4, 4, 4)
            'scene': data['scene'],
            'focal': data['focal'],
            'c': data['c'],
            'ego_images': data['ego_images'],                   # (F, 3, H, W)
            'rel_poses': data['rel_poses'],                    # (F, 4, 4, 4)
        }


class H2ODatasetTest(Dataset):
    """
    H2O Dataset for testing/inference.
    Similar to H2ODataset but loads from test split.
    Supports both single-view and multi-view formats.
    """
    
    def __init__(self, args, sample_frames=1):
        self.root = args.h2o_root if hasattr(args, 'h2o_root') else "/cluster/project/cvg/data/H2O"
        self.sample_frames = sample_frames
        self.exo_width = args.exo_width
        self.exo_height = args.exo_height
        self.ego_width = args.ego_width
        self.ego_height = args.ego_height
        
        # Load test split
        split_path = args.test_dict
        with open(split_path, 'r') as f:
            self.samples = [json.loads(line) for line in f]
        
        # Detect format: multi-view if exo is a list
        self.multi_view = isinstance(self.samples[0]['exo'], list) if self.samples else False
        
        self._intrinsics_cache = {}
        format_str = "multi-view (4 cameras)" if self.multi_view else "single-view (replicated)"
        print(f"H2ODatasetTest initialized with {len(self.samples)} samples [{format_str}]")
    
    def _load_intrinsics(self, cam_dir):
        if cam_dir in self._intrinsics_cache:
            return self._intrinsics_cache[cam_dir]
        
        intrinsics_path = os.path.join(cam_dir, "cam_intrinsics.txt")
        with open(intrinsics_path, 'r') as f:
            values = f.read().strip().split()
        
        fx, fy, cx, cy, width, height = map(float, values)
        intrinsics = {
            'focal': torch.tensor([fx, fy], dtype=torch.float32),
            'c': torch.tensor([cx, cy], dtype=torch.float32),
            'width': int(width),
            'height': int(height)
        }
        self._intrinsics_cache[cam_dir] = intrinsics
        return intrinsics
    
    def _load_extrinsics(self, cam_dir, frame_id):
        extrinsics_path = os.path.join(cam_dir, "cam_pose", f"{frame_id}.txt")
        with open(extrinsics_path, 'r') as f:
            values = list(map(float, f.read().strip().split()))
        return torch.tensor(values, dtype=torch.float32).reshape(4, 4)
    
    def _parse_image_path(self, rel_path):
        parts = rel_path.split('/')
        cam_dir = os.path.join(self.root, *parts[:-2])
        frame_id = parts[-1].replace('.png', '')
        return cam_dir, frame_id
    
    def _load_image(self, path, width, height):
        with Image.open(path) as img:
            img_resized = img.resize((width, height), Image.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img_resized)).float()
            img_normalized = img_tensor / 127.5 - 1
            return img_normalized.permute(2, 0, 1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single test sample."""
        sample = self.samples[idx]
        
        # Load ego image
        ego_path = os.path.join(self.root, sample['ego'])
        pixel_values = self._load_image(ego_path, self.ego_width, self.ego_height).unsqueeze(0)
        ego_images = self._load_image(ego_path, self.ego_width // 8, self.ego_height // 8).unsqueeze(0)
        
        # Load ego pose
        ego_cam_dir, ego_frame = self._parse_image_path(sample['ego'])
        ego_pose = self._load_extrinsics(ego_cam_dir, ego_frame).unsqueeze(0)
        
        if self.multi_view:
            # Multi-view: load 4 different exo images with their own poses
            exo_paths = sample['exo']  # List of 4 paths
            condition_values = []
            exo_poses = torch.empty((4, 4, 4))
            condition_files = []
            
            for j, exo_rel_path in enumerate(exo_paths):
                exo_path = os.path.join(self.root, exo_rel_path)
                exo_img = self._load_image(exo_path, self.exo_width, self.exo_height)
                condition_values.append(exo_img.unsqueeze(0))
                condition_files.append(exo_path)
                
                # Load extrinsics for this specific exo camera
                exo_cam_dir, exo_frame = self._parse_image_path(exo_rel_path)
                exo_poses[j] = self._load_extrinsics(exo_cam_dir, exo_frame)
            
            # Get intrinsics from first exo camera
            first_exo_cam_dir, _ = self._parse_image_path(exo_paths[0])
            intrinsics = self._load_intrinsics(first_exo_cam_dir)
        else:
            # Single-view: replicate to 4 cameras
            exo_path = os.path.join(self.root, sample['exo'])
            exo_img = self._load_image(exo_path, self.exo_width, self.exo_height)
            condition_values = [exo_img.unsqueeze(0) for _ in range(4)]
            condition_files = [exo_path]
            
            # Load exo pose and replicate
            exo_cam_dir, exo_frame = self._parse_image_path(sample['exo'])
            exo_pose = self._load_extrinsics(exo_cam_dir, exo_frame)
            exo_poses = exo_pose.unsqueeze(0).repeat(4, 1, 1)
            
            intrinsics = self._load_intrinsics(exo_cam_dir)
        
        # Scale intrinsics for resized images
        scale_x = self.exo_width / intrinsics['width']
        scale_y = self.exo_height / intrinsics['height']
        focal = intrinsics['focal'].clone()
        focal[0] *= scale_x
        focal[1] *= scale_y
        c = intrinsics['c'].clone()
        c[0] *= scale_x
        c[1] *= scale_y
        
        # Coordinate conversion
        ego_pose[:, :, 1:3] *= -1
        exo_poses[:, :, 1:3] *= -1
        
        # Relative poses to all 4 exo cameras
        # Result: (1, 4, 4, 4) which squeezes to (4, 4, 4) for single frame
        rel_poses = torch.matmul(
            torch.linalg.inv(ego_pose).unsqueeze(1),  # (1, 1, 4, 4)
            exo_poses.unsqueeze(0)                     # (1, 4, 4, 4)
        )  # Result: (1, 4, 4, 4)
        
        scene = '/'.join(sample['ego'].split('/')[:3])
        
        return {
            'pixel_values': pixel_values,
            'pixel_files': [ego_path],
            'condition_values': condition_values,
            'condition_files': condition_files,
            'ego_poses': ego_pose,
            'exo_poses': exo_poses,
            'scene': scene,
            'focal': focal,
            'c': c,
            'ego_images': ego_images,
            'ego_files': [ego_path],
            'idx': idx,
            'rel_poses': rel_poses.squeeze(),  # (4, 4, 4) for single frame
        }
