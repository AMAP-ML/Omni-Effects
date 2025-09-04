import os
import gc
import json
import random
import numpy as np
from contextlib import contextmanager
from func_timeout import func_timeout, FunctionTimedOut
import cv2

import torch
from torch.utils.data import BatchSampler
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import crop, center_crop, resize, hflip
import torch.nn.functional as F

from decord import VideoReader


VIDEO_READER_TIMEOUT = 20


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


def get_video_reader_batch(video_reader, indices):
    frames = video_reader.get_batch(indices).asnumpy()
    return frames


def resize_frames(
    frames, 
    cond_frames,
    height, 
    width, 
    video_reshape_mode="center"
):
    image_size = height, width
    reshape_mode = video_reshape_mode
    if frames.shape[3] / frames.shape[2] > image_size[1] / image_size[0]:
        frames = resize(
            frames,
            size=[image_size[0], int(frames.shape[3] * image_size[0] / frames.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
        if len(cond_frames) > 0:
            cond_frames = [
                resize(
                    cond_frame,
                    size=[image_size[0], int(cond_frame.shape[3] * image_size[0] / cond_frame.shape[2])],
                    interpolation=InterpolationMode.BICUBIC,
                ) for cond_frame in cond_frames]
    else:
        frames = resize(
            frames,
            size=[int(frames.shape[2] * image_size[1] / frames.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )
        if len(cond_frames) > 0:
            cond_frames= [
                resize(
                    cond_frame,
                    size=[int(cond_frame.shape[2] * image_size[1] / cond_frame.shape[3]), image_size[1]],
                    interpolation=InterpolationMode.BICUBIC,
                ) for cond_frame in cond_frames]

    h, w = frames.shape[2], frames.shape[3]
    frames = frames.squeeze(0)

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    frames = crop(frames, top=top, left=left, height=image_size[0], width=image_size[1])
    if len(cond_frames) > 0:
        cond_frames = [crop(cond_frame, top=top, left=left, height=image_size[0], width=image_size[1]) for cond_frame in cond_frames]
    return frames, cond_frames


def random_horizontal_crop_and_flip(video, mask, freeze_prob=0.2):
    
    coords = torch.nonzero(mask[0][0], as_tuple=False)
    min_width, max_width = coords[:, 1].min(), coords[:, 1].max()
    min_height, max_height = coords[:, 0].min(), coords[:, 0].max()

    min_height_ = (max_height + min_height) // 2 - max((max_height - min_height) // 2 + 100, 240)
    min_height_ = max(min_height_, 0)
    max_height_ = (max_height + min_height) // 2 + max((max_height - min_height) // 2 + 100, 240)
    max_height_ = min(max_height_, video.shape[2])

    min_width_ = (max_width + min_width) // 2 - max((max_width - min_width) // 2 + 100, 180)
    min_width_ = max(min_width_, 0)
    max_width_ = (max_width + min_width) // 2 + max((max_width - min_width) // 2 + 100, 180)
    max_width_ = min(max_width_, video.shape[3])

    cropped_video = video[:, :, min_height_:max_height_, min_width_:max_width_]
    cropped_mask = mask[:, :, min_height_:max_height_, min_width_:max_width_]

    if random.random() > 0.5:  # 50% chance to flip
        cropped_video = torch.flip(cropped_video, dims=[3])
        cropped_mask = torch.flip(cropped_mask, dims=[3])

    # With probability freeze_prob, freeze video (use first frame) and zero mask
    if random.random() < freeze_prob:
        # Repeat first frame for all frames
        first_frame = cropped_video[0:1].clone()
        cropped_video = first_frame.repeat(cropped_video.shape[0], 1, 1, 1)
        # Zero out mask
        cropped_mask = torch.zeros_like(cropped_mask)
    
    return cropped_video, cropped_mask

def data_augmentation_multi(video_list, 
                            mask_list, 
                            target_height=480,
                            random_probs=[0.2, 0.5, 0.8, 0.9]):
    """
    Args:
        video_list: List of videos (each tensor of shape [f, c, h, w])
        mask_list: List of mask videos (each tensor of shape [f, c, h_m, w_m])
        target_height: Target height for resizing
    
    Returns:
        combined_video: Combined video tensor [f, c, H, W_total]
        processed_masks: List of processed mask tensors [f, c, H, W_total] each
    """
    assert len(video_list) == len(mask_list)
 
    if random.random() < random_probs[0]:
        return video_list[0], [mask_list[0]] * len(mask_list)

    cropped_videos = []
    cropped_masks = []
    for video, mask in zip(video_list, mask_list):
        # Random crop video and mask at same relative position
        f, c, h, w = video.shape
        video_crop, mask_crop = random_horizontal_crop_and_flip(video, mask)
        
        # Resize to target height (keeping aspect ratio)
        # Video
        f, c, h, w = video_crop.shape
        scale = target_height / h
        new_width = int(w * scale)
        resized_video = F.interpolate(video_crop, size=(target_height, new_width), 
                                     mode='bilinear', align_corners=False)
        cropped_videos.append(resized_video)
        
        # Mask
        f, c, h_m, w_m = mask_crop.shape
        scale = target_height / h_m
        new_width = int(w_m * scale)
        resized_mask = F.interpolate(mask_crop, size=(target_height, new_width), 
                                   mode='nearest')  # Use nearest for masks
        cropped_masks.append(resized_mask)
    
    # 2. Horizontally concatenate the cropped and resized videos
    combined_video = torch.cat(cropped_videos, dim=3)  # Concatenate along width
    f, c, H, W_total = combined_video.shape
    
    # 3. Create full-size masks with values only in corresponding regions
    processed_masks = []
    current_width = 0
    
    for mask in cropped_masks:
        _, _, H_mask, W_mask = mask.shape
        
        # Create a zero tensor with combined_video dimensions
        full_mask = torch.zeros_like(combined_video)
        
        # Place the mask in the correct position
        full_mask[:, :, :, current_width:current_width+W_mask] = mask
        
        processed_masks.append(full_mask)
        current_width += W_mask  # Move to next video's position
    
    return combined_video, processed_masks


def data_augmentation(
    frames, 
    cond_frames,
    random_probs=[0.2, 0.5, 0.8, 0.9]
):
    random_value = random.random()
    if random_value < random_probs[0]:
        return frames, cond_frames
    else:
        coords = torch.nonzero(cond_frames[0][0], as_tuple=False)
        min_width, max_width = coords[:, 1].min(), coords[:, 1].max()
        min_height, max_height = coords[:, 0].min(), coords[:, 0].max()

        min_height_ = (max_height + min_height) // 2 - max((max_height - min_height) // 2 + 100, 240)
        min_height_ = max(min_height_, 0)
        max_height_ = (max_height + min_height) // 2 + max((max_height - min_height) // 2 + 100, 240)
        max_height_ = min(max_height_, frames.shape[2])

        min_width_ = (max_width + min_width) // 2 - max((max_width - min_width) // 2 + 100, 180)
        min_width_ = max(min_width_, 0)
        max_width_ = (max_width + min_width) // 2 + max((max_width - min_width) // 2 + 100, 180)
        max_width_ = min(max_width_, frames.shape[3])

        anchor_frame = frames[0][:, min_height_:max_height_, min_width_:max_width_]
        mirrored_anchor_frame = hflip(anchor_frame)
        new_frames, new_cond_frames = [], []
        for i in range(frames.shape[0]):
            frame = frames[i][:, min_height_:max_height_, min_width_:max_width_]
            cond_frame = cond_frames[i][:, min_height_:max_height_, min_width_:max_width_]
            mirrored_frame = hflip(frame)
            mirrored_cond_frame = hflip(cond_frame)
            if random_probs[0] < random_value < random_probs[1]:
                frame = torch.cat([frame, mirrored_anchor_frame], dim=2)
                cond_frame = torch.cat([cond_frame, torch.zeros_like(mirrored_cond_frame)], dim=2)
            elif random_probs[1] < random_value < random_probs[2]:
                frame = torch.cat([anchor_frame, mirrored_frame], dim=2)
                cond_frame = torch.cat([torch.zeros_like(cond_frame), mirrored_cond_frame], dim=2)
            elif random_probs[2] < random_value < random_probs[3]:
                frame = torch.cat([frame, mirrored_frame], dim=2)
                cond_frame = torch.cat([cond_frame, mirrored_cond_frame], dim=2)
            else:
                frame = torch.cat([anchor_frame, mirrored_anchor_frame], dim=2)
                cond_frame = torch.cat([torch.zeros_like(cond_frame), torch.zeros_like(mirrored_cond_frame)], dim=2)
            new_frames.append(frame)
            new_cond_frames.append(cond_frame)
        new_frames = torch.stack(new_frames, dim=0)
        new_cond_frames = torch.stack(new_cond_frames, dim=0)
        return new_frames, new_cond_frames


class VideoDataset(Dataset):
    def __init__(
        self,
        ann_paths, data_root=None,
        video_sample_size=512,
        video_sample_n_frames=16,
        text_drop_ratio=-1,
        video_length_drop_start=0.0,
        video_length_drop_end=1.0,
        random_crop_prob=0.0,
        num_cond=1,
        multi_concat_prob=0.5
    ):
        self.data_root = data_root

        self.dataset = []

        for ann_path in ann_paths:
            # Loading annotations from files
            print(f"loading annotations from {ann_path} ...")
            dataset = json.load(open(ann_path))

            for data in dataset:
                if data.get('type', 'image') == 'video':
                    self.dataset.append(data)
            del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.text_drop_ratio = text_drop_ratio

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        self.video_sample_n_frames = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self.random_crop_prob = random_crop_prob
        self.num_cond = num_cond
        self.multi_concat_prob = multi_concat_prob

    def random_subset_mask(self, original_mask):
        coords = torch.nonzero(original_mask, as_tuple=False)
        min_height, max_height = coords[:, 0].min(), min(coords[:, 0].max(), self.video_sample_size[0] - self.video_sample_size[0] // 10)
        min_width, max_width = coords[:, 1].min(), coords[:, 1].max()
        
        rect_length = 100

        min_height_ = random.randint(min_height, max_height)
        min_width_ = random.randint(min_width, max_width)
        max_height_ = min(min_height_ + rect_length, original_mask.size(0))
        max_width_ = min(min_width_ + rect_length, original_mask.size(1))

        new_mask = torch.zeros_like(original_mask)
        if random.random() < 0.5:
            new_mask[min_height:min_height_, min_width:max_width] = 1
        else:
            if random.random() < 0.5:
                new_mask[min_height:min_height_, min_width:min_width_] = 1
            else:
                new_mask[min_height:min_height_, min_width_:max_width] = 1
        
        new_mask = new_mask
        
        return new_mask

    def get_batch(self, data_info):
        num_videos = len(data_info)
        video_path_list = []
        text_list = []
        cond_image_list = []
        for i in range(num_videos):
            if data_info[i].get('type', 'image') == 'video':
                if self.num_cond == 0:
                    video_path, text = data_info[i]['file_path'], data_info[i]['text']
                    video_path_list.append(video_path)
                    text_list.append(text)
                elif self.num_cond >= 1:
                    video_path, text, cond_image = data_info[i]['file_path'], data_info[i]['text'], data_info[i]['cond_image']
                    video_path_list.append(video_path)
                    text_list.append(text)
                    cond_image_list.append(cond_image)
            else:
                raise ValueError(f"Not support image input.")

        if self.data_root is None:
            video_path_list = video_path_list
        else:
            video_path_list = [os.path.join(self.data_root, video_path) for video_path in video_path_list]
        
        frames_list = []
        for video_path in video_path_list:
            with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
                if len(video_reader) == 0:
                    raise ValueError(f"No Frames in video")

                indices = np.linspace(0, len(video_reader) - 1, self.video_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, indices)
                    frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    frames = frames[:self.video_sample_n_frames]
                    selected_num_frames = frames.shape[0]

                    # Choose first (4k + 1) frames as this is how many is required by the VAE
                    remainder = (3 + (selected_num_frames % 4)) % 4
                    if remainder != 0:
                        frames = frames[:-remainder]
                    selected_num_frames = frames.shape[0]

                    assert (selected_num_frames - 1) % 4 == 0

                    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            del video_reader

            frames_list.append(frames)
            
        cond_frames_list = []
        for cond_image in cond_image_list:
            with VideoReader_contextmanager(cond_image, num_threads=2) as video_reader:
                try:
                    indices = np.linspace(0, self.video_sample_n_frames, self.video_sample_n_frames + 1, dtype=int)
                    sample_args = (video_reader, indices)
                    cond_frames = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    )
                    cond_frames = cond_frames[:self.video_sample_n_frames]
                    selected_num_frames = cond_frames.shape[0]

                    # Choose first (4k + 1) frames as this is how many is required by the VAE
                    remainder = (3 + (selected_num_frames % 4)) % 4
                    if remainder != 0:
                        frames = frames[:-remainder]
                    selected_num_frames = cond_frames.shape[0]

                    assert (selected_num_frames - 1) % 4 == 0

                    cond_frames = torch.from_numpy(cond_frames).permute(0, 3, 1, 2).contiguous()
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            del video_reader
            cond_frames_list.append(cond_frames)
        
        if self.num_cond == 1:
            frames, cond_frames = data_augmentation(
                frames_list[0], 
                cond_frames_list[0]
            )
            cond_frames_list = [cond_frames]
        elif self.num_cond > 1:
            frames, cond_frames_list = data_augmentation_multi(
                frames_list, 
                cond_frames_list,
                target_height=self.video_sample_size[0]
            )
        elif self.num_cond == 0:
            pass
        else:
            raise ValueError(f"Not support num_cond == {self.num_cond}")

        pixel_values, cond_pixel_values = resize_frames(
            frames, cond_frames_list, 
            self.video_sample_size[0], self.video_sample_size[1],
            video_reshape_mode='random'
        )
        pixel_values = pixel_values / 255.

        cond_pixel_values = [cond_pixel_value / 255. for cond_pixel_value in cond_pixel_values]

        cond_pixel_values = [cond_pixel_value[0].unsqueeze(0) for cond_pixel_value in cond_pixel_values]
            
        pixel_values = self.video_transforms(pixel_values)   
        cond_pixel_values = [ self.video_transforms(cond_pixel_value) for cond_pixel_value in cond_pixel_values]

        if random.random() < self.text_drop_ratio:
            text_list = ["" for _ in text_list]
        return pixel_values, cond_pixel_values, text_list, "video"
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'image')

        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'image')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")
                
                data_info_list = []
                data_info_list.append(data_info_local)
                for _ in range(self.num_cond - 1):
                    if self.multi_concat_prob > random.random():
                        data_info_list.append(data_info_local)
                    else:
                        data_info_list.append(self.dataset[random.randint(0, len(self.dataset)) % len(self.dataset)])
                
                pixel_values, cond_pixel_values, text, data_type = self.get_batch(data_info_list)
                sample['pixel_values'] = pixel_values
                sample['cond_pixel_values'] = torch.cat(cond_pixel_values, dim=0) if len(cond_pixel_values) > 0 else None
                sample['text'] = text
                sample['data_type'] = data_type
                sample['idx'] = idx

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length - 1)
        
        return sample


def collate_fn(examples):
    pixel_values = []
    ref_pixel_values = []
    cond_pixel_values = []
    for example in examples:
        video = example["pixel_values"].unsqueeze(0).permute(0, 2, 1, 3, 4)
        if example["cond_pixel_values"] is not None:
            cond_video = example["cond_pixel_values"].unsqueeze(0).permute(0, 2, 1, 3, 4)
        
        image = video[:, :, :1]

        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=image.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=image.dtype)
        noisy_image = image + torch.randn_like(image) * image_noise_sigma[:, None, None, None, None]

        pixel_values.append(video)
        if example["cond_pixel_values"] is not None:
            cond_pixel_values.append(cond_video)
        ref_pixel_values.append(noisy_image)
        
    pixel_values = torch.cat(pixel_values, dim=0)
    if example["cond_pixel_values"] is not None:
        cond_pixel_values = torch.cat(cond_pixel_values, dim=0)
    ref_pixel_values = torch.cat(ref_pixel_values, dim=0)
    
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    if example["cond_pixel_values"] is not None:
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    ref_pixel_values = ref_pixel_values.to(memory_format=torch.contiguous_format).float()
    
    captions = example['text']
    return {
        'pixel_values': pixel_values,
        'cond_pixel_values': cond_pixel_values if example["cond_pixel_values"] is not None else None,
        'ref_pixel_values': ref_pixel_values,
        'captions': captions
    }
