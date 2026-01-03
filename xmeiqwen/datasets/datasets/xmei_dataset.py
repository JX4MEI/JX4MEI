import os
import json
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from xmeiqwen.datasets.datasets.base_dataset import BaseDataset
from xmeiqwen.processors.video_processor import load_video
from xmeiqwen.models.ImageBind.data import load_audio, transform_audio 
import config

class XMEI_Dataset(BaseDataset):
    def __init__(self, vis_processor=None, txt_processor=None, img_processor=None,
                 dataset_cfg=None, model_cfg=None, split="train"):
        self.dataset = 'XMEI'
        self.split = split

        self.annotation_path = dataset_cfg.build_info.annotation.get(split)
        self.vis_root = dataset_cfg.build_info.storage 
        self.wav_root = dataset_cfg.build_info.storage 
        
        self.video_placeholder = "<Video><ImageHere></Video>" 
        self.audio_placeholder = "<Audio><AudioHere></Audio>"
        self.num_video_query_token = 32
        self.num_audio_query_token = 32
        
        print(f"[{self.split}] Loading annotations from: {self.annotation_path}")
        with open(self.annotation_path, 'r', encoding='utf-8') as f:
            self.annotation = json.load(f)
            
        self.instruction_template = (
            "Identify the speaker's emotion and intent based on the video, audio, and subtitles. "
            "Then, provide a reasoning explaining why they co-occur."
        )
        
        super().__init__(vis_processor=vis_processor, 
                         txt_processor=txt_processor,
                         img_processor=img_processor,
                         vis_root=self.vis_root,
                         wav_root=self.wav_root,
                         model_cfg=model_cfg,
                         dataset_cfg=dataset_cfg)
    
    def __len__(self):
        return len(self.annotation)

    def _get_video_path(self, sample):
        return os.path.join(self.vis_root, sample.get('video_path', sample['video_id'] + '.mp4'))

    def _get_audio_path(self, sample):
        return os.path.join(self.wav_root, sample.get('audio_path', sample['video_id'] + '.wav'))

    def __getitem__(self, index):
        sample = self.annotation[index]
        
        # ================= Video Loading =================
        video_path = self._get_video_path(sample)
        try:
            frames = load_video(
                video_path, 
                n_frms=self.vis_processor.n_frms, 
                height=self.vis_processor.image_size, 
                width=self.vis_processor.image_size,
                sampling="uniform"
            ) 
            video_tensor = self.vis_processor(frames)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            video_tensor = torch.zeros(3, self.vis_processor.n_frms, self.vis_processor.image_size, self.vis_processor.image_size)
            frames = [Image.new('RGB', (224, 224)) for _ in range(self.vis_processor.n_frms)]

        audio_path = self._get_audio_path(sample)
        try:
            audio_tensor = load_audio(audio_path)
            audio_tensor = transform_audio(audio_tensor)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            audio_tensor = torch.zeros(1, 1, 8, 1024) 

        video_desc = sample.get('video_description', "")
        audio_desc = sample.get('audio_description', "")
        subtitle = sample.get('subtitle', "")
        emotion = sample.get('emotion', "neutral")
        intent = sample.get('intent', "neutral")
        explanation = sample.get('explanation', "")
        
        video_tokens = "<Video>" + "<ImageHere>" * self.num_video_query_token + "</Video>"
        audio_tokens = "<Audio>" + "<AudioHere>" * self.num_audio_query_token + "</Audio>"
        
        prompt = (
            f"{video_tokens}{audio_tokens} "
            f"The subtitle of this video is \"{subtitle}\". "
            f"{self.instruction_template}"
        )
        
        # 2. 构建 Answer
        target = f"The user's emotion is {emotion}, intent is {intent}. The underlying reason is: {explanation}."
    
        
        processed_text = self.process_llm_text(prompt, target, self.txt_processor.tokenizer)

        return {
            "frames": video_tensor,       
            "raw_frames": frames,        
            "audios": audio_tensor,       
            "raw_audios": audio_path,     
            "video_captions": video_desc, 
            "audio_captions": audio_desc, 
            
            "input_ids": processed_text['input_ids'],
            "labels": processed_text['labels'],
            "attention_masks": processed_text['attention_mask'],
            
            "image_id": sample.get('video_id', str(index)),
            "face_or_frame": self.dataset_cfg.get("face_or_frame", "multiframe") 
        }

    def process_llm_text(self, prompt, target, tokenizer, max_length=1024):
        full_text = prompt + " " + target + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=True).input_ids[0]

        input_ids = tokenizer(full_text, return_tensors='pt', add_special_tokens=True, 
                              padding='max_length', truncation=True, max_length=max_length).input_ids[0]

        labels = input_ids.clone()

        prompt_len = min(len(prompt_ids), max_length)
        labels[:prompt_len] = -100

        labels[input_ids == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id)
        }