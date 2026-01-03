import copy
import einops
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

from xmeiqwen.common.registry import registry
from xmeiqwen.models.blip2 import Blip2Base
from xmeiqwen.models.tokenizer import load_tokenizer_from_LLM
from xmeiqwen.models.LQ_Former.lq_former_module import LQFormerModule 
import config

@registry.register_model("xmeiqwen")
class XMEIQwen(Blip2Base):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/xmeiqwen.yaml",
    }

    def __init__(
        self,
        visual_encoder_name,
        acoustic_encoder_name,
        llama_model_name,
        frozen_video_proj,
        frozen_video_Qformer,
        frozen_audio_Qformer,
        frozen_audio_proj,
        frozen_llm,
        lora_r,
        num_video_query_token,
        num_audio_query_token,
        num_image_query_token,
    ):
        super().__init__()

        # ==================== 1. Load LLM (Qwen2.5) ====================
        print(f'====== Loading LLM: {llama_model_name} ======')
        self.llama_model_name = llama_model_name    
        self.llama_tokenizer = load_tokenizer_from_LLM(llama_model_name)
        
        # 定义 Special Tokens
        DEFAULT_IMAGE_PATCH_TOKEN = config.DEFAULT_IMAGE_PATCH_TOKEN
        DEFAULT_AUDIO_PATCH_TOKEN = config.DEFAULT_AUDIO_PATCH_TOKEN
        DEFAULT_FRAME_PATCH_TOKEN = config.DEFAULT_FRAME_PATCH_TOKEN
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]
        self.FRAME_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_FRAME_PATCH_TOKEN]

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            config.PATH_TO_LLM[llama_model_name],
            torch_dtype=torch.float16
        )

        # 冻结原始 LLM 参数
        for param in self.llama_model.parameters():
            param.requires_grad = False

        # ==================== 2. Apply LoRA ====================
        print(f'====== Using LoRA on LLM (r={lora_r}) ======')
        
        layer_num = len(self.llama_model.model.layers)
        target_modules=['model.layers.'+str(i)+'.'+ k for i in range(layer_num) for k in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj","mlp.down_proj","mlp.up_proj"]]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=lora_r, 
            lora_alpha=32, 
            lora_dropout=0.05, 
            target_modules=target_modules
        )
        self.llama_model = get_peft_model(self.llama_model, peft_config)
        
        if frozen_llm:
            for param in self.llama_model.parameters(): param.requires_grad = False
            print('freeze: LLAMA Model (including LoRA)')
        else:
            print('trainable: LLAMA Model (LoRA parameters)')
        self.llama_model.print_trainable_parameters()

        # ==================== 3. Visual Branch (CLIP + LQ-Former) ====================
        print('====== Loading Visual Encoder (CLIP) ======')
        self.visual_encoder = registry.get_visual_encoder_class(visual_encoder_name)()
        self.num_video_query_token = num_video_query_token
        
        self.video_frame_position_embedding = nn.Embedding(32, self.visual_encoder.hidden_size)

        print('====== Loading Video LQ-Former ======')
        self.video_Qformer = LQFormerModule(
            num_hidden_layers=2,
            visual_width=self.visual_encoder.hidden_size,
            cross_attention_freq=1
        )

        # 冻结/解冻逻辑
        if frozen_video_Qformer:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            print('freeze: Video LQ-Former')
        else:
            print('trainable: Video LQ-Former')
        video_hidden_size = 768 
        self.video_llama_proj = nn.Linear(video_hidden_size, self.llama_model.config.hidden_size)
        
        if frozen_video_proj:
            for param in self.video_llama_proj.parameters(): param.requires_grad = False
        
        # 纯图处理 (Optional, if used)
        self.image_llama_proj = nn.Linear(self.visual_encoder.hidden_size, self.llama_model.config.hidden_size)
        self.num_image_query_token = num_image_query_token

        # ==================== 4. Audio Branch (HUBERT + LQ-Former) ====================
        print(f'====== Loading Audio Encoder (HUBERT) ======')
        self.acoustic_encoder = registry.get_acoustic_encoder_class(acoustic_encoder_name)()
        self.num_audio_query_token = num_audio_query_token

        self.audio_position_embedding = nn.Embedding(8, self.acoustic_encoder.hidden_size)

        print('====== Loading Audio LQ-Former ======')
        self.audio_Qformer = LQFormerModule(
            num_hidden_layers=2,
            visual_width=self.acoustic_encoder.hidden_size,
            cross_attention_freq=1
        )

        if frozen_audio_Qformer:
            for name, param in self.audio_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.audio_position_embedding.named_parameters():
                param.requires_grad = False
            print('freeze: Audio LQ-Former')
        else:
            print('trainable: Audio LQ-Former')

        # Projector
        audio_hidden_size = 768
        self.audio_llama_proj = nn.Linear(audio_hidden_size, self.llama_model.config.hidden_size)
        
        if frozen_audio_proj:
            for param in self.audio_llama_proj.parameters(): param.requires_grad = False

    def encode_video_lqformer(self, video, raw_video, captions=None):
        device = video.device
        with self.maybe_autocast():
            # 1. Visual Features [B, T, Q, H] or [B, T, H]
            frame_hidden_state = self.visual_encoder(video, raw_video).to(device)
            batch_size, time_length = frame_hidden_state.size()[:2]

            if len(frame_hidden_state.size()) == 4:
                # [b, t, q, h] -> [b, (t q), h]
                frame_hidden_state = einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h')
            
            # 3. LQ-Former Forward
            if captions is None:
                captions = [""] * batch_size
            
            video_hidden, _ = self.video_Qformer(frame_hidden_state, captions)
            inputs_llama = self.video_llama_proj(video_hidden)

        return inputs_llama

    def encode_audio_lqformer(self, audio, raw_audio, captions=None):
        device = audio.device
        with self.maybe_autocast():
            # 1. Audio Features
            audio_hidden_state = self.acoustic_encoder(audio, raw_audio).to(device)
            batch_size, time_length = audio_hidden_state.size()[:2]

            # 2. Position Embeddings
            position_ids = torch.arange(time_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
            audio_pos = self.audio_position_embedding(position_ids)
            audio_hidden_state = audio_hidden_state + audio_pos

            # 3. LQ-Former Forward
            if captions is None:
                captions = [""] * batch_size

            audio_hidden, _ = self.audio_Qformer(audio_hidden_state, captions)

            # 4. Project
            inputs_llama = self.audio_llama_proj(audio_hidden)
    
        return inputs_llama

    def encode_image_token(self, image, raw_image):
        device = image.device
        with self.maybe_autocast():
            frame_hidden_state = self.visual_encoder(image, raw_image).to(device)
            if len(frame_hidden_state.size()) == 4:
                frame_hidden_state = frame_hidden_state.flatten(1, 2)
            elif len(frame_hidden_state.size()) == 3:
                frame_hidden_state = frame_hidden_state.expand(-1, self.num_image_query_token, -1)
            inputs_llama = self.image_llama_proj(frame_hidden_state)
        return None, inputs_llama

    def forward(self, samples):
        video_captions = samples.get('video_captions', None) 
        audio_captions = samples.get('audio_captions', None)

        frame_llms, audio_llms, image_llms = None, None, None
        
        # 1. Video
        if 'frames' in samples: 
            frame_llms = self.encode_video_lqformer(
                samples['frames'],  
                samples['raw_frames'], 
                captions=video_captions
            )
        
        # 2. Audio
        if 'audios' in samples: 
            audio_llms = self.encode_audio_lqformer(
                samples['audios'],  
                samples['raw_audios'],
                captions=audio_captions
            )
        
        # 3. Image (if present)
        if 'images' in samples: 
            _, image_llms = self.encode_image_token(samples['images'], samples['raw_images'])
        
        # 4. LLM Input Construction
        input_ids = samples['input_ids']
        temp_input_ids = copy.deepcopy(input_ids)
        # Mask special tokens
        temp_input_ids[temp_input_ids == self.FRAME_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.AUDIO_PATCH_TOKEN_ID] = 0
        temp_input_ids[temp_input_ids == self.IMAGE_PATCH_TOKEN_ID] = 0
        
        temp_input_embedding = self.llama_model.model.model.embed_tokens(temp_input_ids)

        cur_idx = 0
        new_input_embeds = []
        
        # 替换逻辑
        for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
            for (patch_token_id, query_token_number, embeds) in [
                (self.FRAME_PATCH_TOKEN_ID, self.num_video_query_token, frame_llms),
                (self.AUDIO_PATCH_TOKEN_ID, self.num_audio_query_token, audio_llms),
                (self.IMAGE_PATCH_TOKEN_ID, self.num_image_query_token, image_llms)
            ]:
                if (cur_input_ids == patch_token_id).sum() != 0:
                    if embeds is None: continue 
                    
                    cur_features = embeds[cur_idx]
                    
                    # 确保维度匹配 (LQ-Former 输出长度可能变化，这里假设对齐或裁剪)
                    num_patches = (cur_input_ids == patch_token_id).sum().item()
                    if cur_features.shape[0] != num_patches:
                         # 简单的对齐保护
                         if cur_features.shape[0] > num_patches:
                             cur_features = cur_features[:num_patches]
                         else:
                             # 若特征不足，需要 Padding 或报错，这里暂时报错以提示检查数据
                             raise ValueError(f"Feature dim {cur_features.shape[0]} != Token dim {num_patches}")

                    masked_indices = torch.where(cur_input_ids == patch_token_id)[0]
                    mask_index_start = masked_indices[0]
                    
                    cur_input_embeds = torch.cat((
                        cur_input_embeds[:mask_index_start], 
                        cur_features, 
                        cur_input_embeds[mask_index_start+num_patches:]
                    ), dim=0)
            
            new_input_embeds.append(cur_input_embeds)
            cur_idx += 1
            
        inputs_embeds = torch.stack(new_input_embeds, dim=0)
        
        targets = samples['labels']
        attention_mask = samples['attention_masks']
        
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets
            )
        
        # 返回 loss
        return {"loss": outputs.loss}

    @classmethod
    def from_config(cls, cfg):
        
        visual_encoder_name   = cfg.get("visual_encoder", "CLIP_VIT_LARGE") 
        acoustic_encoder_name = cfg.get("acoustic_encoder", "HUBERT_LARGE")
        llama_model_name      = cfg.get("llama_model", "Qwen25") 
        
        frozen_video_Qformer    = cfg.get("frozen_video_Qformer", False)
        frozen_video_proj       = cfg.get("frozen_video_proj", False)
        frozen_audio_Qformer    = cfg.get("frozen_audio_Qformer", False)
        frozen_audio_proj       = cfg.get("frozen_audio_proj", False)
        frozen_llm              = cfg.get("frozen_llm", False)
        lora_r                  = cfg.get("lora_r", 16)

        num_audio_query_token = cfg.get("num_audio_query_token", 32)
        num_video_query_token = cfg.get("num_video_query_token", 32)
        num_image_query_token = cfg.get("num_image_query_token", 32)

        model = cls(
            visual_encoder_name=visual_encoder_name,
            acoustic_encoder_name=acoustic_encoder_name,
            llama_model_name=llama_model_name,
            frozen_video_proj=frozen_video_proj,
            frozen_audio_proj=frozen_audio_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            frozen_llm=frozen_llm,
            lora_r=lora_r,
            num_video_query_token=num_video_query_token,
            num_audio_query_token=num_audio_query_token,
            num_image_query_token=num_image_query_token,
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load Checkpoint for Secondary Training: {}".format(ckpt_path))
            # weights_only=True 更安全
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print("Checkpoint Loaded. Missing keys (expected for new modules):", msg.missing_keys)
            
        return model