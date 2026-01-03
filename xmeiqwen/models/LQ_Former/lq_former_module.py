import torch
import torch.nn as nn
from transformers import BertTokenizerFast, BertConfig
from xmeiqwen.models.Qformer import BertLMHeadModel

class LQFormerModule(nn.Module):
    def __init__(self, num_hidden_layers=2, visual_width=768, cross_attention_freq=1):
        super().__init__()
        bert_config_path = "/datas/huggingface/bert-base-uncased" 
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_config_path)
        encoder_config = BertConfig.from_pretrained(bert_config_path)
        encoder_config.encoder_width = visual_width
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        
        self.num_semantic_tokens = 16 
        self.num_learnable_tokens = 16
        self.total_tokens = self.num_semantic_tokens + self.num_learnable_tokens 
        
        self.bert = BertLMHeadModel(config=encoder_config)

        self.bert.cls = None 

        self.learnable_query = nn.Parameter(
            torch.zeros(1, self.num_learnable_tokens, encoder_config.hidden_size)
        )
        self.learnable_query.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        self.output_proj = nn.Linear(encoder_config.hidden_size, encoder_config.hidden_size)

    def forward(self, modality_features, captions):
        """
        Args:
            modality_features: [batch, time_steps, dim] (Visual or Audio features)
            captions: List[str] - Modality specific captions
        Returns:
            enhanced_features: [batch, 32, dim]
            attention_mask: [batch, 32]
        """
        device = modality_features.device
        batch_size = modality_features.size(0)
        text_inputs = self.tokenizer(
            captions, 
            padding="max_length",
            truncation=True, 
            max_length=self.num_semantic_tokens, 
            return_tensors="pt"
        ).to(device)

        semantic_embeds = self.bert.bert.embeddings(
            input_ids=text_inputs.input_ids,
            token_type_ids=text_inputs.token_type_ids
        )

        learnable_embeds = self.learnable_query.expand(batch_size, -1, -1)
        query_embeds = torch.cat([semantic_embeds, learnable_embeds], dim=1)
        semantic_mask = text_inputs.attention_mask # [batch, 16]
        learnable_mask = torch.ones(batch_size, self.num_learnable_tokens).to(device) # [batch, 16]
        
        query_attention_mask = torch.cat([semantic_mask, learnable_mask], dim=1)
        encoder_attention_mask = torch.ones(modality_features.size()[:-1], dtype=torch.long).to(device)

        output = self.bert.bert(
            query_embeds=query_embeds,             
            attention_mask=query_attention_mask,    
            encoder_hidden_states=modality_features,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
    
        enhanced_features = output.last_hidden_state
        enhanced_features = self.output_proj(enhanced_features)
        
        return enhanced_features, query_attention_mask