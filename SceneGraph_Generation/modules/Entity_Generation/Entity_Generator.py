import torch.nn as nn
import torch
from SceneGraph_Generation.modules.Entity_Generation.Foundation_Encoders import CLIPEncoder
import torch.nn.functional as F
from peft import get_peft_model

class Feature_Fuser(nn.Module):
    def __init__(self, vector_dim, cnnmodel, vitmodel):
        super(Feature_Fuser, self).__init__()
        self.vector_dim = vector_dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.original_encoder = CLIPEncoder(vitmodel)
        self.entity_encoder = CLIPEncoder(cnnmodel)
        # cnn dim: 1024
        # vit dim: 512
        
        self.tau = 1
        
    def resize_2_n(self, x, n: int): # x: [batch, 2 * (vector_dim // 8 + 3 + 3 + 64) + 512]
        x = x.unsqueeze(1)
        x_resized = F.interpolate(
            x,
            size=n,
            mode='linear',
            align_corners=False
        )
        return x_resized.squeeze(1) # [batch, n]
    
    def forward(self,
                entity_features, # [batch, vector_dim // 8 + 3 + 3 + 64, entity_height, entity_width]
                entity_originals, # [batch, 3, entity_height, entity_width]
                originals # [batch, 3, height, width]
                ):
        entity_features = torch.cat([
                                    self.avg_pool(entity_features),
                                    self.max_pool(entity_features)
                                    ], dim=1)
        entity_features = entity_features.flatten(1) # [batch, 2 * (vector_dim // 8 + 3 + 3 + 64)]
        entity_originals = self.entity_encoder(entity_originals) # [batch, 1024]
        originals = self.original_encoder(originals) # [batch, 512]
        entity_features = torch.cat([entity_features, entity_originals], dim=1) # [batch, 2 * (vector_dim // 8 + 3 + 3 + 64) + 1024]
        unfused_entity_features = entity_features.clone() # [batch, 2 * (vector_dim // 8 + 3 + 3 + 64) + 1024]
        unfused_entity_features = F.normalize(unfused_entity_features, dim=-1)
        entity_features = self.resize_2_n(entity_features, 512) # [batch, 512]
        
        # features fusion from global and local in paper: ConceptFusion
        entity_features = F.normalize(entity_features, dim=-1) # [batch, 512]
        originals = F.normalize(originals, dim=-1) # [batch, 512]
        sim_matrix = entity_features @ entity_features.T
        batch_size = sim_matrix.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix[~mask].view(batch_size, batch_size - 1)
        patch_sim = sim_matrix.mean(dim=-1) # [batch]
        global_sim = (originals * entity_features).sum(dim=-1) # [batch]
        w = (patch_sim + global_sim) / self.tau # [batch]
        weights = F.softmax(w, dim=0).unsqueeze(-1) # [batch, 1]
        entity_features = weights * originals + (1 - weights) * entity_features # [batch, 512]
        entity_features = torch.concat([entity_features, unfused_entity_features], dim=-1) # [batch, 2 * (vector_dim // 8 + 3 + 3 + 64) + 512 + 1024]  fused features
        return entity_features

class Entity_Generator(nn.Module):
    def __init__(self, vector_dim, ocrmodel, lora_config):
        super(Entity_Generator, self).__init__()
        self.has_lora = False
        self.vector_dim = vector_dim
        self.feature_projection_head = nn.Sequential(
            nn.Linear(2 * (vector_dim // 8 + 3 + 3 + 64) + 512 + 1024, 1536)
        )
        self.language_model = ocrmodel.model.language_model
        if lora_config is not None:
            self.language_model = get_peft_model(self.language_model, lora_config)
            self.has_lora = True
        dtype = self.language_model.dtype
        self.lm_head = ocrmodel.lm_head
        self.lm_head.requires_grad_(False) # the head is frozen all the time
        
        self.feature_projection_head = self.feature_projection_head.to(dtype=dtype)
        self.offset_head = nn.Sequential(
            nn.LayerNorm(2 * (vector_dim // 8 + 3 + 3 + 64) + 512 + 1024),
            nn.Linear(2 * (vector_dim // 8 + 3 + 3 + 64) + 512 + 1024, 2)
        )
        
        self.inferring = False # Inference mode
        
    def set_train_stage(self, stage = "stage1"):
        if stage == "stage1":
            for param in self.language_model.parameters():
                param.requires_grad = False
            for param in self.feature_projection_head.parameters():
                param.requires_grad = True
        else:
            for param in self.feature_projection_head.parameters():
                param.requires_grad = False
            for name, param in self.language_model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
    def forward(self,
                fused_entity_features, # [batch, 2 * (vector_dim // 8 + 3 + 3 + 64) + 512 + 1024]
                input_ids = None, # [batch, seq_len]
                attention_mask = None, # [batch, seq_len]
                ):
        # relation is True: [batch, 2, ...]
        # relation is False: [batch, ...]
        device = fused_entity_features.device
        lm_dtype = self.language_model.dtype
        entity_features = fused_entity_features # [batch, 2 * (vector_dim // 8 + 3 + 3 + 64) + 512 + 1024]
        
        if not self.inferring: # training and testing mode
            # training mode
            offsets = self.offset_head(entity_features) # [batch, 2]
            features_embed = self.feature_projection_head(entity_features.to(lm_dtype)) # [batch, 1536]
            features_embed = features_embed.unsqueeze(1) # [batch, 1, 1536]
            input_embeds = self.language_model.embed_tokens(input_ids) # [batch, seq_len, 1536]
            input_embeds = torch.cat(
                [features_embed, input_embeds[:, :-1, :]], 
                dim=1
            ) # [batch, seq_len, 1536]
            
            prefix_mask = torch.ones(input_ids.size(0), 1).to(device)
            attention_mask = torch.cat(
                [prefix_mask, attention_mask[:, :-1]],
                dim=1
            ) # [batch, seq_len]
            
            outputs = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state
            logits = self.lm_head(hidden_states) # [batch, seq_len, vocab_size]
            
            return logits, offsets
        else:
            # inference mode
            offsets = self.offset_head(entity_features) # [batch, 2]
            features_embed = self.feature_projection_head(entity_features.to(lm_dtype))  # [batch, 1536]
            features_embed = features_embed.unsqueeze(1)  # [batch, 1, 1536]
            
            generated_ids = torch.empty((entity_features.size(0), 0), dtype=torch.long, device=device)
            
            # generate for texts
            for _ in range(self.max_length):
                if generated_ids.size(1) > 0:
                    input_embeds = self.language_model.embed_tokens(generated_ids)  # [batch, cur_len, 1536]
                    input_embeds = torch.cat([features_embed, input_embeds], dim=1) # [batch, cur_len + 1, 1536]
                else:
                    input_embeds = features_embed # [batch, 1, 1536]
                
                attention_mask = torch.ones(input_embeds.size()[:2], device=device)  # [batch, cur_len+1]
                
                outputs = self.language_model(inputs_embeds=input_embeds,
                                              attention_mask=attention_mask
                                              )
                next_token_logits = self.lm_head(outputs.last_hidden_state[:, -1, :])  # [batch, vocab_size]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch, 1]
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                if (next_token == self.language_model.config.eos_token_id).all():
                    break
                
            return offsets, generated_ids  # [batch, 2] [batch, generated_seq_len]
        
    def lora_merge(self):
        if self.has_lora is True:
            self.language_model = self.language_model.merge_and_unload()