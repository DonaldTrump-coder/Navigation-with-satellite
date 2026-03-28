from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

def load_ocr_model(model_path):
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)
    return model

if __name__ == '__main__':
    model_path = 'D:/Projects/Navigation-with-satellite/SceneGraph_Generation/models/GLM-OCR'
    model = load_ocr_model(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    
    language_model = model.model.language_model
    lm_head = model.lm_head
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    features = torch.randn(1, 1, 1536).to(device).to(dtype)
    eos_token_id = processor.tokenizer.eos_token_id
    generated_ids = []
    inputs_embeds = features
    attention_mask = torch.ones(1, 1).to(device)
    max_new_tokens = 20
    for _ in range(max_new_tokens):
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        logits = lm_head(outputs.last_hidden_state)
        next_token_logits = logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1)
        if next_token_id.item() == eos_token_id:
            break
        generated_ids.append(next_token_id.item())
        next_token_embed = language_model.embed_tokens(next_token_id).unsqueeze(1)
        inputs_embeds = torch.cat([inputs_embeds, next_token_embed], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones(1, 1).to(device)], dim=1)
    text = processor.decode(generated_ids, skip_special_tokens=True)
    print(text)