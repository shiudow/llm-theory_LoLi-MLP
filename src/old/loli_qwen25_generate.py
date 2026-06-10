
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class LogLinearActiveHead(nn.Module):
    def __init__(self, in_dim, intermediate_dim):
        super().__init__()
        self.in_dim = in_dim
        self.expanded_dim = in_dim * 2
        self.intermediate_dim = intermediate_dim
        self.fc_high = nn.Linear(self.expanded_dim, intermediate_dim, bias=True)
        self.down_proj = nn.Linear(intermediate_dim, in_dim, bias=False)

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        direction = x / (norm + 1e-8)
        log_norm = torch.log(norm + 1.0)
        x_log_scale = direction * log_norm
        
        x_double = torch.cat([x, x_log_scale], dim=-1)
        x_double = x_double.to(dtype=self.fc_high.weight.dtype)
        
        # 正しい計算順序：広げてからクランプ（残差なし）
        high_dim_features = self.fc_high(x_double)
        high_dim_active = torch.clamp(high_dim_features, min=0.0, max=1.0)
        output = self.down_proj(high_dim_active)
        return output

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    START_LAYER = 4
    
    print("1. ベースモデルとトークナイザーの準備中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    print(f"2. 脳外科手術（レイヤー {START_LAYER} 以降に【残差なし】独自ヘッドを移植）...")
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size
    
    for i, layer in enumerate(model.model.layers):
        if i >= START_LAYER:
            new_mlp = LogLinearActiveHead(in_dim=hidden_dim, intermediate_dim=intermediate_dim)
            new_mlp = new_mlp.to(device=device, dtype=layer.input_layernorm.weight.dtype)
            layer.mlp = new_mlp

    print("3. 完走した重み（my_loli_qwen25_straight.pt）を脳にロード中...")
    state_dict = torch.load("./my_loli_qwen25_straight.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    model.eval()
    print("\n--- ロード完了！知能テストを開始します ---")

    prompt = "こんにちは、今日の天気は"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
    generated_text = tokenizer.decode(outputs, skip_special_tokens=True)
    print("\n==================================================")
    print("★【生成結果】カスタムLoLi-Qwen（残差なし50万件完走版）:")
    print("==================================================")
    print(generated_text)
    print("==================================================\n")

if __name__ == "__main__":
    main()
