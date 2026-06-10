
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

original_zeros = []
loli_zeros = []

class LogLinearActiveHead(nn.Module):
    def __init__(self, in_dim, intermediate_dim):
        super().__init__()
        self.fc_high = nn.Linear(in_dim * 2, intermediate_dim, bias=True)
        self.down_proj = nn.Linear(intermediate_dim, in_dim, bias=False)

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        direction = x / (norm + 1e-8)
        log_norm = torch.log(norm + 1.0)
        x_log_scale = direction * log_norm
        
        x_double = torch.cat([x, x_log_scale], dim=-1)
        x_double = x_double.to(dtype=self.fc_high.weight.dtype)
        
        # 4864次元に広げる
        high_dim_features = self.fc_high(x_double)
        # 広げた後でクランプ
        high_dim_active = torch.clamp(high_dim_features, min=0.0, max=1.0)
        
        # ★【真の計測位置】4864次元の内部で完全に0に叩き落とされた割合（％）を直接カウント
        zero_ratio = (high_dim_active == 0.0).float().mean().item() * 100
        loli_zeros.append(zero_ratio)
        
        output = self.down_proj(high_dim_active)
        return output

def original_mlp_hook(module, input, output):
    zero_ratio = (output == 0.0).float().mean().item() * 100
    original_zeros.append(zero_ratio)

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    START_LAYER = 4
    
    print("1. オリジナルQwenモデルをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    for i, layer in enumerate(original_model.model.layers):
        if i >= START_LAYER:
            layer.mlp.act_fn.register_forward_hook(original_mlp_hook)
        
    print(f"2. 改造用モデル（LoLi-Qwen）にヘッドを部分移植中...")
    loli_model = copy.deepcopy(original_model).to(device)
    hidden_dim = loli_model.config.hidden_size
    intermediate_dim = loli_model.config.intermediate_size
    
    for i, layer in enumerate(loli_model.model.layers):
        if i >= START_LAYER:
            layer.mlp = LogLinearActiveHead(in_dim=hidden_dim, intermediate_dim=intermediate_dim).to(device)

    print("3. 完走した独自の重み（my_loli_qwen25_straight.pt）をロード中...")
    state_dict = torch.load("./my_loli_qwen25_straight.pt", map_location=device, weights_only=True)
    loli_model.load_state_dict(state_dict)
    print("→ ロード成功。")
        
    original_model.eval()
    loli_model.eval()
    
    prompt = "こんにちは、今日の天気は"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"\nプロンプト「{prompt}」を入力して4864次元のスパース率を計測中...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = original_model(**inputs)
            _ = loli_model(**inputs)
            
    avg_orig = sum(original_zeros) / len(original_zeros) if original_zeros else 0
    avg_loli = sum(loli_zeros) / len(loli_zeros) if loli_zeros else 0
    
    print("\n" + "="*60)
    print(f"★【再検証結果】4864次元空間の真の0スパース率比較")
    print("="*60)
    print(f"■ 改造前（オリジナル Qwen 2.5）の平均 0スパース率: {avg_orig:.4f} %")
    print(f"■ 改造後（カスタム LoLi-Qwen）  の平均 0スパース率: {avg_loli:.4f} %")
    print("-" * 60)
    
    if avg_loli > avg_orig:
        diff = avg_loli - avg_orig
        print(f"成果: 独自理論により、4864次元の脳内の無駄な情報がさらに 【 {diff:.2f}% 】 完全に0へ削ぎ落とされました！")
    else:
        print("スパース率に顕著な差が見られませんでした。")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
