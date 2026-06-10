
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# 1. 独自理論ヘッド（Qwenダイナミックレンジ対応 ＆ 残差なし 100% BF16密演算版）
# =====================================================================
class LogLinearActiveHead(nn.Module):
    def __init__(self, in_dim, intermediate_dim):
        super().__init__()
        self.in_dim = in_dim                    # 896
        self.expanded_dim = in_dim * 2          # 1792
        self.intermediate_dim = intermediate_dim  # 4864

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

def test_inference(model, tokenizer, prompt, device, title):
    print(f"\nプロンプト: 「{prompt}」")
    print("-" * 50)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,       # 少し長めに喋らせます
                do_sample=True,          
                temperature=0.7,         
                top_p=0.9,               
                pad_token_id=tokenizer.eos_token_id
            )
            
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)
    print("-" * 50)

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    START_LAYER = 4  # 学習側と完全同期
    
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

    weight_path = "./my_loli_qwen25_straight_mega.pt"
    print(f"3. 中断・保存された独自の重み（{weight_path}）をロード中...")
    
    try:
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print("→ ロード成功！")
    except FileNotFoundError:
        print(f"⚠️ {weight_path} が見つかりません。ファイル名を確認してください。")
        return
    
    model.eval()
    print("\n==================================================")
    print("★【検証】LoLi-Qwen（Loss 2.95 / 残差なしスパース脳）の知能テスト")
    print("==================================================")

    # テスト1: 昨夜から追いかけている気象の文脈がどう進化したか
    test_inference(model, tokenizer, "こんにちは、今日の天気は", device, "テスト1")
    
    # テスト2: Wikipediaの硬派な客観的知識をどう復元するか
    test_inference(model, tokenizer, "日本の首都は", device, "テスト2")

    print("==================================================\n")

if __name__ == "__main__":
    main()
