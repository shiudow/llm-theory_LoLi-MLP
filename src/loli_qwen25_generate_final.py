
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# =====================================================================
# 独自理論ヘッド（100% BF16密演算・残差なし完全スパース版）
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
        
        high_dim_features = self.fc_high(x_double)
        high_dim_active = torch.clamp(high_dim_features, min=0.0, max=1.0)
        output = self.down_proj(high_dim_active)
        return output

def ask_loli_chat(model, tokenizer, user_query, device):
    # Ichikaraデータに完全同期したプロンプト形式
    formatted_prompt = f"ユーザー: {user_query}\nシステム: "
    print(f"\n質問: 「{user_query}」")
    print("-" * 50)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,       
                do_sample=True,          
                temperature=0.3,         # 確実なファクトを引き出す低めの設定
                top_p=0.8,               
                pad_token_id=tokenizer.eos_token_id
            )
            
    # ★バグ修正の核心：outputs[0] を渡すことで、リストではなく「単一の純粋な文字列」としてデコードさせます
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # プロンプト部分をカットして、AIの純粋な回答だけを抽出
    response_only = generated_text[len(formatted_prompt):].strip()
    
    print(response_only if response_only else "(返答が空です)")
    print("-" * 50)

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

    weight_path = "./my_loli_qwen25_chat_final.pt"
    print(f"3. 公式データで仕上げ完了したチャット重み（{weight_path}）をロード中...")
    
    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print("→ ロード成功！")
    
    model.eval()
    print("\n==================================================")
    print("★【最終章・対話テスト】公式Ichikaraデータ適応版の知能")
    print("==================================================")

    # 運命のテスト
    ask_loli_chat(model, tokenizer, "日本の首都はどこですか？", device)
    
    # テスト2
    ask_loli_chat(model, tokenizer, "こんにちは！今日の天気を教えてください。", device)

    print("==================================================\n")

if __name__ == "__main__":
    main()
