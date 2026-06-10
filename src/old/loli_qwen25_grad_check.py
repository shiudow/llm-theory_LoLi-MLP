import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

# =====================================================================
# 独自理論ヘッド（検証用）
# =====================================================================
class LogLinearActiveHead(nn.Module):
    def __init__(self, in_dim, intermediate_dim):
        super().__init__()
        self.fc_high = nn.Linear(in_dim * 2, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, in_dim, bias=False)

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        direction = x / (norm + 1e-8)
        log_norm = torch.log(norm + 1.0)
        x_log_scale = direction * log_norm
        
        x_double = torch.cat([x, x_log_scale], dim=-1)
        x_active = torch.clamp(x_double, min=0.0, max=1.0)
        x_active = x_active.to(dtype=self.fc_high.weight.dtype)
        
        high_dim_features = self.fc_high(x_active)
        output = self.down_proj(high_dim_features)
        return output

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    START_LAYER = 4  # 部分移植の開始レイヤー

    print("1. モデルとトークナイザーをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    print(f"2. レイヤー {START_LAYER} 以降に独自ヘッドを移植中...")
    hidden_dim = model.config.hidden_size
    intermediate_dim = model.config.intermediate_size
    
    for i, layer in enumerate(model.model.layers):
        if i >= START_LAYER:
            new_mlp = LogLinearActiveHead(in_dim=hidden_dim, intermediate_dim=intermediate_dim)
            new_mlp = new_mlp.to(device=device, dtype=layer.input_layernorm.weight.dtype)
            
            # 勾配を計算できるようにrequires_gradをTrueにする
            for param in new_mlp.parameters():
                param.requires_grad = True
                
            layer.mlp = new_mlp

    print("3. バックプロパゲーションを1ステップ再現して勾配をスキャン中...")
    model.train()
    
    # 動作検証用の短い入力を流す
    prompt = "こんにちは、今日の天気は"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # 簡単な言語モデルタスクの計算
    x_inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(x_inputs)
        logits = outputs.logits
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), labels.reshape(-1))

    # 逆伝播を実行して「勾配」を発生させる
    loss.backward()

    # =====================================================================
    # 【本命】各レイヤーの fc_high の勾配の強さ（L1ノルム）を計測してグラフ化
    # =====================================================================
    print("\n" + "="*75)
    print("★【可視化】レイヤー深部への『勾配消失（学習シグナルの減衰）』チェック")
    print("="*75)
    print(" 階層  |  勾配の強さ (平均絶対値)  | 伝播の視覚グラフ (エネルギー残量)")
    print("-" * 75)

    grad_values = []
    layers_indices = []

    # 移植した各レイヤーの勾配を集計
    for i in range(START_LAYER, len(model.model.layers)):
        grad = model.model.layers[i].mlp.fc_high.weight.grad
        if grad is not None:
            # 勾配の平均絶対値（L1ノルム）を計算
            grad_mean = grad.abs().mean().item()
            grad_values.append(grad_mean)
            layers_indices.append(i)

    # 視覚グラフの最大幅を調整するための基準値
    max_grad = max(grad_values) if grad_values else 1.0

    for idx, grad_val in zip(layers_indices, grad_values):
        # 最大の勾配の強さを40文字として、相対的な長さをグラフ化
        bar_length = int((grad_val / (max_grad + 1e-12)) * 40)
        bar = "█" * bar_length
        
        # 非常に小さな勾配（消失状態）をわかりやすく指数表記で出力
        print(f" L{idx:02}   |   {grad_val:.2e}           | {bar}")

    print("-" * 75)
    print(" 読み方: 出力は「L23（深部）」から「L04（浅部）」に向かって逆方向に流れます。")
    print("        上（L04）に比べて下（L23）のバーが極端に短ければ、クランプによって")
    print("        『奥の階層まで学習のシグナルが届いていない（勾配消失）』証拠です。")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()

