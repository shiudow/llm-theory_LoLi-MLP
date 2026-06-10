import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import copy

# 構造比較用の独自ヘッド定義
class LogLinearActiveHead(nn.Module):
    def __init__(self, in_dim, intermediate_dim):
        super().__init__()
        self.fc_high = nn.Linear(in_dim * 2, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, in_dim, bias=False)
    def forward(self, x): return x

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    START_LAYER = 4  # 部分移植の開始レイヤー
    
    print("1. オリジナルQwenモデルの構造をスキャン中...")
    original_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print(f"2. 改造用モデル（LoLi-Qwen）の第{START_LAYER}層以降へ独自ヘッドを移植中...")
    loli_model = copy.deepcopy(original_model)
    hidden_dim = loli_model.config.hidden_size
    intermediate_dim = loli_model.config.intermediate_size
    
    for i, layer in enumerate(loli_model.model.layers):
        if i >= START_LAYER:
            layer.mlp = LogLinearActiveHead(in_dim=hidden_dim, intermediate_dim=intermediate_dim)

    # =====================================================================
    # 【本命】全24レイヤーの mlp 構造を一覧比較表示
    # =====================================================================
    print("\n" + "="*75)
    print(f"★【レイヤー構造比較】全24層の MLP (FFN) パーツのビフォーアフター")
    print("="*75)
    print(f" 階層 | {'[改造前] オリジナル Qwen 2.5':<32} | {'[改造後] カスタム LoLi-Qwen':<32}")
    print("-" * 75)
    
    for i in range(len(original_model.model.layers)):
        orig_mlp_name = original_model.model.layers[i].mlp.__class__.__name__
        loli_mlp_name = loli_model.model.layers[i].mlp.__class__.__name__
        
        # 移植対象外の層（0〜3層）と、移植対象の層（4〜23層）でマークを変える
        status_marker = "★" if i >= START_LAYER else "  "
        
        print(f" {status_marker}L{i:02} | {orig_mlp_name:<32} | {loli_mlp_name:<32}")
        
    print("-" * 75)
    print(" 注: ★マークがついている階層（L04〜L23）が、独自理論ヘッドへの差し替え領域です。")
    print("     L00〜L03の浅い階層は、オリジナルの流暢なシグナルを保つためそのまま残されています。")
    print("="*75 + "\n")

    # =====================================================================
    # おまけ：特定の差し替えレイヤー（第4層）の詳細な中身の対比
    # =====================================================================
    print("【深堀り】第4レイヤー（移植境界）の具体的な内部グラフ対比:")
    print("-" * 75)
    print("▼ [改造前] オリジナルの第4層 MLP:")
    print(original_model.model.layers[TARGET_LAYER:=START_LAYER].mlp)
    print("\n▼ [改造後] 独自理論に切り替わった第4層 MLP:")
    print(loli_model.model.layers[TARGET_LAYER].mlp)
    print("-" * 75 + "\n")

if __name__ == "__main__":
    main()
