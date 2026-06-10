import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

def draw_ascii_histogram(tensor, title, bins=10):
    """テンソルの値の分布をターミナル上にASCIIアートのグラフで描画する関数"""
    data = tensor.detach().cpu().float().flatten()
    
    min_val = data.min().item()
    max_val = data.max().item()
    mean_val = data.mean().item()
    std_val = data.std().item()
    
    print(f"\n📊 【{title}】")
    print(f"   [統計値] 最小値: {min_val:.4f} | 最大値: {max_val:.4f} | 平均値: {mean_val:.4f} | 標準偏差: {std_val:.4f}")
    
    if max_val == min_val:
        print("   (すべての値が同一です)")
        return

    # ヒストグラムのカウント計算
    counts = torch.histc(data, bins=bins, min=min_val, max=max_val)
    max_count = counts.max().item()
    if max_count == 0: max_count = 1
        
    # グラフの描画
    print("   [分布グラフ]")
    step = (max_val - min_val) / bins
    for i in range(bins):
        lower = min_val + i * step
        upper = lower + step
        bar_length = int((counts[i].item() / max_count) * 40) # 最大幅40文字
        bar = "█" * bar_length
        print(f"   {lower:6.3f} ~ {upper:6.3f} : {bar} ({int(counts[i].item())})")

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_LAYER = 4  # 学習（部分移植）に合わせたターゲットレイヤーの指定
    
    print("1. オリジナルQwenモデルのロード中...")
    original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # ① 改造前のオリジナルモデル（第4レイヤーのゲート投影層）の重み分布
    orig_weight = original_model.model.layers[TARGET_LAYER].mlp.gate_proj.weight
    draw_ascii_histogram(orig_weight, f"改造前: オリジナルQwen(第{TARGET_LAYER}層 gate_proj) の重み分布")

    # ② 先ほど学習したご自身のカスタム重みのロードと解析
    print(f"\n2. 先ほど学習・保存した独自の重み（my_loli_qwen25.pt）から第{TARGET_LAYER}層の重みを解析中...")
    try:
        state_dict = torch.load("./my_loli_qwen25.pt", map_location=device, weights_only=True)
        
        # ターゲットレイヤーの自作 fc_high の重みをキー名から直接抽出
        loli_weight_key = f"model.layers.{TARGET_LAYER}.mlp.fc_high.weight"
        
        if loli_weight_key in state_dict:
            loli_weight = state_dict[loli_weight_key]
            draw_ascii_histogram(loli_weight, f"改造後: カスタムLoLi-Qwen(第{TARGET_LAYER}層 fc_high) の重み分布")
        else:
            print(f"⚠️ 指定されたキー {loli_weight_key} が重みファイル内に見つかりません。")
            
    except Exception as e:
        print(f"⚠️ 独自の重みファイルの読み込みまたは解析に失敗しました。詳細: {e}")

if __name__ == "__main__":
    main()

