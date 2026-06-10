
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

def draw_ascii_histogram_with_isolated_zero(tensor, title, bins=10):
    """他を含めず、完全な0の度数だけを完全に孤立・隔離して描画するヒストグラム関数"""
    data = tensor.detach().cpu().float().flatten()
    total_elements = data.numel()
    
    # 1. 完全な0 (1e-7未満の極小値) のみを隔離してカウント
    zero_mask = torch.abs(data) < 1e-7
    zero_count = torch.sum(zero_mask).item()
    zero_ratio = (zero_count / total_elements) * 100
    
    # 2. 0以外の純粋な実数データだけを抽出
    non_zero_data = data[~zero_mask]
    
    print(f"\n📊 【{title}】")
    print(f"   [統計] 全要素数: {total_elements:,} 個 | 完全な0: {zero_count:,} 個 ({zero_ratio:.4f} %)")
    
    if non_zero_data.numel() == 0:
        print("   (完全な0だけで満たされています)")
        return

    min_val = non_zero_data.min().item()
    max_val = non_zero_data.max().item()
    mean_val = non_zero_data.mean().item()
    std_val = non_zero_data.std().item()
    print(f"   [0以外の実数統計] 最小: {min_val:.4f} | 最大: {max_val:.4f} | 平均: {mean_val:.4f} | 標準偏差: {std_val:.4f}")
    
    # 3. 度数分布（グラフ）の描画計算
    counts = torch.histc(non_zero_data, bins=bins, min=min_val, max=max_val)
    
    # 0の数と0以外の山を両方考慮した最大値を基準にする
    max_count = max(counts.max().item(), zero_count)
    if max_count == 0: max_count = 1
        
    print("   [隔離型・度数分布グラフ]")
    
    # ★【一番上の特等席】完全な0だけの度数を、他を一切含めずにプロット
    zero_bar_len = int((zero_count / max_count) * 40)
    zero_bar = "█" * zero_bar_len
    print(f"    [完全な0.000] : {zero_bar:<40} ({zero_count})")
    print("    ------------------------------------------ (ここから0以外の実数の山)")
    
    # 0以外の実数の山を描画
    if max_val != min_val:
        step = (max_val - min_val) / bins
        for i in range(bins):
            lower = min_val + i * step
            upper = lower + step
            bar_length = int((counts[i].item() / max_count) * 40)
            bar = "█" * bar_length
            print(f"   {lower:6.3f} ~ {upper:6.3f} : {bar:<40} ({int(counts[i].item())})")


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    START_LAYER = 4
    
    print("1. オリジナルQwenモデルのロード中...")
    original_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    print("2. 完走した独自の重み（my_loli_qwen25_straight.pt）をロード中...")
    try:
        state_dict = torch.load("./my_loli_qwen25_straight.pt", map_location=device, weights_only=True)
    except FileNotFoundError:
        print("⚠️ my_loli_qwen25_straight.pt が見つかりません。")
        return

    # =====================================================================
    # ① 完全な0を隔離した、重みの度数分布（ヒストグラム）の比較
    # =====================================================================
    print("\n" + "="*75)
    print("★【重みの度数分布】完全な0だけを隔離した形状のビフォーアフター")
    print("="*75)
    
    # 改造前オリジナル (第4層)
    orig_w = original_model.model.layers[START_LAYER].mlp.gate_proj.weight
    draw_ascii_histogram_with_isolated_zero(orig_w, f"改造前: オリジナルQwen (第{START_LAYER}層 gate_proj)")
    
    # 改造後カスタム (第4層)
    loli_w_key = f"model.layers.{START_LAYER}.mlp.fc_high.weight"
    if loli_w_key in state_dict:
        draw_ascii_histogram_with_isolated_zero(state_dict[loli_w_key], f"改造後: カスタムLoLi-Qwen (第{START_LAYER}層 fc_high)")

    # =====================================================================
    # ② レイヤーの左（入力）から右（出力）への分布一覧表
    # =====================================================================
    print("\n" + "="*75)
    print("★【レイヤー分布一覧表】入力(左)から出力(右)へ流れる自作重み（fc_high）のレントゲン")
    print("="*75)
    print(" 階層  |   最小値   |   最大値   |   平均値   |  標準偏差  | 重み0の割合 ")
    print("-" * 75)
    
    total_layers = len(original_model.model.layers)
    for i in range(START_LAYER, total_layers):
        key = f"model.layers.{i}.mlp.fc_high.weight"
        if key in state_dict:
            w = state_dict[key].detach().cpu().float()
            
            w_min = w.min().item()
            w_max = w.max().item()
            w_mean = w.mean().item()
            w_std = w.std().item()
            
            # 0のカウント
            zeros = torch.sum(torch.abs(w) < 1e-7).item()
            z_ratio = (zeros / w.numel()) * 100
            
            # テーブル形式で出力
            print(f" L{i:02}   |  {w_min:>.4f}   |   {w_max:>.4f}   |  {w_mean:>.4f}   |   {w_std:>.4f}   |   {z_ratio:>5.2f} %")
            
    print("-" * 75)
    print(" 読み方: 上（L04）から下（L23）に向かって、知能のネットワークを左から右へ")
    print("        信号が流れていきます。各層の重みのレンジや0の比率の変化を確認できます。")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()

