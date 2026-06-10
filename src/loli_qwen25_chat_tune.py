
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# =====================================================================
# 1. 独自理論ヘッド（Qwenダイナミックレンジ対応 ＆ 残差なし 100% BF16密演算版）
# =====================================================================
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
        
        high_dim_features = self.fc_high(x_double)
        high_dim_active = torch.clamp(high_dim_features, min=0.0, max=1.0)
        output = self.down_proj(high_dim_active)
        return output


def main():
    # --- チャット仕上げ用のハイパーパラメータ（約20〜30分のじっくり放置設定） ---
    MAX_LENGTH = 128                
    BATCH_SIZE = 4                  
    GRAD_ACCUMULATION_STEPS = 4     # 実質バッチ16の安定学習
    LEARNING_RATE = 4e-5            # 138万件の基礎知識（Loss 2.08）を壊さないための愛護的な低学習率
    EPOCHS = 1

    base_weight_path = "./my_loli_qwen25_straight_mega.pt"
    output_filename = "./my_loli_qwen25_chat_final.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    print("【フェーズ2・最終章】発見していただいた公式Ichikaraデータによる対話調教")
    print(f"最終セーブ保存先: {output_filename}")
    
    print("\n1. オリジナルQwenモデルのロードと脳外科手術...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(device)

    for param in model.parameters():
        param.requires_grad = False
        
    hidden_dim = model.config.hidden_size          
    intermediate_dim = model.config.intermediate_size  
    
    for i, layer in enumerate(model.model.layers):
        if i >= 4:
            new_mlp = LogLinearActiveHead(in_dim=hidden_dim, intermediate_dim=intermediate_dim)
            new_mlp = new_mlp.to(device=device, dtype=layer.input_layernorm.weight.dtype)
            for param in new_mlp.parameters():
                param.requires_grad = True 
            layer.mlp = new_mlp

    print(f"2. 138万件完走重み（{base_weight_path}）を脳にロードして完全同期...")
    state_dict = torch.load(base_weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    print("3. 発見した公式「msfm/ichikara-instruction-all」データセットをロード中...")
    # ★修正の核心：見つけていただいた公式の正しいパスを指定！
    raw_dataset = load_dataset("msfm/ichikara-instruction-all", split="train")
    
    print(f"   → 総対話数: {len(raw_dataset)} 件を対話プロンプトにパッキング中...")
    
    def tokenize_function(examples):
        # 画面で見つかった「text（質問）」と「output（回答）」のキー形式を美しくチャット形式に縫い合わせる
        prompts = [f"ユーザー: {q}\nシステム: {a}" for q, a in zip(examples["text"], examples["output"])]
        return tokenizer(prompts, truncation=True, max_length=MAX_LENGTH, padding="max_length")

    # トークナイズ加工を実行
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    vocab_size = model.config.vocab_size
    loss_fn = nn.CrossEntropyLoss()

    print(f"4. 0-1スパース空間へのチャットルール調教開始（全 {len(dataloader)} ステップ）...")
    model.train()
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            inputs = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                outputs = model(inputs)
                logits = outputs.logits
                loss = loss_fn(logits.reshape(-1, vocab_size), labels.reshape(-1))
                loss = loss / GRAD_ACCUMULATION_STEPS

            loss.backward()
            running_loss += loss.item() * GRAD_ACCUMULATION_STEPS

            if (step + 1) % GRAD_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # 100ステップ（重み更新25回）ごとに進行状況をプリント
                if (step + 1) % 100 == 0:
                    current_loss = running_loss / 100
                    print(f"Step {step+1}/{len(dataloader)} | Chat-Loss: {current_loss:.4f}")
                    running_loss = 0.0

    print(f"チャット化完全成功！最終重みを保存します: {output_filename}")
    torch.save(model.state_dict(), output_filename)
    print("すべての工程が完了しました。")

if __name__ == "__main__":
    main()
