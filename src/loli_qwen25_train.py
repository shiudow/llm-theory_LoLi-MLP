
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


# =====================================================================
# 2. メイン学習ルーチン（自動バックアップセーブ機能付き）
# =====================================================================
def main():
    MAX_LENGTH = 256
    BATCH_SIZE = 4                  
    GRAD_ACCUMULATION_STEPS = 8     
    LEARNING_RATE = 3e-4            
    EPOCHS = 1
    START_LAYER = 4                 

    output_filename = "./my_loli_qwen25_straight_mega.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    print(f"検証モード: 【残差なし ＋ 138万件フル（自動セーブ対応）】")
    print(f"保存先: {output_filename}")
    
    print("\n1. オリジナル Qwen 2.5 モデルとトークナイザーをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(model_name:="Qwen/Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    print(f"2. 脳外科手術（レイヤー {START_LAYER} 以降に独自ヘッドを移植）...")
    for param in model.parameters():
        param.requires_grad = False
        
    hidden_dim = model.config.hidden_size          
    intermediate_dim = model.config.intermediate_size  
    
    for i, layer in enumerate(model.model.layers):
        if i >= START_LAYER:
            new_mlp = LogLinearActiveHead(in_dim=hidden_dim, intermediate_dim=intermediate_dim)
            new_mlp = new_mlp.to(device=device, dtype=layer.input_layernorm.weight.dtype)
            
            for param in new_mlp.parameters():
                param.requires_grad = True
                
            layer.mlp = new_mlp

    print("3. 日本語Wikipediaデータの読み込みとトークナイズ加工中（138万件フル）...")
    raw_dataset = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train")
    raw_dataset = raw_dataset.select(range(len(raw_dataset))) 
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    
    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text", "id", "url", "title"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    dataloader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)
    
    vocab_size = model.config.vocab_size
    loss_fn = nn.CrossEntropyLoss()

    print("5. メガスケール自動セーブ学習の開始...")
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
                
                # ★1000ステップごとに進捗表示し、同時に重みを「自動上書き保存」する！
                if (step + 1) % 1000 == 0:
                    current_loss = running_loss / 1000
                    print(f"Step {step+1}/{len(dataloader)} | Loss: {current_loss:.4f} [自動バックアップ実行中...]")
                    
                    # 途中経過を安全にSSDへ退避（Ctrl+Cの対策）
                    torch.save(model.state_dict(), output_filename)
                    running_loss = 0.0

    print(f"学習完全成功！最終重みを保存します: {output_filename}")
    torch.save(model.state_dict(), output_filename)
    print("すべての工程が完了しました。")

if __name__ == "__main__":
    main()
