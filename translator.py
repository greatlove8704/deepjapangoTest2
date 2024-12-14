import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from preprocesstest import CustomTokenizer, CustomTranslationDataset, DataPreprocessor
from typing import Optional, Dict
from analysis import TranslationAnalyzer
import time
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

torch.backends.cuda.max_split_size_mb = 128

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.3, max_len: int = 96): # giam tu 128 ve 96
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class JapaneseVietnameseTranslator(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int = 256, # giam xuong 512 ve 256
            nhead: int = 4, # giu nguyen
            num_encoder_layers: int = 3, # giam tu 4 xuong 3
            num_decoder_layers: int = 3, # giam tu 4 xuong 3
            dim_feedforward: int = 256, # giam tu 512 xuong 256
            dropout: float = 0.3, # tang dropout 
            max_len: int = 96
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size + 4
        self.tgt_vocab_size = tgt_vocab_size + 4
        self.d_model = d_model

        self.src_norm = nn.LayerNorm(d_model)
        self.tgt_norm = nn.LayerNorm(d_model)

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model, padding_idx=0)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.output_projection = nn.Linear(d_model, self.tgt_vocab_size)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='linear')

        with torch.no_grad():
            self.output_projection.weight *= 0.1
            if self.output_projection.bias is not None:
                self.output_projection.bias.zero_()

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones((sz, sz), device=device, dtype=torch.bool), diagonal=1)
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_attention_mask: Optional[torch.Tensor] = None,
            tgt_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if src.size(0) == 0 or tgt.size(0) == 0:
            raise ValueError("Empty batch detected")

        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embed = self.src_norm(self.pos_encoder(src_embed))

        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.tgt_norm(self.pos_encoder(tgt_embed))

        src_padding_mask = (src == 0)
        tgt_padding_mask = (tgt == 0)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)

        if src_attention_mask is not None:
            src_attention_mask = src_attention_mask.bool()
            src_padding_mask = src_padding_mask | (~src_attention_mask)
        if tgt_attention_mask is not None:
            tgt_attention_mask = tgt_attention_mask.bool()
            tgt_padding_mask = tgt_padding_mask | (~tgt_attention_mask[:, :-1])

        try:
            output = self.transformer(
                src_embed,
                tgt_embed,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )

            return self.output_projection(output)
        except Exception as e:
            print(f"Error in transformer forward pass: {str(e)}")
            raise

class TranslationTrainer:
    def __init__(
            self,
            model: JapaneseVietnameseTranslator,
            train_loader: DataLoader,
            val_loader: DataLoader,
            save_dir: str,
            learning_rate: float = 0.00005, # giu nguyen learning rate
            gradient_accumulation_steps: int = 2, # 4 xuong 2, cập nhật trọng số thường xuyên hơn
            weight_decay: float = 0.0005
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5, # giu nguyen
            patience=3, # tang tu 2 len 3
            verbose=True
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.device = torch.device('cpu')
        self.model.to(self.device)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        valid_batches = 0
        self.optimizer.zero_grad()

        for i, batch in enumerate(self.train_loader):
            if batch is None or not batch['source_ids'].size(0):
                continue

            try:
                src = batch['source_ids'].to(self.device)
                tgt = batch['target_ids'].to(self.device)
                src_mask = batch['source_attention_mask'].to(self.device)
                tgt_mask = batch['target_attention_mask'].to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                outputs = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))

                if not torch.isfinite(loss):
                    print(f"NaN loss detected in batch {i}, skipping...")
                    continue

                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (i + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                    valid_gradients = True
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            valid_gradients = valid_gradients and torch.isfinite(param.grad).all()

                    if valid_gradients:
                        self.optimizer.step()
                    else:
                        print("Invalid gradients detected, skipping update")

                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.gradient_accumulation_steps
                valid_batches += 1

                if i % 10 == 0:
                    print(f"Batch {i}: Loss = {loss.item() * self.gradient_accumulation_steps:.4f}")

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue

        return total_loss / valid_batches if valid_batches > 0 else float('inf')

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0
        valid_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if batch is None or not batch['source_ids'].size(0):
                    continue

                try:
                    src = batch['source_ids'].to(self.device)
                    tgt = batch['target_ids'].to(self.device)
                    src_mask = batch['source_attention_mask'].to(self.device)
                    tgt_mask = batch['target_attention_mask'].to(self.device)

                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]

                    outputs = self.model(src, tgt_input, src_mask, tgt_mask)
                    loss = self.criterion(outputs.reshape(-1, outputs.size(-1)), tgt_output.reshape(-1))

                    if torch.isfinite(loss):
                        total_loss += loss.item()
                        valid_batches += 1

                except Exception as e:
                    print(f"Error during validation batch {i}: {e}")
                    continue

        avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
        self.scheduler.step(avg_loss)
        return avg_loss

def main():
    gc.collect()
    start_time = time.time()

    try:
        print("Loading data...")
        preprocessor = DataPreprocessor('./data/aligned', './data/preprocessed_test')
        train_loader, val_loader = preprocessor.process()

        src_vocab_size = train_loader.dataset.tokenizer.sp_ja.vocab_size()
        tgt_vocab_size = val_loader.dataset.tokenizer.sp_vi.vocab_size()

        # Create log file
        log_file = Path("training_log.txt")
        with open(log_file, "w") as f:
            f.write("Training Log\n")
            f.write("="*50 + "\n")
            f.write(f"Japanese Vocabulary Size: {src_vocab_size}\n")
            f.write(f"Vietnamese Vocabulary Size: {tgt_vocab_size}\n")
            f.write("="*50 + "\n\n")

        print(f"Japanese Vocabulary Size: {src_vocab_size}")
        print(f"Vietnamese Vocabulary Size: {tgt_vocab_size}")

        print("Initializing model...")
        model = JapaneseVietnameseTranslator(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=256,
            nhead=4,
            num_encoder_layers=3, # 
            num_decoder_layers=3,
            dim_feedforward=256,
            dropout=0.3,
            max_len=96
        )

        trainer = TranslationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir='checkpoints',
            learning_rate=0.00005,
            gradient_accumulation_steps=2, # giam tu 6 xuong 4 xuong 2
            weight_decay=0.0005 # giu nguyen
        )

        analyzer = TranslationAnalyzer()

        num_epochs = 10 # giu nguyen

        best_val_loss = float('inf')
        smoother = SmoothingFunction()
        weights = (0.25, 0.25, 0.25, 0.25)
        max_eval_batches = 100 # tang len 100

        print("Starting training...")
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Training phase
            train_loss = trainer.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")

            # Validation phase
            val_loss = trainer.validate()
            print(f"Validation Loss: {val_loss:.4f}")

            # Calculate BLEU score tren tap validation
            model.eval()
            all_predictions = []
            all_references = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= max_eval_batches:
                        break

                    src = batch['source_ids'].to(trainer.device)
                    tgt = batch['target_ids'].to(trainer.device)

                    for sample_idx in range(len(src)):
                        src_sentence = src[sample_idx:sample_idx+1]
                        tgt_sentence = torch.tensor([[2]], device=trainer.device)  # BOS token

                        for _ in range(96):
                            out = model(src_sentence, tgt_sentence)
                            next_token = torch.argmax(out[0, -1])
                            tgt_sentence = torch.cat([tgt_sentence, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                            if next_token.item() == 3:  # EOS token
                                break

                        pred_text = val_loader.dataset.tokenizer.sp_vi.decode(tgt_sentence[0][1:].tolist())
                        ref_text = val_loader.dataset.tokenizer.sp_vi.decode(tgt[sample_idx][1:].tolist())

                        # Tokenize de tinh BLEU
                        pred_tokens = val_loader.dataset.tokenizer.sp_vi.encode_as_pieces(pred_text)
                        ref_tokens = [val_loader.dataset.tokenizer.sp_vi.encode_as_pieces(ref_text)]

                        all_predictions.append(pred_tokens)
                        all_references.append(ref_tokens)

            # Calculate corpus BLEU
            bleu_score = corpus_bleu(all_references, all_predictions,
                                   weights=weights,
                                   smoothing_function=smoother.method1)

            epoch_time = time.time() - epoch_start_time
            with open(log_file, "a") as f:
                f.write(f"\nEpoch {epoch + 1}/{num_epochs}\n")
                f.write(f"Training Loss: {train_loss:.4f}\n")
                f.write(f"Validation Loss: {val_loss:.4f}\n")
                f.write(f"BLEU Score: {bleu_score:.4f}\n")
                f.write(f"Epoch Time: {epoch_time/60:.2f} minutes\n")
                f.write("-"*50 + "\n")
            analyzer.log_stats(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                batch_size=train_loader.batch_size,
                predictions=[p for p in all_predictions],
                references=[r[0] for r in all_references],
                tokenizer=val_loader.dataset.tokenizer.sp_vi
            )

            print(f"BLEU Score: {bleu_score:.4f}")
            print(f"Epoch time: {epoch_time/60:.2f} minutes")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = trainer.save_dir / f"best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer.model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'bleu_score': bleu_score,
                }, save_path)
                print(f"Saved best model with validation loss: {val_loss:.4f}")

            # Save checkpoint
            checkpoint_path = trainer.save_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'bleu_score': bleu_score,
            }, checkpoint_path)

        total_time = time.time() - start_time
        print("\nTraining completed!")
        print(f"Total training time: {total_time/3600:.2f} hours")
        print(f"Best validation loss: {best_val_loss:.4f}")

        with open(log_file, "a") as f:
            f.write("\nTraining Summary\n")
            f.write("="*50 + "\n")
            f.write(f"Total training time: {total_time/3600:.2f} hours\n")
            f.write(f"Best validation loss: {best_val_loss:.4f}\n")
            f.write("="*50 + "\n")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Log error
        with open(log_file, "a") as f:
            f.write(f"\nError during training: {str(e)}\n")
        raise

if __name__ == "__main__":
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    main()