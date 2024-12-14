import torch
import pandas as pd
import sentencepiece as spm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomTokenizer:
    def __init__(self, vocab_size_ja: int = 2500, vocab_size_vi: int = 900): # giu nguyen 
        self.vocab_size_ja = vocab_size_ja
        self.vocab_size_vi = vocab_size_vi
        self.sp_ja = spm.SentencePieceProcessor()
        self.sp_vi = spm.SentencePieceProcessor()

    def train(self, ja_texts: List[str], vi_texts: List[str], model_dir: str) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        temp_ja = model_dir / "temp_ja.txt"
        temp_vi = model_dir / "temp_vi.txt"

        try:
            temp_ja.write_text('\n'.join(str(text) for text in ja_texts), encoding='utf-8')
            temp_vi.write_text('\n'.join(str(text) for text in vi_texts), encoding='utf-8')

            logger.info("Training Japanese tokenizer...")
            spm.SentencePieceTrainer.train(
                input=str(temp_ja),
                model_prefix=str(model_dir / "spm_ja"),
                vocab_size=self.vocab_size_ja,
                character_coverage=0.9995, # giu nguyen
                model_type='unigram',
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )

            logger.info("Training Vietnamese tokenizer...")
            spm.SentencePieceTrainer.train(
                input=str(temp_vi),
                model_prefix=str(model_dir / "spm_vi"),
                vocab_size=self.vocab_size_vi,
                character_coverage=0.9990, # giu nguyen
                model_type='unigram',
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3
            )

            self.sp_ja.load(str(model_dir / "spm_ja.model"))
            self.sp_vi.load(str(model_dir / "spm_vi.model"))
            logger.info("Tokenizers loaded successfully")

        finally:
            temp_ja.unlink(missing_ok=True)
            temp_vi.unlink(missing_ok=True)

    def encode(self, text: str, language: str, max_length: int = 96) -> Dict[str, torch.Tensor]:
        if not text or not isinstance(text, str) or language not in ['ja', 'vi']:
            logger.warning(f"Invalid input for encoding: {text}, language: {language}")
            return None

        sp_model = self.sp_ja if language == 'ja' else self.sp_vi
        try:
            # Encode as pieces 
            tokens = sp_model.encode_as_ids(text)

            # Add bos and eos tokens
            tokens = [sp_model.bos_id()] + tokens + [sp_model.eos_id()]

            # Create attention mask 
            attention_mask = [1] * len(tokens)

            if len(tokens) > max_length:
                tokens = tokens[:max_length - 1] + [sp_model.eos_id()]
                attention_mask = attention_mask[:max_length]
            else:
                padding_length = max_length - len(tokens)
                tokens += [sp_model.pad_id()] * padding_length
                attention_mask += [0] * padding_length  # 0s for padding

            # Check token IDs hop le hay khong
            if any(token < 0 or token >= sp_model.vocab_size() for token in tokens):
                logger.warning(f"Invalid token ID found. Vocab size: {sp_model.vocab_size()}")
                return None

            return {
                'input_ids': torch.tensor(tokens, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None

class CustomTranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: CustomTokenizer, max_length: int = 96):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.valid_indices = self._get_valid_indices()
        logger.info(f"Created dataset with {len(self.valid_indices)} valid samples out of {len(df)}")

    def _get_valid_indices(self) -> List[int]:
        valid_indices = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            ja_encoded = self.tokenizer.encode(row['japanese'], 'ja', self.max_length)
            vi_encoded = self.tokenizer.encode(row['vietnamese'], 'vi', self.max_length)
            if ja_encoded is not None and vi_encoded is not None:
                valid_indices.append(idx)
        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[self.valid_indices[idx]]
        ja_encoded = self.tokenizer.encode(row['japanese'], 'ja', self.max_length)
        vi_encoded = self.tokenizer.encode(row['vietnamese'], 'vi', self.max_length)

        return {
            'source_ids': ja_encoded['input_ids'],
            'source_attention_mask': ja_encoded['attention_mask'],
            'target_ids': vi_encoded['input_ids'],
            'target_attention_mask': vi_encoded['attention_mask']
        }

class DataPreprocessor:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.tokenizer = CustomTokenizer()
        self.tokenizer_dir = self.output_dir / "tokenizers"
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        self.train_df = None
        self.val_df = None

    def load_data(self) -> pd.DataFrame:
        csv_path = self.data_dir / 'parallel.csv'
        df = pd.read_csv(csv_path)
        df = df.dropna().reset_index(drop=True)
        df = df.sample(n=30000, random_state=42).reset_index(drop=True) # Use 30,000 pairs
        logger.info(f"Loaded {len(df)} samples from {csv_path}")
        return df

    def process(self) -> Tuple[DataLoader, DataLoader]:
        df = self.load_data()
        logger.info("Training tokenizers...")
        self.tokenizer.train(df['japanese'].tolist(), df['vietnamese'].tolist(), self.tokenizer_dir)

        # Chia dữ liệu thành tập training và validation một cách ngẫu nhiên
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Reset index cho train_df và val_df
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        self.train_df = train_df
        self.val_df = val_df

        logger.info(f"Split data: {len(train_df)} training samples, {len(val_df)} validation samples")

        train_dataset = CustomTranslationDataset(train_df, self.tokenizer)
        val_dataset = CustomTranslationDataset(val_df, self.tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=12, # giam tu 16 ve 12
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=12, # giam tu 16 ve 12
            shuffle=False,
            drop_last=True,
            collate_fn=self.collate_fn
        )

        logger.info("Data preprocessing completed")
        return train_loader, val_loader

    def collate_fn(self, batch):
        if not batch:
            return None

        source_ids = torch.nn.utils.rnn.pad_sequence(
            [item['source_ids'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        source_attention_masks = torch.nn.utils.rnn.pad_sequence(
            [item['source_attention_mask'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        target_ids = torch.nn.utils.rnn.pad_sequence(
            [item['target_ids'] for item in batch],
            batch_first=True,
            padding_value=0
        )
        target_attention_masks = torch.nn.utils.rnn.pad_sequence(
            [item['target_attention_mask'] for item in batch],
            batch_first=True,
            padding_value=0
        )

        return {
            'source_ids': source_ids,
            'source_attention_mask': source_attention_masks,
            'target_ids': target_ids,
            'target_attention_mask': target_attention_masks
        }

def main():
    logger.info("Starting preprocessi")
    preprocessor = DataPreprocessor('data/aligned', 'data/preprocessed_test')
    train_loader, val_loader = preprocessor.process()
    logger.info(" completed successfully")

if __name__ == "__main__":
    main()