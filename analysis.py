# phat hien loi trong ham tinh bleu score sau khi da train xong mo hinh, do ko co thoi gian chay lai ( mat vai tieng ) nen em da viet lai 1 file tinh bleu score rieng dua tren training log
import matplotlib.pyplot as plt
import json
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

class TranslationAnalyzer:
    def __init__(self):
        self.history = []
        Path("plots").mkdir(exist_ok=True)
        
    def log_stats(self, epoch, train_loss, val_loss, batch_size, 
                  predictions=None, references=None, tokenizer=None):
        """Log training statistics"""
        run_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'batch_size': batch_size
        }
        
        # Calculate BLEU
        if predictions and references and tokenizer:
            bleu = self.calculate_bleu(predictions, references)
            run_data['bleu'] = bleu
            
        self.history.append(run_data)
        self.plot_metrics()
        
    def calculate_bleu(self, predictions, references):
        """Calculate corpus BLEU score"""
        smoother = SmoothingFunction()
        weights = (0.25, 0.25, 0.25, 0.25)  # weights for 4-gram BLEU
        
        try:
            if not predictions or not references:
                return 0
                
            # Calculate corpus BLEU
            bleu = corpus_bleu(references, predictions,
                             weights=weights,
                             smoothing_function=smoother.method1)
            return bleu
            
        except Exception as e:
            print(f"Error calculating BLEU score: {e}")
            return 0
        
    def plot_metrics(self):
        """Plot training metrics with improved visualization"""
        epochs = [data['epoch'] for data in self.history]
        train_losses = [data['train_loss'] for data in self.history]
        val_losses = [data['val_loss'] for data in self.history]
        
        plt.figure(figsize=(15, 5))
        
        # Plot losses 
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', marker='o')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Plot BLEU scores
        if 'bleu' in self.history[0]:
            plt.subplot(1, 2, 2)
            bleu_scores = [data['bleu'] for data in self.history]
            
            # Plot actual BLEU scores 
            plt.plot(epochs, bleu_scores, 'g-', label='BLEU Score', marker='o')
            plt.title('BLEU Score Progression')
            plt.xlabel('Epoch')
            plt.ylabel('BLEU Score')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            max_bleu = max(bleu_scores)
            plt.ylim(0, max_bleu * 1.1)  # Add 10% padding tren max value
            
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
            
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        metrics_file = Path('plots/training_metrics.json')
        metrics = {
            'epochs': epochs,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'bleu_scores': [data.get('bleu', 0) for data in self.history]
        }
        metrics_file.write_text(json.dumps(metrics, indent=2))