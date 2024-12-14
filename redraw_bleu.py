# Hàm tính lại Bleu score dua tren training log
import matplotlib.pyplot as plt
import re

def parse_log(log_file):
    epochs = []
    bleu_scores = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'BLEU Score' in line:
                match = re.search(r'Epoch (\d+)/\d+.*BLEU Score: ([\d\.]+)', line)
                if match:
                    epochs.append(int(match.group(1)))
                    bleu_scores.append(float(match.group(2)))
                else:
                    match = re.search(r'BLEU Score: ([\d\.]+)', line)
                    if match:
                        epochs.append(len(epochs) + 1)
                        bleu_scores.append(float(match.group(1)))
    return epochs, bleu_scores

def plot_bleu(epochs, bleu_scores):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, bleu_scores, marker='o', linestyle='-', color='#448437', label='BLEU Score')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('BLEU Score Progression')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('bleu_score_corrected.png')  #luu bieu do moi = bleu_score_corrected.png
    plt.show()

if __name__ == '__main__':
    log_file = 'training_log.txt'
    epochs, bleu_scores = parse_log(log_file)
    plot_bleu(epochs, bleu_scores)