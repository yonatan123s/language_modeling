import argparse
import time
import math
import torch
import torch.nn as nn
from data import PTBDataset
from model import RNNModel
from ntasgd import NTASGD
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank LSTM Language Model')
parser.add_argument('--data', type=str, default='../next_word_pred/ptbdataset',
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=39,
                    help='upper epoch limit (original paper used 39 epochs)')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--emsize', type=int, default=1500,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1500,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=5.0,
                    help='gradient clipping')
parser.add_argument('--epochs_decay', type=int, default=6,
                    help='number of epochs before learning rate decay')
parser.add_argument('--tie_weights', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--mode', choices=['train', 'eval', 'test'], required=True,
                    help='Mode to run the script: train, eval, test')
parser.add_argument('--checkpoint', type=str, default='weights.pth',
                    help='Path to save/load the model checkpoint')

parser.add_argument("--weight_decay", type=float, default=1.2e-6, help="The weight decay parameter.")
parser.add_argument("--non_mono", type=int, default=5, help="The masking length for non-monotonicity. Referred to as 'n' in the paper.")
parser.add_argument("--dropout_i", type=float, default=0.4, help="The dropout parameter on word vectors.")
parser.add_argument("--dropout_l", type=float, default=0.3, help="The dropout parameter between LSTM layers.")
parser.add_argument("--dropout_o", type=float, default=0.4, help="The dropout parameter on the last LSTM layer.")
parser.add_argument("--dropout_e", type=float, default=0.1, help="The dropout parameter on the embedding layer.")
args = parser.parse_args()
# Set the random seed for reproducibility
torch.manual_seed(1111)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Load data
###############################################################################

dataset = PTBDataset(args.data)
ntokens = len(dataset.dictionary)


def get_seq_len(bptt):
    seq_len = bptt if np.random.random() < 0.95 else bptt / 2
    seq_len = round(np.random.normal(seq_len, 5))
    while seq_len <= 5 or seq_len >= 90:
        seq_len = bptt if np.random.random() < 0.95 else bptt / 2
        seq_len = round(np.random.normal(seq_len, 5))
    return seq_len


# Function to batchify data
def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(dataset.train_data, args.batch_size)
val_data = batchify(dataset.valid_data, eval_batch_size)
test_data = batchify(dataset.test_data, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

model = RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout_i, args.dropout_l, args.dropout_o, args.dropout_e, args.tie_weights).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = NTASGD(model.parameters(), lr=args.lr, n=args.non_mono, weight_decay=args.weight_decay, fine_tuning=False)

###############################################################################
# Helper functions
###############################################################################

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def get_batch_train(source, i):
    seq_len = get_seq_len(args.bptt)
    seq_len = min(seq_len, len(source) - 1 - i)
    lr = seq_len/args.bptt*args.lr
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target, lr

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size, device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = tuple(h.detach() for h in hidden)
    return total_loss / (len(data_source) - 1)

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size, device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets, lr = get_batch_train(train_data, i)
        optimizer.lr(lr)
        hidden = tuple(h.detach() for h in hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        if batch % 200 == 0 and batch > 0:
            cur_loss = total_loss / 200
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                  'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // args.bptt, lr,
                      elapsed * 1000 / 200, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def test_interactive():
    model.eval()
    word_to_idx = dataset.dictionary.word2idx
    idx_to_word = dataset.dictionary.idx2word

    print("Enter 'end' or 'stop' to exit.")
    while True:
        input_seq = input("Enter a sequence of words: ")
        if input_seq.lower() in ['end', 'stop']:
            break
        words = input_seq.strip().split()
        # Convert words to indices
        input_indices = []
        for word in words:
            if word in word_to_idx:
                input_indices.append(word_to_idx[word])
            else:
                print(f"Word '{word}' not in vocabulary, using <unk> token.")
                input_indices.append(word_to_idx.get('<unk>', 0))

        input_tensor = torch.tensor(input_indices, dtype=torch.long).unsqueeze(1).to(device)
        hidden = model.init_hidden(1, device)

        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            last_output = output[-1]
            probabilities = torch.softmax(last_output, dim=0)
            top5_prob, top5_idx = torch.topk(probabilities, k=5)
            top5_words = [idx_to_word[idx] for idx in top5_idx.tolist()]
            print("Top 5 predictions:")
            for i, (word, prob) in enumerate(zip(top5_words, top5_prob.tolist())):
                print(f"{i+1}. {word} ({prob*100:.2f}%)")

def calculate_perplexity():
    # Load the saved model
    with open(args.checkpoint, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))

    model.eval()
    datasets = {'Train': train_data, 'Validation': val_data, 'Test': test_data}
    batch_sizes = {'Train': args.batch_size, 'Validation': eval_batch_size, 'Test': eval_batch_size}

    for name, data_source in datasets.items():
        total_loss = 0.
        hidden = model.init_hidden(batch_sizes[name], device)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, args.bptt):
                data, targets = get_batch(data_source, i)
                output, hidden = model(data, hidden)
                output_flat = output
                total_loss += len(data) * criterion(output_flat, targets).item()
                hidden = tuple(h.detach() for h in hidden)
        avg_loss = total_loss / (len(data_source) - 1)
        perplexity = math.exp(avg_loss)
        print(f'{name} Perplexity: {perplexity:.2f}')

###############################################################################
# Main Execution
###############################################################################

if args.mode == 'train':
    # Training Loop
    lr = args.lr
    best_val_loss = None

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.checkpoint, 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen.
                lr /= 1.2
    except KeyboardInterrupt:
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.checkpoint, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

elif args.mode == 'eval':
    # Evaluate the model on train, validation, and test datasets
    calculate_perplexity()

elif args.mode == 'test':
    # Load the saved model
    with open(args.checkpoint, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    test_interactive()
