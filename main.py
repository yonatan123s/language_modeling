import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from data import PTBDataset
from model import RNNModel

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
args = parser.parse_args()

# Set the random seed for reproducibility
torch.manual_seed(1111)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Load data
###############################################################################

dataset = PTBDataset(args.data)
ntokens = len(dataset.dictionary)

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

model = RNNModel('LSTM', ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tie_weights).to(device)

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(dataset.dictionary)
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
    ntokens = len(dataset.dictionary)
    hidden = model.init_hidden(args.batch_size, device)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        hidden = tuple(h.detach() for h in hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

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

# Loop over epochs.
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
            with open('weights.pth', 'wb') as f:
                torch.save(model.state_dict(), f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen.
            lr /= 1.2
except KeyboardInterrupt:
    print('Exiting from training early')

# Load the best saved model.
with open('weights.pth', 'rb') as f:
    model.load_state_dict(torch.load(f))

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
