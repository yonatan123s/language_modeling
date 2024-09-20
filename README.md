### Overview
The code provides an implementation of a two-layer LSTM language model with dropout, designed for word-level prediction tasks. It trains on the PTB dataset and aims to achieve perplexity scores close to those reported in the original paper for the large model (test perplexity around 78).

### Features
- **LSTM Model**: Two-layer LSTM with 1500 hidden units and 1500-dimensional embeddings.
- **Dropout Regularization**: Dropout applied to embeddings and between LSTM layers.
- **Backpropagation Through Time (BPTT)**: Sequence length of 35 time steps.
- **Gradient Clipping**: Prevents exploding gradients during training.
- **Learning Rate Annealing**: Reduces learning rate when validation loss plateaus.

**Modes of Operation**:
- **Training**: Train the model on the PTB training data.
- **Evaluation**: Evaluate the model's perplexity on train, validation, and test sets.
- **Interactive Testing**: Generate next-word predictions interactively.

### Usage

#### Training the Model
To train the model, run:

```
python main.py --mode train
```
Optional Arguments:

Specify a different data directory:

'python main.py --mode train --data path/to/ptbdataset'
Adjust hyperparameters (e.g., learning rate, number of epochs):


'python main.py --mode train --lr 1.0 --epochs 39'
Evaluating the Model
To evaluate the model's perplexity on the train, validation, and test datasets, run:


'python main.py --mode eval --checkpoint weights.pth'
Ensure that the --checkpoint argument points to the saved model weights.

Interactive Testing
To interactively test the model and get top-5 next-word predictions, run:


'python main.py --mode test --checkpoint weights.pth'
Enter a sequence of words when prompted, and the model will output the top-5 predicted next words with their probabilities.

Arguments and Hyperparameters
The script accepts several command-line arguments to configure the model and training process:

Data Parameters:

--data: Location of the data corpus (default: ../next_word_pred/ptbdataset).
Model Parameters:

--emsize: Size of word embeddings (default: 1500).
--nhid: Number of hidden units per LSTM layer (default: 1500).
--nlayers: Number of LSTM layers (default: 2).
--dropout: Dropout applied to layers (default: 0.65).
--tie_weights: Tie the word embedding and softmax weights.
Training Parameters:

--epochs: Upper epoch limit (default: 39).
--batch_size: Batch size (default: 20).
--bptt: Sequence length for BPTT (default: 35).
--lr: Initial learning rate (default: 1.0).
--clip: Gradient clipping threshold (default: 5.0).
--epochs_decay: Number of epochs before learning rate decay (default: 6).
Runtime Parameters:

--mode: Mode to run the script (train, eval, test).
--checkpoint: Path to save/load the model checkpoint (default: weights.pth).
Example Commands
Train the Model


'python main.py --mode train --data ptbdataset --epochs 39 --lr 1.0'
Evaluate the Model

'python main.py --mode eval --checkpoint weights.pth'
Interactive Testing

'python main.py --mode test --checkpoint weights.pth'
