import torch
import torch.nn as nn


class VariationalDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, dropout):
        if self.training:
            # Create a dropout mask for the feature dimensions (batch_size, hidden_size)
            mask = torch.empty(input.size(1), input.size(2), device=input.device).bernoulli_(1 - dropout)
            mask = mask / (1 - dropout)  # Scale the mask to maintain the expected value
            mask = mask.expand_as(input)  # Expand the mask across the time dimension
            return mask * input
        else:
            return input


class RNNModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout_i=0.4, dropout_l=0.3, dropout_o=0.4, dropout_e=0.1,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        # Variational dropout
        self.variational_dropout = VariationalDropout()

        # Embedding layer
        self.encoder = nn.Embedding(ntoken, ninp)

        # LSTM layers
        self.rnn = nn.LSTM(ninp, nhid, nlayers)

        # Dropout parameters
        self.dropout_i = dropout_i  # Dropout on input word vectors
        self.dropout_l = dropout_l  # Dropout between LSTM layers
        self.dropout_o = dropout_o  # Dropout on the output of the last LSTM layer
        self.dropout_e = dropout_e  # Dropout on the embedding layer

        # Final output layer (decoder)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights between decoder and encoder
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When tying weights, nhid must be equal to ninp')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_seq, hidden):
        # Apply variational dropout to the embedding layer (dropout_e)
        emb = self.variational_dropout(self.encoder(input_seq), self.dropout_e)

        # Apply variational dropout to the word vectors (dropout_i) before feeding to LSTM
        emb = self.variational_dropout(emb, self.dropout_i)

        # Pass through the LSTM
        output, hidden = self.rnn(emb, hidden)

        # Apply variational dropout between LSTM layers (dropout_l)
        for i in range(self.nlayers - 1):
            output = self.variational_dropout(output, self.dropout_l)

        # Apply variational dropout to the output of the last LSTM layer (dropout_o)
        output = self.variational_dropout(output, self.dropout_o)

        # Reshape the output for the decoder
        output = output.reshape(output.size(0) * output.size(1), output.size(2))

        # Decode the output
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid).to(device),
                weight.new_zeros(self.nlayers, batch_size, self.nhid).to(device))
