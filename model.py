import torch.nn as nn


class RNNModel(nn.Module):
    
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.65, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights 
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
        emb = self.drop(self.encoder(input_seq))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output = output.reshape(output.size(0) * output.size(1), output.size(2))
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, batch_size, self.nhid).to(device),
                weight.new_zeros(self.nlayers, batch_size, self.nhid).to(device))
