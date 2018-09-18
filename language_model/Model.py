import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from IPython import embed

class LangModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, cuda=True, device=0):
        super(LangModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, bias=False)
        self.decoder = nn.Linear(nhid, ntoken)
        self.use_cuda = cuda
        self.device = device

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_encoder(self, init_matrix):
        if self.use_cuda:
            self.encoder.weight = nn.Parameter(torch.FloatTensor(init_matrix).cuda(self.device))
            #self.embed_weight = self.embed.weight
        else:
            self.encoder.weight = nn.Parameter(torch.FloatTensor(init_matrix))

    def forward(self, input, hidden=None):
        input = input[:-1, :]
        emb = self.encoder(input)
        #output, hidden = self.rnn(emb, hidden)
        output, hidden = self.rnn(emb)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        output = decoded.view(output.size(0), output.size(1), decoded.size(1))
        outputs = [F.log_softmax(output[:, _i,:]).view(output.size(0), 1, output.size(2)) for _i in range(output.size(1))]
        output = torch.cat(outputs, dim=1)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class BiLangModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout, cuda=True, device=0):
        super(BiLangModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp) # encoder == embedding
        #self.fw_rnn = nn.LSTM(ninp, nhid, nlayers, bias=False)
        #self.bw_rnn = nn.LSTM(ninp, nhid, nlayers, bias=False)
        #self.rnn = nn.LSTM(ninp, nhid, nlayers, bidirectional=True, bias=False, dropout=dropout)
        self.rnn = nn.LSTM(ninp, nhid, 1, bidirectional=True, dropout=dropout)
        self.decoder = nn.Linear(nhid * 2, ntoken)

        self.use_cuda = cuda
        self.device = device

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        #self.init_weights()

    def init_weights(self, param_init=0.1):
        #initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        #self.decoder.bias.data.fill_(0)
        #self.decoder.weight.data.uniform_(-initrange, initrange)
        for p in self.parameters():
            p.data.uniform_(-param_init, param_init)

    def init_encoder(self, init_matrix):
        if self.use_cuda:
            self.encoder.weight = nn.Parameter(torch.FloatTensor(init_matrix).cuda(self.device))
            #self.embed_weight = self.encoder.weight
        else:
            self.encoder.weight = nn.Parameter(torch.FloatTensor(init_matrix))

    def forward(self, input, hidden=None):
        #fw_emb = self.encoder(input[:-2, :])
        #bw_emb = self.encoder(input[2::-1, :])
        emb = self.encoder(input)
        #output, hidden = self.rnn(emb, hidden)
        #fw_output, _hidden = self.fw_rnn(fw_emb)
        #bw_output, _hidden = self.bw_rnn(bw_emb)
        output, hidden = self.rnn(emb)
        fw_output = output[:-2, :, :self.nhid]
        bw_output = output[2:, :, self.nhid:]
        #fw_output = output[2:, :, :200]
        #bw_output = output[:-2, :, 200:]
        #embed()
        output = torch.cat([fw_output, bw_output], dim=2)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        output = decoded.view(output.size(0), output.size(1), decoded.size(1))
        outputs = [F.log_softmax(output[:, _i,:]).view(output.size(0), 1, output.size(2)) for _i in range(output.size(1))]
        output = torch.cat(outputs, dim=1)
        return output, hidden

    '''
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
    '''
    