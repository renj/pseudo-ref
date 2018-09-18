import time
import torch.optim as optim
import math
import torch
from torch.autograd import Variable
import sys
import loader
import Model
from torchtext import data
import utils
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')



if __name__ == '__main__':
    device = 3
    epochs = 60
    batch_size = 64
    save = 'LM'
    model_type = 'BiLangModel'
    data_set = './data/coco' # location of the data corpus
    rnn_model = 'LSTM'
    embed = 'glove.6B.300d' # location of the embedding file
    emsize = 300 # size of word embeddings
    nhid = 500 # humber of hidden units per layer
    nlayers = 1  # number of layers
    lr = 2.0 # initial learning rate
    weight_decay = 0.5 # weight decay
    dropout = 0.5
    vocab_size = 40000
    clip = 0.5 # gradient clipping
    slen = 20 # sequence length
    seed = 1111 # random seed
    fine_tune = False
    cuda = True # use CUDA
    log_interval = 800 # report interval
    param_init = 0.1

    print locals()

    def check_list(x):
        if type(x) is list:
            return x[0]
        return x

    #device, epochs, batch_size = check_list(device), check_list(epochs), check_list(batch_size)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)


    ###############################################################################
    # Load data
    ###############################################################################

    text_field = data.Field(lower=True)
    label_field = data.Field(lower=True)
    corpus = loader.Corpus(data_set, text_field, label_field, batch_size, max_size=vocab_size)
    #text_field.vocab.load_vectors(embed, wv_type='glove.6B2', wv_dim=200)
    #embed()
    #help(text_field.vocab.load_vectors)
    #text_field.vocab.load_vectors(embed_path, wv_dim=200)
    #text_field.vocab.load_vectors(embed_path)
    text_field.vocab.load_vectors(embed)
    #text_field.vocab.load_vectors('glove.6B.200d')

    '''
    def batchify(data, bsz):
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        data = data.view(bsz, -1).t().contiguous()
        if cuda:
            data = data.cuda()
        return data

    eval_batch_size = batch_size 
    train_data = batchify(corpus.train, batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)
    '''

    ###############################################################################
    # Build the model
    ###############################################################################

    #ntokens = len(corpus.dictionary)
    ntokens = len(label_field.vocab.itos)
    if model_type == 'LangModel':
        model = Model.LangModel(rnn_model, ntokens, emsize, nhid, nlayers, cuda, device)
    elif model_type == 'BiLangModel':
        model = Model.BiLangModel(rnn_model, ntokens, emsize, nhid, nlayers, dropout, cuda, device)
    if cuda:
        model.cuda(device)

    print(model)

    if param_init != 0.0:
        model.init_weights(param_init)
    model.init_encoder(text_field.vocab.vectors)

    print('voc size:', len(text_field.vocab.itos))
    #criterion = nn.CrossEntropyLoss()
    criterion = utils.LanguageModelCriterion()

    ###############################################################################
    # Training code
    ###############################################################################

    '''
    def clip_gradient(model, clip):
        """Computes a gradient clipping coefficient based on gradient norm."""
        totalnorm = 0
        for p in model.parameters():
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
        totalnorm = math.sqrt(totalnorm)
        return min(1, clip / (totalnorm + 1e-6))
    '''


    def clip_gradient(optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-grad_clip, grad_clip)

    def repackage_hidden(h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)


    def get_batch(source, i, evaluation=False):
        seq_len = min(slen, len(source) - 1 - i)
        data = Variable(source[i:i+seq_len], volatile=evaluation)
        target = Variable(source[i+1:i+1+seq_len].view(-1))
        return data, target

    def get_mask(feature):
        mask = torch.from_numpy(np.ones(feature.size())).float()
        n_zeros = 0
        for ix in range(mask.size(1)):
            _l = mask.size(0) - 1
            while feature[_l][ix].data is 1:
                _l -= 1
                mask[_l][ix] = 0
                n_zeros += 0
        mask = Variable(mask)
        mask = mask.cuda(model.device)
        assert(feature.data.sum() == (feature * mask.long()).data.sum() + n_zeros)
        return mask


    def evaluate(data_source):
        total_loss = 0
        model.eval()
        #ntokens = len(corpus.dictionary)
        #hidden = model.init_hidden(eval_batch_size)
        #for i in range(0, data_source.size(0) - 1, slen):
        for i, batch in enumerate(data_source):
            feature, targets = batch.text, batch.label
            if model_type.startswith('Bi'):
                targets = targets[1:-1, :]
            else:
                targets = targets[1:, :]
            if model.use_cuda:
                feature, targets = feature.cuda(model.device), targets.cuda(model.device)
            #data, targets = get_batch(data_source, i, evaluation=True)
            #output, hidden = model(data, hidden)
            output, hidden = model(feature)
            mask = get_mask(targets)
            #output_flat = output.view(-1, ntokens)
            #total_loss += len(data) * criterion(output_flat, targets, mask).data
            total_loss += criterion(output, targets, mask).data
            #hidden = repackage_hidden(hidden)
        model.train()
        return total_loss[0] / len(data_source)

    def train():
        total_loss = 0
        start_time = time.time()
        #ntokens = len(corpus.dictionary)
        #hidden = model.init_hidden(batch_size)
        #for batch, i in enumerate(range(0, train_data.size(0) - 1, slen)):
        #optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        if fine_tune:
            ignored_params = list(map(id, model.encoder.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            optimizer = optim.SGD(base_params, lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr)
        for i, batch in enumerate(corpus.train_iters):
            feature, targets = batch.text, batch.label
            #data, targets = get_batch(train_data, i)
            if model_type.startswith('Bi'):
                targets = targets[1:-1, :]
            else:
                targets = targets[1:, :]
                #feature = feature[:-1, :]
            if model.use_cuda:
                feature, targets = feature.cuda(model.device), targets.cuda(model.device)
            #hidden = repackage_hidden(hidden)
            #model.zero_grad()
            optimizer.zero_grad()
            # output: slen, batch, ntokens
            # feature: slen, batch
            #embed()
            mask = get_mask(targets)
            #output, hidden = model(feature, hidden)
            output, hidden = model(feature)
            #print output.size()
            #print targets.size()
            #loss = criterion(output.view(-1, ntokens), targets)
            #output = F.log_softmax(output)
            #embed()
            loss = criterion(output, targets, mask)
            loss.backward()

            #clipped_lr = lr * clip_gradient(model, clip)
            clip_gradient(optimizer, clip)
            optimizer.step()
            '''
            for p in model.parameters():
                p.data.add_(-clipped_lr, p.grad.data)
            '''

            total_loss += loss.data

            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss[0] / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | ' 'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, i, 2 // slen, lr,
                    elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()


    # Loop over epochs.
    prev_val_loss = None
    #save = 'REP_' + save[0]
    save = '%s_%s_%.1f_%s'%(save, data_set.split('/')[2], lr, model_type)
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        #embed()
        train()
        #val_loss = evaluate(val_data)
        val_loss = evaluate(corpus.valid_iters)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ' 'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Anneal the learning rate.
        if prev_val_loss and val_loss > prev_val_loss:
            lr *= weight_decay
        prev_val_loss = val_loss

        # Run on test data and save the model.
        test_loss = evaluate(corpus.test_iters)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
        if save != '':
            #if type(save) is list:
            #    save = save[0]
            _save = './model/%s_%d_%.2f.pth'%(save, epoch, math.exp(test_loss))
            with open(_save, 'wb') as f:
                torch.save(model, f)
