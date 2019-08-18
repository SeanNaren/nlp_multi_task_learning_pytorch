import torch.nn as nn
from torch.autograd import Variable


class EncoderModel(nn.Module):
    """Model include a transducer to predict at each time steps"""

    def __init__(self, ntoken, emsize, nhid,
                 nlayers=1, dropout=0.2, rnn_type='LSTM', bi=False, pretrained_vectors=None, word_dict=None):
        super().__init__()
        print ("=== init EncoderModel ==")
        self.drop = nn.Dropout(dropout)
        print ("self.drop:", self.drop)
        self.embed = nn.Embedding(ntoken, emsize)
        print ("self.embed:", self.embed)
        self.rnn_type = rnn_type
        print ("self.rnn_type:", self.rnn_type)

        # Select RNN cell type from LSTM, GRU, and Elman
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(emsize, nhid, nlayers, bidirectional=bi)
            print ("self.rnn:", self.rnn)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(emsize, nhid, nlayers, bidirectional=bi)
            print ("self.rnn:", self.rnn)
        else:
            self.rnn = nn.RNN(emsize, nhid, nlayers, bidirectional=bi)
            print ("self.rnn:", self.rnn)
    
        print ("self.rnn:", self.rnn)
        self.init_weights()
        self.nhid = nhid
        print ("self.nhid:", self.nhid)
        self.nlayers = nlayers
        print ("self.nlayers:", self.nlayers)
        self.bi = bi
        print ("self.bi:", self.bi)
        if pretrained_vectors is not None:
            print ("pretrained_vectors is None")
            for x, word in enumerate(word_dict.idx2word):
                if word in pretrained_vectors.stoi:
                    pt_idx = pretrained_vectors.stoi[word]
                    #print ("pt_idx:", pt_idx)
                    self.embed.weight[x].data.copy_(pretrained_vectors.vectors[pt_idx])

    def init_weights(self):
        init_range = 0.1
        self.embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, input, hidden):
        embeded = self.drop(self.embed(input))
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(embeded, hidden)
        output = self.drop(output)
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()),
                Variable(weight.new(self.nlayers*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()))
                    

class LinearDecoder(nn.Module):
    """Linear decoder to decoder the outputs from the RNN Encoder.
        Then we can get the results of different tasks."""

    def __init__(self, nhid, ntags, bi=False):
        super().__init__()
        print ("== LinearDecoder init ==")
        self.linear = nn.Linear(nhid*(1+int(bi)), ntags)
        print ("self.linear:", self.linear)
        self.init_weights()
        self.nin = nhid
        print ("self.nin:", self.nin)
        self.nout = ntags
        print ("self.nout:", self.nout)
        self.bi = bi
        print ("self.bi:", self.bi)
    
    def init_weights(self):
        init_range = 0.1
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, input):
        print ("=== LinearDecoder === ")
        print ("input:", input.shape)
        logit = self.linear(input.view(input.size(0)*input.size(1), input.size(2)))
        print ("logit:", logit.shape)
        out = logit.view(input.size(0), input.size(1), logit.size(1))
        print ("out:", out.shape)
        return logit.view(input.size(0), input.size(1), logit.size(1))


class JointModel(nn.Module):
    """Joint Model to joint training two tasks.
       You can also only select one train mode to train one task.
       For args to specified the detail of training, include the task
       output and which layer we put it in. Number of tag first and 
       then number of layer."""

    def __init__(self, ntoken, emsize, nhid, *args,
                 dropout=0.2, rnn_type='LSTM', bi=False, train_mode='Joint', pretrained_vectors=None, vocab=None):
        super().__init__()
        print ("== JointModel init ==")
        self.ntoken = ntoken
        print ("self.ntoken:", self.ntoken)
        self.emsize = emsize
        print ("self.emsize:", self.emsize)
        self.nhid = nhid
        print ("self.nhid:", self.nhid)
        self.dropout = dropout
        print ("self.dropout:", self.dropout)
        self.rnn_type = rnn_type
        print ("self.rnn_type:", self.rnn_type)
        self.bi = bi
        print ("self.bi:", self.bi)
        self.train_mode = train_mode
        print ("self.train_mode:", self.train_mode)
        # According to train type, take arguments
        if train_mode == 'Joint':
            print ("Joint")
            self.ntags1 = args[0]
            self.nlayers1 = args[1]
            self.ntags2 = args[2]
            self.nlayers2 = args[3]
            self.ntags3 = args[4]
            self.nlayers3 = args[5]
            print ("self.ntags1:", self.ntags1)
            print ("self.ntags2:", self.ntags2)
            print ("self.ntags3:", self.ntags3)

            print ("self.nlayers1:", self.nlayers1)
            print ("self.nlayers2:", self.nlayers2)
            print ("self.nlayers3:", self.nlayers3)

            if self.nlayers1 == self.nlayers2 == self.nlayers3:
                self.rnn = EncoderModel(ntoken, emsize, nhid, self.nlayers1,
                                        dropout, rnn_type, bi, pretrained_vectors, vocab)
                print ("self.rnn:", self.rnn)
            else:
                print ("joint training, in different layers")
                print ("  self.nlayers1:", self.nlayers1)
                print ("  self.nlayers2:", self.nlayers2)
                print ("  self.nlayers3:", self.nlayers3)

                # Lower layer
                print ("rnn1 : Lower layer")
                print ("EncoderModel")
                self.rnn1 = EncoderModel(ntoken, emsize, nhid, self.nlayers1,
                                         dropout, rnn_type, bi, pretrained_vectors, vocab)
                print ("self.rnn1:", self.rnn1)

                # Higher layer 1
                print ("rnn2: Higher layer 1")
                print (rnn_type)
                if rnn_type == 'LSTM':
                    self.rnn2 = nn.LSTM(nhid * (1 + int(bi)), nhid,
                                        self.nlayers2 - self.nlayers1,
                                        bidirectional=bi)
                    print ("self.rnn2:", self.rnn2)
                elif rnn_type == 'GRU':
                    self.rnn2 = nn.GRU(nhid * (1 + int(bi)), nhid,
                                       self.nlayers2 - self.nlayers1,
                                       bidirectional=bi)
                    print ("self.rnn2:", self.rnn2)
                else:
                    self.rnn2 = nn.RNN(nhid * (1 + int(bi)), nhid,
                                       self.nlayers2 - self.nlayers1,
                                       bidirectional=bi)
                print ("self.rnn2:", self.rnn2)

                # Higher layer 2
                print ("rnn3: Higher layer 2")
                if rnn_type == 'LSTM':
                    self.rnn3 = nn.LSTM(nhid * (1 + int(bi)), nhid,
                                        self.nlayers3 - self.nlayers2,
                                        bidirectional=bi)
                elif rnn_type == 'GRU':
                    self.rnn3 = nn.GRU(nhid * (1 + int(bi)), nhid,
                                       self.nlayers3 - self.nlayers2,
                                       bidirectional=bi)
                else:
                    self.rnn3 = nn.RNN(nhid * (1 + int(bi)), nhid,
                                       self.nlayers3 - self.nlayers2,
                                       bidirectional=bi)
                print ("self.rnn3:", self.rnn3)

            # Decoders for three tasks
            print ("Decoders for two tasks")
            self.linear1 = LinearDecoder(nhid, self.ntags1, bi)
            print ("self.linear1:", self.linear1)

            self.linear2 = LinearDecoder(nhid, self.ntags2, bi)
            print ("self.linear2:", self.linear2)

            self.linear3 = LinearDecoder(nhid, self.ntags3, bi)
            print ("self.linear3:", self.linear3)

        else:
            self.ntags = args[0]
            print ("self.ntags:", self.ntags)
            self.nlayers = args[1]
            print ("self.nlayers:", self.nlayers)
            self.rnn = EncoderModel(ntoken, emsize, nhid, self.nlayers, 
                                    dropout, rnn_type, bi, pretrained_vectors, vocab)
            print ("self.rnn:", self.rnn)
            self.linear = LinearDecoder(nhid, self.ntags, bi)
            print ("self.linear:", self.linear)
            print ("== End JointModel init ==")
        
    def forward(self, input, *hidden):
        print ("=== JointModel fwd===")
        print ("input:", input.shape)
        print ("hidden layers:", len(hidden)) #hidden of all layers e.g 3
        if self.train_mode == 'Joint':
            print ("Joind mode")
            # when the number of layers is same, hidden layers are shared
            # and connected to different outputs
            if self.nlayers1 == self.nlayers2 == self.nlayers3:
                print ("same layer")
                print ("self.nlayers1:", self.nlayers1)
                print ("self.nlayers2:", self.nlayers2)
                print ("self.nlayers3:", self.nlayers3)
                print ("self.nlayers1 == self.nlayers2 == self.nlayers3")

                logits, shared_hidden = self.rnn(input, hidden[0]) #lower layer
                print ("logits:", logits.shape)
                print ("shared_hidden:", len(shared_hidden))

                outputs_pos = self.linear1(logits) #lower layer has output labels, pos-tags
                print ("outputs_pos:", outputs_pos.shape)

                outputs_chunk = self.linear2(logits) #same lower layer has output labels, chunk labels as well
                print ("outputs_chunk:", outputs_chunk.shape)

                outputs_ner = self.linear3(logits) #same lower layer has output labels, ner labels similar to the above two
                print ("outputs_ner:", outputs_ner.shape)

                print ("return output_pos, output_chunk, output_ner, hidden:", outputs_pos.shape, outputs_chunk.shape, outputs_ner.shape, len(hidden))
                return outputs_pos, outputs_chunk, outputs_ner, shared_hidden
            # cascading architecture where low-level tasks flow into high level
            else:
                print ("different layers")
                print ("(cascading architecture where low-level tasks flow into high level)")

                #for lstm, we have 3 outputs: output, (h_n, c_n)
                #see https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm

                # POS tagging task
                logits_pos, hidden_pos = self.rnn1(input, hidden[0])
                if (self.rnn_type == 'LSTM'):
                    hidden_pos_hn = hidden_pos[0]
                    hidden_pos_cn = hidden_pos[1]
                    print ("hidden_pos_hn:", hidden_pos_hn.shape) #h_n
                    print ("hidden_pos_cn:", hidden_pos_cn.shape) #c_n
                else:
                    print("hidden_pos:", hidden_pos.shape)
                print ("logits_pos:", logits_pos.shape)

                self.rnn2.flatten_parameters()
                print ("self.rnn2, flatten params:", self.rnn2)

                # chunking using POS
                logits_chunk, hidden_chunk = self.rnn2(logits_pos, hidden[1])
                if (self.rnn_type == 'LSTM'):
                    print ("hidden_chunk_hn:", hidden_chunk[0].shape) #h_n
                    print ("hidden_chunk_cn:", hidden_chunk[1].shape) #c_n
                else:
                    print("hidden_chunk:", hidden_pos.shape)
                print ("logits_chunk:", logits_chunk.shape)

                self.rnn3.flatten_parameters()
                print ("self.rnn3, flatten params:", self.rnn3)

                # NER using chunk
                logits_ner, hidden_ner = self.rnn3(logits_chunk, hidden[2])
                if (self.rnn_type == 'LSTM'):
                    print ("hidden_ner_hn:", hidden_ner[0].shape) #h_n
                    print ("hidden_ner_cn:", hidden_ner[1].shape) #c_n
                else:
                    print("hidden_ner:", hidden_pos.shape)
                print ("logits_ner:", logits_pos.shape)

                outputs_pos = self.linear1(logits_pos)
                print ("outputs_pos:", outputs_pos.shape)

                outputs_chunk = self.linear2(logits_chunk)
                print ("outputs_chunk:", outputs_chunk.shape)

                outputs_ner = self.linear3(logits_ner)
                print ("outputs_ner:", outputs_ner.shape)

                if (self.rnn_type == 'LSTM'):
                    print ("return output_pos, output_chunk, hidden_pos_h_n/c_n, hidden_chunk_hn/c_n, hidden_ner_hn/c_n:", outputs_pos.shape, outputs_chunk.shape, outputs_ner.shape, hidden_pos[0].shape, hidden_pos[1].shape, hidden_chunk[0].shape, hidden_chunk[1].shape, hidden_ner[0].shape, hidden_ner[1].shape)

                return outputs_pos, outputs_chunk, outputs_ner, hidden_pos, hidden_chunk, hidden_ner
        else:
            logits, hidden = self.rnn(input, hidden[0])
            if (self.rnn_type == 'LSTM'):
                print ("hidden:", hidden[0].shape, hidden[1].shape)
            else:
                print ("hidden:", hidden.shape)
            print ("logits:", logits.shape)

            outputs = self.linear(logits)
            print ("outputs:", outputs.shape)
            print ("return outputs, hidden:", outputs[0].shape, outputs[1].shape, hidden[0].shape, hidden[1].shape)
            return outputs, hidden

    def init_rnn2_hidden(self, batch_size):
        weight = next(self.rnn2.parameters()).data
        return (Variable(weight.new((self.nlayers2 - self.nlayers1)*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()),
                Variable(weight.new((self.nlayers2 - self.nlayers1)*(1+int(self.bi)), 
                                    batch_size, self.nhid).zero_()))

    def init_rnn3_hidden(self, batch_size):
        weight = next(self.rnn3.parameters()).data
        return (Variable(weight.new((self.nlayers3 - self.nlayers2) * (1 + int(self.bi)),
                                    batch_size, self.nhid).zero_()),
                Variable(weight.new((self.nlayers3 - self.nlayers2) * (1 + int(self.bi)),
                                    batch_size, self.nhid).zero_()))
