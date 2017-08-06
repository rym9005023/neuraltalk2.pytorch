from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import numpy as np
import torch.nn.init as init

from torch.nn import Parameter
from functools import wraps

from misc.weight_norm import WeightNorm


class convBlock(nn.Module):
    def __init__(self, rnn_size):
        super(convBlock, self).__init__()
        self.conv1 = WeightNorm(nn.Conv1d(rnn_size, rnn_size*2, 3, stride=1, padding=2))
        #self.fc0 = nn.Linear(rnn_size, rnn_size)
        self.conv2 = (nn.Conv2d(rnn_size*2, int(rnn_size/4), 1, stride=1, padding=0))
        self.conv3 = (nn.Conv2d(int(rnn_size/4), int(rnn_size/4), 3, stride=1, padding=1))
        self.conv4 = (nn.Conv2d(int(rnn_size/4), int(rnn_size), 1, stride=1, padding=0))
        #self.bn1 = nn.InstanceNorm1d(rnn_size, affine=True)
        #self.bn2 = nn.InstanceNorm1d(rnn_size, affine=True)

    def forward(self, sequence, fc, conv_feat):
        f = self.conv1(sequence)
        f = f[:,:,:-2]
        f1, f2 = torch.chunk(f, 2, dim=1)
        f2 = F.sigmoid(f2)
        f = torch.mul(f1, f2)
        #f = self.bn1(f)


        #f = self.conv2(f)
        #fc = self.fc0(fc)
        #fc = F.softmax(fc)

        #fc = fc.unsqueeze(2)
        #f = f*fc.expand_as(f)
        #f = self.bn2(f)
        f = F.relu(f, inplace=True)
        f = f + sequence
        f = F.dropout(f, p=0.2)

        f_last = f[:,:,-1]

        f_last = f_last.unsqueeze(2).unsqueeze(3)

        f_last = f_last.expand_as(conv_feat)
        conv_feat = torch.cat([f_last, conv_feat], 1)

        conv_feat = self.conv2(conv_feat)
        conv_feat = F.relu(conv_feat, inplace=True)

        conv_feat = self.conv3(conv_feat)
        conv_feat = F.relu(conv_feat, inplace=True)

        conv_feat = self.conv4(conv_feat)
        conv_feat = F.relu(conv_feat, inplace=True)

        return f, conv_feat

class convModel(nn.Module):
    def __init__(self, rnn_size):
        super(convModel, self).__init__()
        self.Block1 = convBlock(rnn_size)
        self.Block2 = convBlock(rnn_size)
        self.Block3 = convBlock(rnn_size)
        self.Block4 = convBlock(rnn_size)
        self.Block5 = convBlock(rnn_size)
        self.att_trans = nn.Sequential(nn.Linear(rnn_size, rnn_size),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5))
        self.alpha = nn.Linear(rnn_size, 1)
        self.beta = nn.Linear(rnn_size, 1)
        self.rnn_size = rnn_size
        #self.Block6 = convBlock(rnn_size)

    def forward(self, f, fc, att_feats):
        f, att_feats = self.Block1(f, fc, att_feats)
        f, att_feats = self.Block2(f, fc, att_feats)
        f, att_feats = self.Block3(f, fc, att_feats)
        f, att_feats = self.Block4(f, fc, att_feats)
        f, att_feats = self.Block5(f, fc, att_feats)
        #f = self.Block6(f, fc)

        att_attention = att_feats.view(att_feats.size(0), self.rnn_size, -1).transpose(1,2).contiguous().view(-1, self.rnn_size)
        att_attention = self.att_trans(att_attention)
        att_attention = F.tanh(att_attention)
        att_attention = self.alpha(att_attention)
        att_attention = att_attention.view(att_feats.size(0), 49)

        f_last = f[:,:,-1]

        f_att = self.beta(f_last)

        att_attention = torch.cat([att_attention, f_att],1)

        att_attention = F.softmax(att_attention)
        att_attention = att_attention.view(att_feats.size(0), 1, -1)
        att_feats = att_feats.view(att_feats.size(0), self.rnn_size, -1)

        att_feats = torch.cat([att_feats, f_last.unsqueeze(2)], 2)

        att_attention = att_attention.expand_as(att_feats)
        att_feats = att_attention * att_feats

        return att_feats.mean(2).squeeze(2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight)
    elif classname.find('Linear') != -1:
        init.xavier_uniform(m.weight)

class DFConvS(nn.Module):
    def __init__(self, opt):
        super(DFConvS, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.rnn_size),
                                nn.ReLU(True),
                                nn.Dropout(self.drop_prob_lm))
        self.pos_embed = nn.Embedding(20, self.rnn_size)
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(True),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(True),
                                    nn.Dropout(self.drop_prob_lm))


        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)

        self.attout_embed = nn.Sequential(nn.Linear(self.rnn_size, self.rnn_size),
                                    nn.ReLU(True),
                                    nn.Dropout(self.drop_prob_lm))

        self.convModule = convModel(self.rnn_size)
        #self.convModule.apply(weights_init)

        self.w2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.i2h = nn.Linear(self.rnn_size, self.rnn_size)



    def core(self, fc_feats, att_feats, output_encode, xt):
        #print(output_encode.size())
        #output_encode = output_encode.view(-1, self.vocab_size + 1)
        #print(output_encode.size())
        #output_encode = self.embed(output_encode)
        #output_encode = output_encode.view(fc_feats.size(0), -1, self.rnn_size).transpose((0,2,1))

        #padd = np.zeros((fc_feats.size(0), self.rnn_size, self.seq_length - output_encode.size(2)), np.float32)
        #padd = Variable(torch.from_numpy(padd).cuda(), requires_grad=False)
        if xt is not None:
            xt = self.w2h(fc_feats) + self.i2h(xt)
            output_encode = torch.cat([output_encode, xt.unsqueeze(2)], 2)
        att_feats = att_feats.transpose(1,2).contiguous().view(fc_feats.size(0), self.rnn_size, 7 ,7)



        decoder_out = self.convModule(output_encode[:,:,1:], fc_feats, att_feats)
        #print(decoder_out.size(), att_feats.size())
        #att_matrix = torch.bmm(att_feats, decoder_out)

        return self.attout_embed(decoder_out), output_encode

    def fast_core(self, fc_feats, att_feats, output_encode):
        #print(output_encode.size())
        #output_encode = output_encode.view(-1, self.vocab_size + 1)
        #print(output_encode.size())
        #output_encode = self.embed(output_encode)
        #output_encode = output_encode.view(fc_feats.size(0), -1, self.rnn_size).transpose((0,2,1))

        #padd = np.zeros((fc_feats.size(0), self.rnn_size, self.seq_length - output_encode.size(2)), np.float32)
        #padd = Variable(torch.from_numpy(padd).cuda(), requires_grad=False)
        #if xt is not None:
        # output_encode = torch.cat([output_encode, xt.unsqueeze(2)], 2)

        output_encode = output_encode[:,:,1:]


        decoder_out = self.convModule(output_encode, fc_feats)
        #print(decoder_out.size(), att_feats.size())

        for i in range(decoder_out.size(2)):
            sentence_att = decoder_out[:,:,i]

            lan_em = self.beta(sentence_att)

            sentence_att = sentence_att.unsqueeze(1)

            att_em = att_feats.view(-1, self.rnn_size)

            att_em = self.att_transA(att_em) + self.att_transB(sentence_att.expand_as(att_feats).contiguous().view(-1, self.rnn_size))
            att_em = F.tanh(att_em)

            att_em = self.alpha(att_em).view(att_feats.size(0), -1)

            attention = torch.cat([att_em, lan_em], 1)
            attention = F.softmax(attention)



            att_feats_extra = torch.cat([att_feats, sentence_att], 1)
            spatial_att = torch.bmm(attention.unsqueeze(1), att_feats_extra)
            spatial_att = spatial_att.view(fc_feats.size(0), fc_feats.size(1))

            spatial_att = self.attout_embed(spatial_att)

            if i == 0:
                attention_out = spatial_att
                attention_out = attention_out.unsqueeze(1)
            else:
                attention_out = torch.cat([attention_out, spatial_att.unsqueeze(1)],1)  

        return attention_out

    def _forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        #state = self.init_hidden(batch_size)
        output_encode = np.zeros((batch_size, self.rnn_size, 1), np.float32)
        output_encode = Variable(torch.from_numpy(output_encode).cuda(), requires_grad=False)
        #output_encode = []

        outputs = []
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(att_feats.size(0), -1 ,self.rnn_size)


        for i in range(seq.size(1)-1):
            it = seq[:, i].clone()  
            #if i == 0:
            #    print(it.sum())            
            if i >= 1 and seq[:, i].data.sum() == 0:
                break
            pos = Variable(fc_feats.data.new(batch_size).long().fill_(i), requires_grad=False)
            xt = self.embed(it) + self.pos_embed(pos)
            xt = self.w2h(fc_feats) + self.i2h(xt)
            output_encode = torch.cat([output_encode, xt.unsqueeze(2)], 2)

        output = self.fast_core(fc_feats, att_feats, output_encode)
        #print(output.size())
        tp = output.view(-1, self.rnn_size)           
        tp = F.log_softmax(self.logit(tp))
        return tp.view(output.size(0), output.size(1), -1)

    def forward(self, fc_feats, att_feats, seq):
        batch_size = fc_feats.size(0)
        #state = self.init_hidden(batch_size)
        output_encode = np.zeros((batch_size, self.rnn_size, 1), np.float32)
        output_encode = Variable(torch.from_numpy(output_encode).cuda(), requires_grad=False)
        #output_encode = []

        outputs = []
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(att_feats.size(0), -1 ,self.rnn_size)


        for i in range(seq.size(1)-1):

            #p = np.random.randint(10)

            #if p < 3:


            it = seq[:, i].clone()  
            #if i == 0:
            #    print(it.sum())              
            if i >= 1 and seq[:, i].data.sum() == 0:
                break
            pos = Variable(fc_feats.data.new(batch_size).long().fill_(i), requires_grad=False)
            xt = self.embed(it) + self.pos_embed(pos)
            #xt = self.w2h(fc_feats) + self.i2h(xt)
            #output_encode = torch.cat([output_encode, xt.unsqueeze(2)], 2)

            output, output_encode = self.core(fc_feats, att_feats, output_encode, xt) 
            output = F.log_softmax(self.logit(output))
            outputs.append(output)
            sampleLogprobs, it = torch.max(output.data, 1)

        #print('data = ', seq[:, 5].data[0], seq[:, 6].data[0], it.cpu().numpy()[0])

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)


    def sample_beam(self, fc_feats, att_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(att_feats.size(0), -1 ,self.rnn_size)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            output_encode = np.zeros((beam_size, self.rnn_size, 1), np.float32)
            output_encode = Variable(torch.from_numpy(output_encode).cuda(), requires_grad=False)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
            done_beams = []
            for t in range(self.seq_length + 1):
                if t == 0: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                else:
                    """pem a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float() # lets go to CPU for more efficiency in indexing operations
                    ys,ix = torch.sort(logprobsf,1,True) # sorted array of logprobs along each previous beam (last true = descending)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 1:  # at first time step only the first beam is active
                        rows = 1
                    for c in range(cols):
                        for q in range(rows):
                            # compute logprob of expanding beam q with word in (sorted) position c
                            local_logprob = ys[q,c]
                            candidate_logprob = beam_logprobs_sum[q] + local_logprob
                            candidates.append({'c':ix.data[q,c], 'q':q, 'p':candidate_logprob.data[0], 'r':local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])

                    # construct new beams
                    new_state = output_encode.clone()
                    if t > 1:
                        # well need these as reference when we fork beams around
                        beam_seq_prev = beam_seq[:t-1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t-1].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        # fork beam index q into index vix
                        if t > 1:
                            beam_seq[:t-1, vix] = beam_seq_prev[:, v['q']]
                            beam_seq_logprobs[:t-1, vix] = beam_seq_logprobs_prev[:, v['q']]

                        # rearrange recurrent states
                        new_state[vix] = output_encode[v['q']]
                        #for state_ix in range(len(new_state)):
                            # copy over state in previous beam q to new beam at vix
                        #    new_state[state_ix][0, vix] = state[state_ix][0, v['q']] # dimension one is time step

                        # append new end terminal at the end of this beam
                        beam_seq[t-1, vix] = v['c'] # c'th word is the continuation
                        beam_seq_logprobs[t-1, vix] = v['r'] # the raw logprob here
                        beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam

                        if v['c'] == 0 or t == self.seq_length:
                            # END token special case here, or we reached the end.
                            # add the beam to a set of done beams
                            self.done_beams[k].append({'seq': beam_seq[:, vix].clone(), 
                                                'logps': beam_seq_logprobs[:, vix].clone(),
                                                'p': beam_logprobs_sum[vix]
                                                })
        
                    # encode as vectors
                    it = beam_seq[t-1]
                    it = it.cuda()
                pos = Variable(fc_feats.data.new(beam_size).long().fill_(t), requires_grad=False)
                xt = self.embed(Variable(it, requires_grad=False)) + self.pos_embed(pos)
                
                if t >= 1:
                    output_encode = new_state

                output, output_encode = self.core(tmp_fc_feats, tmp_att_feats, output_encode, xt)     
                logprobs = F.log_softmax(self.logit(output.squeeze(0)))

            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)



    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            #print('####### NOT support BEAM SEARCH #######')
            return self.sample_beam(fc_feats, att_feats, opt)
        xt = None

        fc_feats = self.fc_embed(fc_feats)
        _att_feats = self.att_embed(att_feats.view(-1, self.att_feat_size))
        att_feats = _att_feats.view(att_feats.size(0), -1 ,self.rnn_size)

        batch_size = fc_feats.size(0)
        output_encode = np.zeros((batch_size, self.rnn_size, 1), np.float32)
        output_encode = Variable(torch.from_numpy(output_encode).cuda(), requires_grad=False)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t > 0:
                    sampleLogprobs, it = torch.max(logprobs.data, 1)
                    it = it.view(-1).long()
            if t == 0:
                it = fc_feats.data.new(batch_size).long().zero_()
            pos = Variable(fc_feats.data.new(batch_size).long().fill_(t), requires_grad=False)
            xt = self.embed(Variable(it, requires_grad=False)) + self.pos_embed(pos)

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            #tp = Variable(torch.from_numpy(output_encode).cuda(), requires_grad=False)
            output, output_encode = self.core(fc_feats, att_feats, output_encode, xt)     
            logprobs = F.log_softmax(self.logit(output.squeeze(0)))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)