from decode_params import decode_params
from data import data
from persona import *
from io import open
import string
import numpy as np
import pickle
import linecache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, backward

class decode_model(persona):

    def __init__(self, params):
        with open(params.decode_path+"/params.pickle", 'rb') as file:
            model_params = pickle.load(file)
        for key in model_params.__dict__:
            if key not in params.__dict__:
                params.__dict__[key]=model_params.__dict__[key]
        self.params=params
        self.mode="decoding"
        if self.params.PersonaMode:
            print("decoding in persona mode")
        else:
            print("decoding in non persona mode")
        self.Data=data(self.params)
        self.lstm_source =lstm_source_(self.params)
        self.lstm_target =lstm_target_(self.params)
        self.softmax =softmax_(self.params)
        if self.params.use_GPU:
            self.lstm_source=self.lstm_source.cuda()
            self.lstm_target=self.lstm_target.cuda()
            self.softmax=self.softmax.cuda()
        self.readModel()
        self.ReadDict()

    def sample(self):
        self.model_forward()
        if self.params.max_length==0:
            batch_max_dec_length=torch.ceil(1.5*self.Word_s.size(1))
        else:
            batch_max_dec_length=self.params.max_length
        completed_history={}
        if self.params.use_GPU:
            beamHistory=torch.ones(self.Word_s.size(0),batch_max_dec_length).long().cuda()
        else:
            beamHistory=torch.ones(self.Word_s.size(0),batch_max_dec_length).long()
        for t in range(batch_max_dec_length):
            lstm_input=self.last
            lstm_input.append(self.context)
            if t==0:
                if self.params.use_GPU:
                    lstm_input.append(Variable(torch.LongTensor(self.Word_s.size(0)).fill_(self.Data.EOS).cuda()))
                else:
                    lstm_input.append(torch.LongTensor(self.Word_s.size(0)).fill_(self.Data.EOS))
            else:
                lstm_input.append(Variable(beamHistory[:,t-1]))
            lstm_input.append(self.Padding_s)
            if self.params.PersonaMode:
                lstm_input.append(self.SpeakerID)
            self.lstm_target.eval()
            output=self.lstm_target(lstm_input)
            self.last=output[:-1]
            if self.params.use_GPU:
                err,pred=self.softmax(output[-1],Variable(torch.LongTensor(output[-1].size(0)).fill_(1).cuda()))
            else:
                err,pred=self.softmax(output[-1],torch.LongTensor(output[-1].size(0)).fill_(1))
            prob=pred.data
            prob=torch.exp(prob)
            if not self.params.allowUNK:
                prob[:,0].fill_(0)
            if self.params.setting=="StochasticGreedy":
                select_P,select_words=torch.topk(prob,self.params.StochasticGreedyNum,1,True,True)
                prob=F.normalize(select_P, 1, dim=1)
                next_words_index=torch.multinomial(prob, 1)
                if self.params.use_GPU:
                    next_words=torch.Tensor(self.Word_s.size(0),1).fill_(0).cuda()
                else:
                    next_words=torch.Tensor(self.Word_s.size(0),1).fill_(0)
                for i in range(self.Word_s.size(0)):
                    next_words[i][0]=select_words[i][next_words_index[i][0]]
            elif self.params.setting=="sample":
                next_words=torch.multinomial(prob, 1)
            else:
                next_words=torch.max(prob,dim=1)[1]
            end_boolean_index=torch.eq(next_words,self.Data.EOT)
            if self.params.use_GPU:
                end_boolean_index=end_boolean_index.cuda()
            if end_boolean_index.sum()!=0:
                for i in range(end_boolean_index.size(0)):
                    if end_boolean_index[i][0]==1:
                        example_index=i
                        if example_index not in completed_history:
                            if t!=0:
                                completed_history[example_index]=beamHistory[example_index,:t]
                            else:
                                if self.params.use_GPU:
                                    completed_history[example_index]=torch.Tensor(1,1).fill_(0).cuda()
                                else:
                                    completed_history[example_index]=torch.Tensor(1,1).fill_(0)
            beamHistory[:,t]=next_words.view(-1)
        for i in range(self.Word_s.size(0)):
            if i not in completed_history:
                completed_history[i]=beamHistory[i,:]
        return completed_history


    def decode(self):
        open_train_file=self.params.train_path+self.params.DecodeFile
        speaker_id = self.params.SpeakerID
        if not self.params.PersonaMode:
            speaker_id=0
        output_file=self.params.OutputFolder+"/"+self.params.train_path.split("/")[-1]+"_s"+str(speaker_id)+"_"+self.params.DecodeFile[1:]
        with open(output_file,"w") as open_write_file:
            open_write_file.write("")
        End=0
        batch_n=0
        n_decode_instance=0
        while End==0:
            End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target,self.SpeakerID,self.AddresseID=self.Data.read_train(open_train_file,batch_n)
            if len(self.Word_s)==0:
                break
            n_decode_instance=n_decode_instance+self.Word_s.size(0)
            if self.params.max_decoded_num!=0 and n_decode_instance>self.params.max_decoded_num:
                break
            batch_n=batch_n+1
            self.mode="decoding"
            self.SpeakerID.fill_(self.params.SpeakerID-1)
            self.Word_s=Variable(self.Word_s)
            self.Padding_s=Variable(self.Padding_s)
            self.SpeakerID=Variable(self.SpeakerID)
            if self.params.use_GPU:
                self.Word_s=self.Word_s.cuda()
                self.Padding_s=self.Padding_s.cuda()
                self.SpeakerID=self.SpeakerID.cuda()
            completed_history=self.sample()
            self.OutPut(output_file,completed_history)
            if End==1:
                break
        print("decoding done")

    def OutPut(self,output_file,completed_history):
        for i in range(self.Word_s.size(0)):
            if self.params.output_source_target_side_by_side:
                print_string=self.IndexToWord(self.Source[i].view(-1))
                print_string=print_string+"|"
                print_string=print_string+self.IndexToWord(completed_history[i].view(-1))
                with open(output_file,"a") as file:
                    file.write(print_string+"\n")
            else:
                print_string=self.IndexToWord(completed_history[i].view(-1))
                with open(output_file,"a") as file:
                    file.write(print_string+"\n")

