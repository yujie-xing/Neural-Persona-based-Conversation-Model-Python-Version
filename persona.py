from params import params
from data import data
from io import open
import string
import numpy as np
import pickle
import linecache
import math

import torch
import torch.nn as nn
from torch.autograd import Variable, backward


class attention_feed(nn.Module):
    
    def __init__(self,params):
        super(attention_feed, self).__init__()
        self.params=params
    
    def forward(self,target_t,context,context_mask):
        context_mask_p=(context_mask-1)*100000000
        atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)
        atten=atten+context_mask_p
        atten=nn.Softmax(dim=1)(atten)
        atten=atten.unsqueeze(1)
        context_combined=torch.bmm(atten,context).sum(1)
        return context_combined

class softattention(nn.Module):
    
    def __init__(self,params):
        super(softattention, self).__init__()
        self.params=params
        self.attlinear1=nn.Linear(self.params.dimension,self.params.dimension,False)
        self.attlinear2=nn.Linear(self.params.dimension,self.params.dimension,False)
    
    def forward(self,target_t,context,context_mask):
        context_mask_p=(context_mask-1)*100000000
        atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)
        atten=atten+context_mask_p
        atten=nn.Softmax(dim=1)(atten)
        atten=atten.unsqueeze(1)
        context_combined=torch.bmm(atten,context).sum(1)
        output1=self.attlinear1(context_combined)
        output2=self.attlinear2(target_t)
        output=nn.Tanh()(output1+output2)
        return output


class lstm_source_(nn.Module):
    
    def __init__(self,params):
        super(lstm_source_, self).__init__()
        self.params=params
        self.sembedding=nn.Embedding(self.params.vocab_source,self.params.dimension,padding_idx=self.params.vocab_dummy)
        self.sdropout=nn.Dropout(self.params.dropout)
        for num in range(1,self.params.layers*2+1):
            setattr(self,"slinear"+str(num),nn.Linear(self.params.dimension,4*self.params.dimension,False))

    def forward(self,inputs):
        outputs = []
        for ll in range(self.params.layers):
            prev_h=inputs[ll*2]
            prev_c=inputs[ll*2+1]
            if ll==0:
                x=self.sembedding(inputs[-1])
            else:
                x=outputs[ll*2-2]
            drop_x=self.sdropout(x)
            drop_h=self.sdropout(inputs[ll*2])
            i2h=getattr(self,"slinear"+str(ll*2+1))(drop_x)
            h2h=getattr(self,"slinear"+str(ll*2+2))(drop_h)
            gates=i2h+h2h
            reshaped_gates=gates.view(-1,4,self.params.dimension)
            in_gate= nn.Sigmoid()(reshaped_gates[:,0])
            in_transform= nn.Tanh()(reshaped_gates[:,1])
            forget_gate= nn.Sigmoid()(reshaped_gates[:,2])
            out_gate= nn.Sigmoid()(reshaped_gates[:,3])
            l1=forget_gate*inputs[ll*2+1]
            l2=in_gate*in_transform
            next_c=l1+l2
            next_h= out_gate*(nn.Tanh()(next_c))
            outputs.append(next_h)
            outputs.append(next_c)
        return outputs
        

class lstm_target_(nn.Module):
    
    def __init__(self,params):
        super(lstm_target_, self).__init__()
        self.params=params
        self.embedding=nn.Embedding(self.params.vocab_target,self.params.dimension,padding_idx=self.params.vocab_dummy)
        if self.params.PersonaMode:
            self.speaker_embedding=nn.Embedding(self.params.SpeakerNum,self.params.dimension)
        self.dropout=nn.Dropout(self.params.dropout)
        self.linear=nn.Linear(self.params.dimension,4*self.params.dimension,False)
        if self.params.PersonaMode:
            self.linear_v=nn.Linear(self.params.dimension,4*self.params.dimension,False)
        for num in range(1,self.params.layers*2+1):
            setattr(self,"linear"+str(num),nn.Linear(self.params.dimension,4*self.params.dimension,False))
        self.atten_feed=attention_feed(self.params)
        self.soft_atten=softattention(self.params)
        
    def forward(self,inputs):
        context=inputs[self.params.layers*2]
        x_=inputs[self.params.layers*2+1]
        source_mask=inputs[self.params.layers*2+2]
        outputs=[]
        for ll in range(self.params.layers):
            prev_h=inputs[ll*2]
            prev_c=inputs[ll*2+1]
            if ll==0:
                x=self.embedding(x_)
            else:
                x=outputs[ll*2-2]
            drop_x=self.dropout(x)
            drop_h=self.dropout(inputs[ll*2])
            i2h=getattr(self,"linear"+str(ll*2+1))(drop_x)
            h2h=getattr(self,"linear"+str(ll*2+2))(drop_h)
            if ll==0:
                context1=self.atten_feed(inputs[self.params.layers*2-2],context,source_mask)
                drop_f=self.dropout(context1)
                f2h=self.linear(drop_f)
                gates=(i2h+h2h)+f2h
                if self.params.PersonaMode:
                    speaker_index=inputs[self.params.layers*2+3]
                    speaker_v=self.speaker_embedding(speaker_index)
                    speaker_v=self.dropout(speaker_v)
                    v=self.linear_v(speaker_v)
                    gates=gates+v
            else:
                gates=i2h+h2h
            reshaped_gates = gates.view(-1,4,self.params.dimension)
            in_gate=nn.Sigmoid()(reshaped_gates[:,0])
            in_transform= nn.Tanh()(reshaped_gates[:,1])
            forget_gate=nn.Sigmoid()(reshaped_gates[:,2])
            out_gate=nn.Sigmoid()(reshaped_gates[:,3])
            l1=forget_gate*inputs[ll*2+1]
            l2=in_gate*in_transform
            next_c=l1+l2
            next_h= out_gate*(nn.Tanh()(next_c))
            outputs.append(next_h)
            outputs.append(next_c)
        soft_vector=self.soft_atten(outputs[self.params.layers*2-2],context,source_mask)
        outputs.append(soft_vector)
        return outputs


class softmax_(nn.Module):
    
    def __init__(self,params):
        super(softmax_, self).__init__()
        self.params=params
        self.softlinear=nn.Linear(self.params.dimension,self.params.vocab_target,False)
        
    def forward(self,h,y):
        h2y= self.softlinear(h)
        pred= nn.LogSoftmax(dim=1)(h2y)
        if self.params.use_GPU:
            w=torch.ones(self.params.vocab_target).cuda()
        else:
            w=torch.ones(self.params.vocab_target)
        w[self.params.vocab_dummy]=0
        Criterion=nn.NLLLoss(w,size_average=False,ignore_index=self.params.vocab_dummy)
        err=Criterion(pred, y)
        return err,pred


class persona:
    
    def __init__(self, params):
        self.Data=data(params)
        self.params=params

        self.lstm_source =lstm_source_(self.params)
        self.lstm_target =lstm_target_(self.params)
        self.lstm_source.apply(self.weights_init)
        self.lstm_target.apply(self.weights_init)
        embed=list(self.lstm_source.parameters())[0]
        embed[self.params.vocab_dummy].data.fill_(0)
        embed=list(self.lstm_target.parameters())[0]
        embed[self.params.vocab_dummy].data.fill_(0)
        if self.params.use_GPU:
            self.lstm_source=self.lstm_source.cuda()
            self.lstm_target=self.lstm_target.cuda()
        self.softmax=softmax_(self.params)
        self.softmax.apply(self.weights_init)
        if self.params.use_GPU:
            self.softmax=self.softmax.cuda()
        self.output=self.params.output_file
        if self.output!="":
            with open(self.output,"w") as selfoutput:
                selfoutput.write("")
        if self.params.PersonaMode:
            print("training in persona mode")
        else:
            print("training in non persona mode")
        self.ReadDict()


    def weights_init(self,module):
        classname=module.__class__.__name__
        try:
            module.weight.data.uniform_(-self.params.init_weight,self.params.init_weight)
        except:
            pass
    
    def ReadDict(self):
        self.dict={}
        dictionary=open(self.params.train_path+self.params.dictPath,"r").readlines()
        for index in range(len(dictionary)):
            line=dictionary[index].strip()
            self.dict[index]=line

    def IndexToWord(self,vector):
        if vector.dim()==0:
            vector=vector.view(1)
        string=""
        for i in range(vector.size(0)):
            try:
                string=string+self.dict[int(vector[i])]+" "
            except KeyError: 
                string=string+str(int(vector[i]))+" "
        string=string.strip()
        return string

    def model_forward(self):
        self.context=Variable(torch.Tensor(self.Word_s.size(0),self.Word_s.size(1),self.params.dimension))
        if self.params.use_GPU:
            self.context=self.context.cuda()
        for t in range(self.Word_s.size(1)):
            inputs=[]
            if t==0:
                for ll in range(self.params.layers):
                    if self.params.use_GPU:
                        inputs.append(Variable(torch.zeros(self.Word_s.size(0),self.params.dimension).cuda()))
                        inputs.append(Variable(torch.zeros(self.Word_s.size(0),self.params.dimension).cuda()))
                    else:
                        inputs.append(Variable(torch.zeros(self.Word_s.size(0),self.params.dimension)))
                        inputs.append(Variable(torch.zeros(self.Word_s.size(0),self.params.dimension)))
            else:
                inputs=output
            inputs.append(self.Word_s[:,t])
            if self.mode=="train":
                self.lstm_source.train()
            else:
                self.lstm_source.eval()
            output=self.lstm_source(inputs)
            if t==self.Word_s.size(1)-1:
                self.last=output
            self.SourceVector=output[self.params.layers*2-2]
            self.context[:,t]=output[2*self.params.layers-2]
        if self.mode!="decoding":
            sum_err=0
            total_num=0
            for t in range(self.Word_t.size(1)-1):
                lstm_input=[]
                if t==0:
                    lstm_input=output
                else:
                    lstm_input=output[:-1]
                lstm_input.append(self.context)
                lstm_input.append(self.Word_t[:,t])
                lstm_input.append(self.Padding_s)
                if self.params.PersonaMode:
                    lstm_input.append(self.SpeakerID)
                if self.mode=="train":
                    self.lstm_target.train()
                else:
                    self.lstm_target.eval()
                output=self.lstm_target(lstm_input)
                current_word=self.Word_t[:,t+1]
                err,pred=self.softmax(output[-1],current_word)
                sum_err=sum_err+err
                total_num=total_num+self.Left_t[t+1].size(0)
            if self.mode=="train":
                sum_err.backward()
            return sum_err.data[0], total_num

    def test(self):
        if self.mode=="test":
            open_train_file=self.params.train_path+self.params.dev_file
        sum_err_all=0 
        total_num_all=0
        End=0
        batch_n=0
        while End==0:
            End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target,self.SpeakerID,self.AddresseeID=self.Data.read_train(open_train_file,batch_n)
            batch_n+=1
            if len(self.Word_s)==0 or End==1:
                break
            if (self.Word_s.size(1)<self.params.source_max_length and self.Word_t.size(1)<self.params.target_max_length):
                self.mode="test"
                self.Word_s=Variable(self.Word_s)
                self.Word_t=Variable(self.Word_t)
                self.Padding_s=Variable(self.Padding_s)
                self.SpeakerID=Variable(self.SpeakerID)
                if self.params.use_GPU:
                    self.Word_s=self.Word_s.cuda()
                    self.Word_t=self.Word_t.cuda()
                    self.Padding_s=self.Padding_s.cuda()
                    self.SpeakerID=self.SpeakerID.cuda()
                sum_err,total_num=self.model_forward()
                sum_err_all+=sum_err
                total_num_all+=total_num
        print("perp "+str((1/math.exp(-sum_err_all/total_num_all))))
        if self.output!="":
            with open(self.output,"a") as selfoutput:
                selfoutput.write("standard perp "+str((1/math.exp(-sum_err_all/total_num_all)))+"\n")

    def update(self):
        lr=self.lr
        grad_norm=0
        for module in [self.lstm_source,self.lstm_target,self.softmax]:
            for m in list(module.parameters()):
                m.grad.data = m.grad.data*(1/self.Word_s.size(0))
                grad_norm+=m.grad.data.norm()**2
        grad_norm=grad_norm**0.5
        if grad_norm>self.params.thres:
            lr=lr*self.params.thres/grad_norm
        for module in [self.lstm_source,self.lstm_target,self.softmax]:
            for f in module.parameters():
                f.data.sub_(f.grad.data * lr)

    def save(self):
        torch.save(self.lstm_source.state_dict(),self.params.save_prefix+str(self.iter)+"_source.pkl")
        torch.save(self.lstm_target.state_dict(),self.params.save_prefix+str(self.iter)+"_target.pkl")
        torch.save(self.softmax.state_dict(),self.params.save_prefix+str(self.iter)+"_softmax.pkl")
        print("finished saving")

    def saveParams(self):
        with open(self.params.save_params_file+".pickle","wb") as file:
            pickle.dump(self.params,file)

    def readModel(self):
        self.lstm_source.load_state_dict(torch.load(self.params.model_file+"_source.pkl"))
        self.lstm_target.load_state_dict(torch.load(self.params.model_file+"_target.pkl"))
        self.softmax.load_state_dict(torch.load(self.params.model_file+"_softmax.pkl"))
        print("read model done")

    def clear(self):
        for module in [self.lstm_source,self.lstm_target,self.softmax]:
            module.zero_grad()

    def train(self):
        if self.params.saveModel:
            self.saveParams()
        if self.params.fine_tuning:
            target_model=torch.load(self.params.fine_tuning_model+"_target.pkl")
            linear_v=self.lstm_target.state_dict()["linear_v.weight"]
            speaker=self.lstm_target.state_dict()["speaker_embedding.weight"]
            target_model["linear_v.weight"]=linear_v
            target_model["speaker_embedding.weight"]=speaker
            self.lstm_source.load_state_dict(torch.load(self.params.fine_tuning_model+"_source.pkl"))
            self.lstm_target.load_state_dict(target_model)
            self.softmax.load_state_dict(torch.load(self.params.fine_tuning_model+"_softmax.pkl"))
            print("read model done")
        self.iter=0
        start_halving=False
        self.lr=self.params.alpha
        print("iter  "+str(self.iter))
        self.mode="test"
        self.test()
        while True:
            self.iter+=1
            print("iter  "+str(self.iter))
            if self.output!="":
                with open(self.output,"a") as selfoutput:
                    selfoutput.write("iter  "+str(self.iter)+"\n")
            if self.params.start_halve!=-1:
                if self.iter>self.params.start_halve:
                    start_halving=True
            if start_halving:
                self.lr=self.lr*0.5
            open_train_file=self.params.train_path+self.params.train_file
            End=0
            batch_n=0
            while End==0:
                self.clear()
                End,self.Word_s,self.Word_t,self.Mask_s,self.Mask_t,self.Left_s,self.Left_t,self.Padding_s,self.Padding_t,self.Source,self.Target,self.SpeakerID,self.AddresseeID=self.Data.read_train(open_train_file,batch_n)
                batch_n+=1
                if End==1:
                    break
                train_this_batch=False
                if (self.Word_s.size(1)<60 and self.Word_t.size(1)<60):
                    train_this_batch=True
                if train_this_batch:
                    self.mode="train"
                    self.Word_s=Variable(self.Word_s)
                    self.Word_t=Variable(self.Word_t)
                    self.Padding_s=Variable(self.Padding_s)
                    self.SpeakerID=Variable(self.SpeakerID)
                    if self.params.use_GPU:
                        self.Word_s=self.Word_s.cuda()
                        self.Word_t=self.Word_t.cuda()
                        self.Padding_s=self.Padding_s.cuda()
                        self.SpeakerID=self.SpeakerID.cuda()
                    self.model_forward()
                    self.update()
            self.mode="test"
            self.test()
            if self.params.saveModel:
                self.save()
            if self.iter==self.params.max_iter:
                break