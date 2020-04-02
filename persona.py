from data import data

from os import path
from io import open
import string
import numpy as np
import pickle
import linecache
import math

import torch
from torch import tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class attention_feed(nn.Module):
	
	def __init__(self):
		super(attention_feed, self).__init__()
	
	def forward(self,target_t,context):
		atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)
		mask=((atten!=0).float()-1)*1e9
		atten=atten+mask
		atten=nn.Softmax(dim=1)(atten)
		atten=atten.unsqueeze(1)
		context_combined=torch.bmm(atten,context).sum(1)
		return context_combined

class softattention(nn.Module):
	
	def __init__(self,params):
		super(softattention, self).__init__()
		dim = params.dimension
		self.attlinear=nn.Linear(dim*2,dim,False)
	
	def forward(self,target_t,context):
		atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)
		mask=((atten!=0).float()-1)*1e9
		atten=atten+mask
		atten=nn.Softmax(dim=1)(atten)
		atten=atten.unsqueeze(1)
		context_combined=torch.bmm(atten,context).sum(1)
		output=self.attlinear(torch.cat((context_combined,target_t),-1))
		output=nn.Tanh()(output)
		return output


class lstm_source(nn.Module):
	
	def __init__(self,params):
		super(lstm_source, self).__init__()
		dim = params.dimension
		layer = params.layers
		self.dropout = nn.Dropout(p=params.dropout)
		self.lstms=nn.LSTM(dim,dim,num_layers=layer,batch_first=True,bias=False,dropout=params.dropout)

	def forward(self,embedding,length):
		embedding = self.dropout(embedding)
		packed=pack_padded_sequence(embedding,length,batch_first=True,enforce_sorted=False)
		packed_output,(h,c)=self.lstms(packed)
		context,_= pad_packed_sequence(packed_output,batch_first=True)
		return context,h,c
		

class lstm_target(nn.Module):
	
	def __init__(self,params):
		super(lstm_target, self).__init__()
		
		dim = params.dimension
		layer = params.layers
		self.dropout = nn.Dropout(p=params.dropout)
		self.speaker = params.SpeakerMode
		self.addressee = params.AddresseeMode
		persona_num = params.PersonaNum
		
		if self.speaker:
			self.persona_embedding=nn.Embedding(persona_num,dim)
			self.lstmt=nn.LSTM(dim*3,dim,num_layers=layer,batch_first=True,bias=False,dropout=params.dropout)
		elif self.addressee:
			self.persona_embedding=nn.Embedding(persona_num,dim)
			self.speaker_linear = nn.Linear(dim,dim)
			self.addressee_linear = nn.Linear(dim,dim)
			self.lstmt=nn.LSTM(dim*3,dim,num_layers=layer,batch_first=True,bias=False,dropout=params.dropout)
		else:
			self.lstmt=nn.LSTM(dim*2,dim,num_layers=layer,batch_first=True,bias=False,dropout=params.dropout)
		self.atten_feed=attention_feed()
		self.soft_atten=softattention(params)
		
	def forward(self,context,h,c,embedding,speaker_label,addressee_label):
		embedding = self.dropout(embedding)
		h = self.dropout(h)
		context1=self.atten_feed(h[-1],context)
		context1 = self.dropout(context1)
		lstm_input=torch.cat((embedding,context1),-1)
		if self.speaker:
			speaker_embed=self.persona_embedding(speaker_label)
			speaker_embed = self.dropout(speaker_embed)
			lstm_input=torch.cat((lstm_input,speaker_embed),-1)
		elif self.addressee:
			speaker_embed=self.persona_embedding(speaker_label)
			speaker_embed = self.dropout(speaker_embed)
			addressee_embed=self.persona_embedding(addressee_label)
			addressee_embed = self.dropout(addressee_embed)
			combined_embed = self.speaker_linear(speaker_embed) + self.addressee_linear(addressee_embed)
			combined_embed = nn.Tanh()(combined_embed)
			lstm_input=torch.cat((lstm_input,combined_embed),-1)
		_,(h,c)=self.lstmt(lstm_input.unsqueeze(1),(h,c))
		pred=self.soft_atten(h[-1],context)
		return pred,h,c


class lstm(nn.Module):

	def __init__(self,params,vocab_num,EOT):
		super(lstm, self).__init__()
		dim = params.dimension
		init_weight = params.init_weight
		vocab_num = vocab_num + params.special_word
		self.UNK = params.UNK+params.special_word
		self.EOT = EOT
		self.params=params
		
		self.encoder=lstm_source(params)
		self.decoder=lstm_target(params)
		
		self.sembed=nn.Embedding(vocab_num,dim,padding_idx=0)
		self.sembed.weight.data[1:].uniform_(-init_weight,init_weight)
		self.tembed=nn.Embedding(vocab_num,dim,padding_idx=0)
		self.tembed.weight.data[1:].uniform_(-init_weight,init_weight)
		
		self.softlinear=nn.Linear(dim,vocab_num,False)
		w=torch.ones(vocab_num)
		if not params.cpu:
			w = w.cuda()
		w[:2]=0
		w[self.UNK]=0
		self.loss_function=torch.nn.CrossEntropyLoss(w, ignore_index=0, reduction='sum')

	def forward(self,sources,targets,length,speaker_label,addressee_label):
		source_embed=self.sembed(sources)
		context,h,c=self.encoder(source_embed,length)
		loss=0

		for i in range(targets.size(1)-1):
			target_embed=self.tembed(targets[:,i])
			pred,h,c=self.decoder(context,h,c,target_embed,speaker_label,addressee_label)
			pred=self.softlinear(pred)
			loss+=self.loss_function(pred,targets[:,i+1])
		return loss


class persona:
	
	def __init__(self, params):
		self.params=params
 
		self.ReadDict()
		self.Data=data(params,self.voc)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if params.cpu:
			self.device="cpu"
		self.Model = lstm(params,len(self.voc),self.Data.EOT)
		self.Model.encoder.apply(self.weights_init)
		self.Model.decoder.apply(self.weights_init)
		self.Model.softlinear.apply(self.weights_init)
		self.Model.to(self.device)

		self.output=path.join(params.save_folder,params.output_file)
		if self.output!="":
			with open(self.output,"w") as selfoutput:
				selfoutput.write("")
		if self.params.SpeakerMode:
			print("training in speaker mode")
		elif self.params.AddresseeMode:
			print("training in speaker-addressee mode")
		else:
			print("training in non persona mode")

	def weights_init(self,module):
		classname=module.__class__.__name__
		try:
			module.weight.data.uniform_(-self.params.init_weight,self.params.init_weight)
		except:
			pass
	
	def ReadDict(self):
		self.voc = dict()
		with open(path.join(self.params.data_folder,self.params.dictPath),'r') as doc:
			for line in doc:
				self.voc[line.strip()] = len(self.voc)

	def test(self):
		open_train_file=path.join(self.params.data_folder,self.params.dev_file)
		total_loss = 0
		total_tokens = 0
		END=0
		batch_n=0
		while END==0:
			END,sources,targets,speaker_label,addressee_label,length,token_num,origin = self.Data.read_batch(open_train_file,batch_n)
			batch_n+=1
			if sources is None:
				break
			sources=sources.to(self.device)
			targets=targets.to(self.device)
			speaker_label=speaker_label.to(self.device)
			addressee_label=addressee_label.to(self.device)
			length=length.to(self.device)
			total_tokens+=token_num
			self.Model.eval()
			with torch.no_grad():
				loss = self.Model(sources,targets,length,speaker_label,addressee_label)
				total_loss+=loss.item()
		print(list(self.Model.decoder.persona_embedding.parameters())[0][1,:10])
		print("perp "+str((1/math.exp(-total_loss/total_tokens))))
		if self.output!="":
			with open(self.output,"a") as selfoutput:
				selfoutput.write("standard perp "+str((1/math.exp(-total_loss/total_tokens)))+"\n")

	def update(self):
		lr=self.params.alpha
		grad_norm=0
		for m in list(self.Model.parameters()):
			m.grad.data = m.grad.data*(1/self.source_size)
			grad_norm+=m.grad.data.norm()**2
		grad_norm=grad_norm**0.5
		if grad_norm>self.params.thres:
			lr=lr*self.params.thres/grad_norm
		for f in self.Model.parameters():
			f.data.sub_(f.grad.data * lr)

	def save(self):
		save_path = path.join(self.params.save_folder,self.params.save_prefix)
		torch.save(self.Model.state_dict(),save_path+str(self.iter))
		print("finished saving")

	def saveParams(self):
		save_params_path = path.join(self.params.save_folder,self.params.save_params)
		with open(save_params_path,"wb") as file:
			pickle.dump(self.params,file)

	def readModel(self,save_folder,model_name,re_random_weights=None):
		target_model = torch.load(path.join(save_folder,model_name))
		if re_random_weights is not None:
			for weight_name in re_random_weights:
				random_weight = self.Model.state_dict()[weight_name]
				target_model[weight_name] = random_weight
		self.Model.load_state_dict(target_model)
		print("read model done")

	def train(self):
		if not self.params.no_save:
			self.saveParams()
		if self.params.fine_tuning:
			if self.params.SpeakerMode or self.params.AddresseeMode:
				re_random_weights = ['decoder.persona_embedding.weight'] # Also have to include some layers of the LSTM module...
			else:
				re_random_weights = None
			self.readModel(self.params.save_folder,self.params,fine_tuning_model,re_random_weights)
		self.iter=0
		start_halving=False
		self.lr=self.params.alpha
		print("iter  "+str(self.iter))
		self.test()
		while True:
			self.iter+=1
			print("iter  "+str(self.iter))
			if self.output!="":
				with open(self.output,"a") as selfoutput:
					selfoutput.write("iter  "+str(self.iter)+"\n")
			if self.iter>self.params.start_halve:
				self.lr=self.lr*0.5
			open_train_file=path.join(self.params.data_folder,self.params.train_file)
			END=0
			batch_n=0
			while END==0:
				self.Model.zero_grad()
				END,sources,targets,speaker_label,addressee_label,length,_,_ = self.Data.read_batch(open_train_file,batch_n)
				batch_n+=1
				if sources is None:
					break
				sources=sources.to(self.device)
				targets=targets.to(self.device)
				speaker_label=speaker_label.to(self.device)
				addressee_label=addressee_label.to(self.device)
				length=length.to(self.device)
				self.source_size = sources.size(0)
				self.Model.train()
				loss = self.Model(sources,targets,length,speaker_label,addressee_label)
				loss.backward()
				self.update()
			self.test()
			if not self.params.no_save:
				self.save()
			if self.iter==self.params.max_iter:
				break