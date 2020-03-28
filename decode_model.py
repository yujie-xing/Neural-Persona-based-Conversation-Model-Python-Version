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
from torch import tensor
from torch.autograd import Variable, backward

class lstm_decoder(lstm):

	def forward(self,sources,targets,length,speaker_label,addressee_label,mode='test'):
		source_embed=self.sembed(sources)
		context,h,c=self.encoder(source_embed,length)
		loss=0

		with torch.no_grad():
			if mode=='test':    
				for i in range(targets.size(1)-1):
					target_embed=self.tembed(targets[:,i])
					pred,h,c=self.decoder(context,h,c,target_embed,speaker_label,addressee_label)
					pred=self.softlinear(pred)
					loss+=self.loss_function(pred,targets[:,i+1])
				return loss
			elif mode != 'decode':
				raise NameError('Wrong mode: '+mode)
			elif self.params.setting == 'beam_search':
				start=self.tembed(targets[:,0])
				return self.beam_search(start,context,h,c,speaker_label,addressee_label)
			else: 
				start=self.tembed(targets[:,0])
				pred,h,c=self.decoder(context,h,c,start,speaker_label,addressee_label)
				predicted_word = self.sample(pred)
				prediction = predicted_word.unsqueeze(1).clone()
				for i in range(1,self.params.max_decoding_length):
					pred,h,c=self.decoder(context,h,c,self.tembed(predicted_word),speaker_label,addressee_label)
					predicted_word = self.sample(pred)
					prediction = torch.cat((prediction,predicted_word.unsqueeze(1).clone()),1)
					if (prediction==self.EOT).any(1).all():
						break
				return prediction
				

	def sample(self,pred):
		pred = self.softlinear(pred)
		pred = nn.Softmax(dim=1)(pred)
		if not self.params.allowUNK:
			pred[:,self.UNK].fill_(0)
		if self.params.setting == 'sample':
			predicted_word = torch.multinomial(pred,1).squeeze(1)
			return predicted_word
		elif self.params.setting == 'StochasticGreedy':
			select_P,select_words=torch.topk(pred,self.params.StochasticGreedyNum,1,True,True)
			pred=F.normalize(select_P, 1, dim=1)
			predicted_index =torch.multinomial(pred, 1).squeeze(1)
			predicted_word = select_words[torch.arange(select_words.size(0)),predicted_index]
			return predicted_word
		else:
			raise NameError('No setting called '+self.params.setting)

	def beam_search(self,start,context,h,c,speaker_label,addressee_label):
		 
		pred,h,c=self.decoder(context,h,c,start,speaker_label,addressee_label)
		pred=self.softlinear(pred)
		pred=nn.LogSoftmax(dim=1)(pred)
		probTable,beamHistory = torch.topk(pred,self.params.beam_size,1,True,False)
		beamHistory = beamHistory.unsqueeze(2)

		for i in range(1,self.params.max_decoding_length):
			for k in range(self.params.beam_size):
				pred,h,c=self.decoder(context,h,c,self.tembed(beamHistory[:,k,-1]),speaker_label,addressee_label)
				pred=self.softlinear(pred)
				pred=nn.LogSoftmax(dim=1)(pred)
				prob_k,beam_k = torch.topk(pred,self.params.beam_size,1,True,False)
				prob_k += probTable[:,k].unsqueeze(1)
				beam_k = torch.cat((beamHistory[:,k].unsqueeze(1).expand(beamHistory.size()),
						beam_k.unsqueeze(2)),2)
				if k==0:
					prob = prob_k
					beam = beam_k
				else:
					prob = torch.cat((prob,prob_k),1)
					beam = torch.cat((beam,beam_k),1)
			probTable,index = torch.topk(prob,self.params.beam_size,1,True,False)
			beamHistory = beam[torch.arange(beam.size(0)).view(-1,1).expand(index.size()).contiguous().view(-1),
						index.view(-1),:].view(index.size(0),index.size(1),beam.size(2))

			if (beamHistory == self.EOT).any(dim=2).all():
				break

		predicted_path = torch.argmax(probTable,1)

		return beamHistory[torch.arange(beamHistory.size(0)),predicted_path,:]

class decode_model(persona):

	def __init__(self, params):
		with open(path.join(params.model_folder,params.params_name), 'rb') as file:
			adapted_params = pickle.load(file)
		for key in vars(params):
			vars(adapted_params)[key] = vars(params)[key]
		adapted_params.dev_file = adapted_params.decode_file
		self.params=adapted_params
		if self.params.PersonaMode:
			print("decoding in speaker mode")
		elif self.params.AddresseeMode:
			print("decoding in speaker-addressee mode")
		else:
			print("decoding in non persona mode")

		self.ReadDict()
		self.Data=data(self.params,self.voc)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if self.params.cpu:
			self.device="cpu"

		self.Model = lstm_decoder(self.params,len(self.voc),self.Data.EOT)
		self.readModel(self.params.model_folder,self.params.model_name)
		self.Model.to(self.device)
		self.ReadDictDecode()

		self.output=path.join(self.params.output_folder,self.params.log_file)
		if self.output!="":
			with open(self.output,"w") as selfoutput:
				selfoutput.write("")


	def ReadDictDecode(self):
		self.voc_decode = dict()
		with open(path.join(self.params.data_folder,self.params.dictPath),'r') as doc:
			for line in doc:
				self.voc_decode[len(self.voc_decode)] = line.strip()
		

	def id2word(self, ids):
		### For raw-word data:
		# self.voc_decode[len(self.voc_decode)] = '[unknown]'
		tokens = []
		for i in ids:
			try:
				word = self.voc_decode[int(i)-self.params.special_word]
				tokens.append(word)
			except KeyError:
				break
		return " ".join(tokens)

	def decode(self):
		self.mode="decode"
		open_train_file = path.join(self.params.data_folder,self.params.decode_file)

		if self.params.PersonaMode:
			decode_output = path.join(self.params.output_folder,
							self.params.decode_file+"_S"+str(self.params.SpeakerId)+"_"+self.params.output_file)
		elif self.params.AddresseeMode:
			decode_output = path.join(self.params.output_folder,
							self.params.decode_file+"_S"+str(self.params.SpeakerId)+"A"+str(self.params.AddresseeId)+"_"+self.params.output_file)
		else:
			decode_output = path.join(self.params.output_folder,self.params.decode_file+"_"+self.params.output_file)
		with open(decode_output,"w") as open_write_file:
			open_write_file.write("")

		END=0
		batch_n=0
		n_decode_instance=0
		while END==0:
			END,sources,targets,speaker_label,addressee_label,length,token_num,origin = self.Data.read_batch(open_train_file,batch_n,self.mode)
			if sources is None:
				break
			batch_n+=1
			n_decode_instance += sources.size(0)
			if self.params.max_decoding_number != 0 and n_decode_instance >= self.params.max_decoding_number:
				break
			speaker_label.fill_(self.params.SpeakerId-1)
			addressee_label.fill_(self.params.AddresseeId-1)
			sources=sources.to(self.device)
			targets=targets.to(self.device)
			speaker_label=speaker_label.to(self.device)
			addressee_label=addressee_label.to(self.device)
			peaker_label=speaker_label.to(self.device)
			addressee_label=addressee_label.to(self.device)
			length=length.to(self.device)
			self.origin = origin
			self.source_size = sources.size(0)
			with torch.no_grad():
				completed_history = self.Model(sources,targets,length,speaker_label,addressee_label,self.mode)
			self.OutPut(decode_output,completed_history)
		print("decoding done")

	def OutPut(self,decode_output,completed_history):
		for i in range(self.source_size):
			if self.params.response_only:
				print_string=self.id2word(completed_history[i].cpu().numpy())
				with open(decode_output,"a") as file:
					file.write(print_string+"\n")
			else:
				### If the data contains words, not numbers:
				# print_string = origin
				print_string = self.id2word(self.origin[i])
				print_string += "|"
				print_string += self.id2word(completed_history[i].cpu().numpy())
				with open(decode_output,"a") as file:
					file.write(print_string+"\n")

