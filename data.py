from os import path
import torch
from torch import tensor
import numpy as np
import string
import linecache


class data:

	# Assume the data is of this form: SpeakerId Text|AddresseeId Text
	
	def __init__(self, params, voc):
		self.params = params
		self.voc = voc
		# EOS: End of source, start of target
		self.EOS = 1
		# EOT: End of target
		self.EOT = 2
		self.padding = 0  # Not used, just a reminder
		self.UNK = params.UNK+params.special_word
		
	def encode(self, tokens):
		ids = []
		for token in tokens:
		### For raw-word data:
		# 	try:
		# 		ids.append(self.voc[token]+self.params.special_word)
		# 	except KeyError:
		# 		ids.append(self.UNK)
		###--------------------
		### For data that is already tokenized and transferred to ids:
			# ids.append(int(token)+self.params.special_word)
		### For testing data (numbering starts from 1, not 0):
			ids.append(int(token)-1+self.params.special_word)
		return ids

	def read_batch(self, file, num, mode='train_or_test'):
		origin = []
		sources = np.zeros((self.params.batch_size, self.params.source_max_length+1))
		targets = np.zeros((self.params.batch_size, self.params.source_max_length+1))
		speaker_label = -np.ones(self.params.batch_size)
		addressee_label = -np.ones(self.params.batch_size)
		l_s_set = set()
		l_t_set = set()
		END=0
		a=0
		for i in range(self.params.batch_size):
			line = linecache.getline(file,num*self.params.batch_size+i+1).strip().split("|")
			i-=a
			if line == ['']:
				END = 1
				break
			s = line[-2].split()[:self.params.source_max_length]
			t = line[-1].split()[:self.params.target_max_length]
			if s[1:]==[]:
				a+=1
				continue
			elif t[1:]==[] and mode!='decode':
				a+=1
				continue
			source=self.encode(s[1:])
			target=[self.EOS]+self.encode(t[1:])+[self.EOT]
			l_s=len(source)
			l_t=len(target)
			l_s_set.add(l_s)
			l_t_set.add(l_t)
			### If the data contains words, not numbers:
			# origin.append(' '.join(s[1:]))
			origin.append(source)
			sources[i, :l_s]=source
			targets[i, :l_t]=target
			try:
				speaker_label[i]=int(s[0])-1
				addressee_label[i]=int(t[0])-1
			except:
				print('Persona id cannot be transferred to numbers')
			i+=1

		try:
			max_l_s=max(l_s_set)
			max_l_t=max(l_t_set)
		except ValueError:
			return END,None,None,None,None,None,None,None

		if max_l_s == 0:
			return END,None,None,None,None,None,None,None
		elif max_l_t == 2 and mode != 'decode':
			return END,None,None,None,None,None,None,None
			
		
		sources=sources[:i, : max_l_s]
		targets=targets[:i, : max_l_t]
		speaker_label=speaker_label[:i]
		addressee_label=addressee_label[:i]
		length_s=(sources!=0).sum(1)
		mask_t=np.ones(targets.shape)*(targets!=0)
		token_num=mask_t[:,1:].sum()

		return END,tensor(sources).long(),tensor(targets).long(),tensor(speaker_label).long(),tensor(addressee_label).long(),tensor(length_s).long(),token_num,origin
	