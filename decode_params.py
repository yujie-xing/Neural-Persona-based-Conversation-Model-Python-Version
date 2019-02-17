class decode_params:
	def __init__(self):
		self.beam_size=7
		self.batch_size=256
		self.SpeakerID=2
		self.decode_path="save/testing"
		self.model_file=self.decode_path+"/model1"
		self.DecodeFile="/test.txt"
		self.DiverseRate=0
		self.OutputFolder="outputs"
		self.max_length=20
		self.min_length=0
		self.max_decoded_num=0
		self.onlyPred=True
		self.output_source_target_side_by_side=True
		self.StochasticGreedyNum=1
		self.target_length=0
		self.setting="StochasticGreedy"
		self.allowUNK=False