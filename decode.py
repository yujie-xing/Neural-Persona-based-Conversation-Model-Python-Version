from decode_model import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', type=str, default='data/testing',
					help='the folder that contains your dataset and vocabulary file')
parser.add_argument('--decode_file', type=str, default='test.txt')
parser.add_argument('--dictPath', type=str, default='vocabulary')
parser.add_argument('--model_folder', type=str, default='save/testing')
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--params_name', type=str, default='params')
parser.add_argument('--output_folder', type=str, default='outputs')
parser.add_argument('--log_file', type=str, default='decodelog')
parser.add_argument('--output_file', type=str, default='output.txt')

parser.add_argument('--cpu', action='store_true')

# parser.add_argument('--SpeakerMode', action='store_true')
# parser.add_argument('--AddresseeMode', action='store_true')
parser.add_argument('--SpeakerId', type=int, default=1)
parser.add_argument('--AddresseeId', type=int, default=2)

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_decoding_length", type=int, default=20)
parser.add_argument('--max_decoding_number', type=int, default=0)
parser.add_argument('--allowUNK', action='store_true')
parser.add_argument('--response_only', action='store_true')

parser.add_argument("--setting", type=str, default='StochasticGreedy',
					help='sample, StochasticGreedy, beam_search')
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--StochasticGreedyNum", type=int, default=5)

args = parser.parse_args()
print(args)
print()

if __name__ == '__main__':
	model = decode_model(args)
	model.decode()
	# model.test()
