#!/usr/bin/env python3

"""
This script reads the output (Python dictionary) created by parse.py.
For every kernel (line) in the input it determines
	module / class name e.g. torch.nn.functional
	operator name e.g. linear
	kernel parameters e.g. GEMM M, N, K, datatype
	bytes
	flops
	tensor core usage
	direction (fprop, bprop)
	and other things. Please see the tool usage.
"""

from .usage import parseArgs
from .output import Output
from .utility import Utility
from .pointwise import Pointwise
from .convert import Convert
from .blas import *
from .embedding import Embedding
from .reduction import *
from .dropout import Dropout
from .softmax import *
#from pooling import * # work in progress
from .linear import Linear
from .optim import Adam
from .misc import *
from .conv import Conv
from .activation import Activation
from .index_slice_join_mutate import Cat, Reshape, MaskedScatter, Gather, Nonzero, IndexSelect, MaskedSelect
from .recurrentCell import RNNCell
from .normalization import BatchNorm
from .randomSample import RandPerm
from .loss import MSELoss
from .data import Data

def findFpropKernel(seq):
	#Find the last fprop kernel with the same seqId
	#First look at seqId and then at altSeqId
	for idx in reversed(range(len(kernels))):
		k = kernels[idx]
		if (seq in k['seqId']) and (k['dir'] == "fprop"):
			return idx

	for idx in reversed(range(len(kernels))):
		k = kernels[idx]
		if (seq in k['altSeqId']) and (k['dir'] == "fprop"):
			return idx

	return -1
	#print("Error: seqId {} not found.".format(seq), file=sys.stderr)
	#assert False

def foo(mod, op, d):
	if (op[0] == "linear"):
		return Linear(d)

	elif (mod[0] in["LSTMCell", "GRUCell"]) and (op[0] == "forward"):
		return RNNCell(d)

	elif op[0] in ["conv1d", "conv2d",]:
		return Conv(d)

	elif (op[0] in Pointwise.ops):
		return Pointwise(d)

	elif (op[0] in Convert.ops):
		return Convert(d)

	elif op[0] in ["__matmul__", "matmul"]:
		return Matmul(d)

	elif op[0] == "embedding":
		return Embedding(d)

	elif op[0] == "sum":
		return Sum(d)

	elif op[0] == "mean":
		return Mean(d)

	elif op[0] == "norm":
		return Norm(d)

	elif op[0] == "dropout":
		return Dropout(d)

	elif (op[0] == "cat"):
		return Cat(d)

	elif (op[0] == "reshape"):
		return Reshape(d)

	elif (op[0] == "masked_scatter_"):
		return MaskedScatter(d)

	elif (op[0] == "gather"):
		return Gather(d)

	elif (op[0] == "nonzero"):
		return Nonzero(d)

	elif (op[0] == "index_select"):
		return IndexSelect(d)

	elif (op[0] == "masked_select"):
		return MaskedSelect(d)

	elif op[0] in ["addmm", "addmm_"]:
		return Addmm(d)

	elif op[0] == "mm":
		return Mm(d)

	elif op[0] == "bmm":
		return Bmm(d)

	elif op[0] == "softmax":
		return Softmax(d)

	elif op[0] == "log_softmax":
		return LogSoftmax(d)

	elif op[0] == "mse_loss":
		return MSELoss(d)

	elif op[0] == "adam":
		return Adam(d)

	elif op[0] == "batch_norm":
		return BatchNorm(d)

	elif op[0] == "randperm":
		return RandPerm(d)

	elif op[0] == "copy_":
		return Copy(d)

	elif op[0] == "clone":
		return Clone(d)

	elif op[0] == "contiguous":
		return Contiguous(d)

	elif op[0] == "any":
		return Any(d)

	elif (op[0] in Activation.ops):
		return Activation(d)

	elif op[0] == "to":
		return Convert(d)

	else:
		return Foo(d)

def main():
	#Read cmd line arguments
	cmdArgs = parseArgs()

	output = Output(cmdArgs)
	output.header()

	#Read in all the kernel info
	for idx, line in enumerate(cmdArgs.file):
		kernel = eval(line)
		assert(kernel)
		kernels.append(kernel)

		k = kernel
		d = Data(k)

		mod = k['mod']
		op = k['op']

		flops = 0
		params = {"na":"na"}
		tc = "na"
		bytes = 0

		if (d.dir == "bprop"):
			d.seqMarker = k['seqMarker']
			seq = k['seqId']
			seq = k['seqId'][:1]
			assert (len(seq) == 1), seq
			#assert (seq[0] != 0)
			assert (len(d.seqMarker) > 0)
			#If there is no useful marker associated, use the
			#sequence number to find the kernel from fprop
			if len(d.argMarker) == 0:
				index = findFpropKernel(seq[0])
				if index >= 0:
					d.argMarker = kernels[index]['marker']
					d.modMarker = kernels[index]['reprMarkers']
					mod = kernels[index]['mod']
					op = kernels[index]['op']

					d.layer = kernels[index]['layer']
					d.trace = kernels[index]['trace']

		# Check if marker has our annotations
		if len(d.argMarker) and Utility.hasNVTX(d.argMarker[0]):

			xx = foo(mod, op, d)

			bytes = xx.bytes()
			flops = xx.flops()
			op = xx.op()
			params = xx.params()
			tc = xx.tc()

		if type(op) is list:
			op = op[0] if len(op) else ""
		if type(mod) is list:
			mod = mod[0] if len(mod) else ""
		d.index = idx+1

		# The following 8 come from operator class functions.
		d.setParams(params)
		d.tc = tc
		d.flops = flops
		d.bytes = bytes
		d.mod = mod
		d.op = op

		output.data(d)

kernels = []
if __name__ == '__main__':
	main()
