import errno, os, sys

class Output():
	"""
	This class handles printing of a columed output and a CSV.
	"""

	# The table below is organized as 
	# user_option: [output_header, attribute_in_Data_class, type, min_width_in_columed_output]
	table = {
		"idx":		["Idx",			"index",	int,	7],
		"seq":		["SeqId",		"seqId",	str,	7],
		"altseq":	["AltSeqId",	"altSeqId",	str,	7],
		"tid":		["TId",			"tid",		int,	12],
		"layer":	["Layer", 		"layer",	str,	10],
		"trace":	["Trace",		"trace",	str,	25],
		"dir":		["Direction",	"dir",		str,	5],
		"sub":		["Sub",			"sub",		int,	3],
		"mod":		["Module",		"mod",		str,	15],
		"op":		["Op",			"op",		str,	15],
		"kernel":	["Kernel",		"name",		str,	0],
		"params":	["Params",		"params",	str,	0],
		"sil":		["Sil(ns)",		"sil",		int,	10],
		"tc":		["TC",			"tc",		str,	2],
		"device":	["Device",		"device",	int,	3],
		"stream":	["Stream",		"stream",	int,	3],
		"grid":		["Grid",		"grid",		str,	12],
		"block":	["Block",		"block",	str,	12],
		"flops":	["FLOPs", 		"flops",	int,	12],
		"bytes":	["Bytes",		"bytes", 	int,	12]
	}

	def __init__(self, args):
		self.cols = args.c
		self.csv = args.csv
		self.col = args.w > 0
		self.width = args.w

		w = 0
		for col in self.cols:
			assert col in Output.table.keys()
			w += Output.table[col][3]

		if ((self.col) and (w > self.width)):
			print(f'Minimum width required to print {",".join(self.cols)} = {w}. Exiting.')
			sys.exit(1)

		remainder = self.width - w

		if ("kernel" in self.cols) and ("params" in self.cols):
			Output.table["kernel"][3] = int(remainder/2)
			Output.table["params"][3] = int(remainder/2)
		elif ("kernel" in self.cols):
			Output.table["kernel"][3] = remainder
		elif ("params" in self.cols):
			Output.table["params"][3] = remainder

		#header format
		cadena = ""
		for col in self.cols:
			_,_,t,w = Output.table[col]
			cadena += f"%-{w}.{w}s "

		self.hFormat = cadena

		#data format
		cadena = ""
		for col in self.cols:
			_,_,t,w = Output.table[col]
			if (t == str):
				cadena += f"%-{w}.{w}s "
			elif (t == int):
				cadena += f"%{w}d "

		self.dFormat = cadena

	def foo(self, cadena, pformat):
		if self.csv:
			cadena = ",".join(map(lambda x: f'"{str(x)}"', cadena))
		elif self.col:
			cadena = pformat % cadena
		else:
			cadena = " ".join(map(str,cadena))

		try:
			print(cadena)
		except IOError as e:
			#gracefully handle pipes
			if e.errno == errno.EPIPE:
				# Python flushes standard streams on exit; redirect remaining output
				# to devnull to avoid another BrokenPipeError at shutdown

				devnull = os.open(os.devnull, os.O_WRONLY)
				os.dup2(devnull, sys.stdout.fileno())
				sys.exit(0)
			else:
				sys.exit(-1)

	def header(self):
		cadena = ()
		for col in self.cols:
			h = Output.table[col][0]
			cadena = cadena + (h,)

		self.foo(cadena, self.hFormat)

	def data(self, a):
		direc = "na" if a.dir == "" else a.dir
		op = "na" if a.op == "" else a.op
		mod = "na" if a.mod == "" else a.mod
		cadena = ()
		for col in self.cols:
			attr = Output.table[col][1]
			val = getattr(a, attr)

			if col == "layer":
				assert(type(val) == list)
				val = ":".join(val)
				val = "-" if not val else val

			elif col == "trace":
				assert(type(val) == list)
				if self.col and len(val):
					val = val[-1]
					val = val.split("/")[-1]
				else:
					val = ",".join(val)
					val = "-" if not val else val

			if col in ["seq", "altseq"]:
				assert(type(val) == list)
				val = ",".join(map(str,val))
				val = "-" if not val else val

			cadena = cadena + (val,)

		self.foo(cadena, self.dFormat)
