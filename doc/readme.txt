Installation
============
You will need to install Julia from http://julialang.org/downloads/ to run .jl files.

Files
=====
src/ directory contains source files for recurrent networks
	RNN.jl implements common functions and constants needed by both Elman and Jordan networks
	ElmanRNN.jl implements an Elman recurrent network
	JordanRNN.jl implements a Jordan recurrent network
	Both ElmanRNN and JordanRNN contain initialization, training, and evaluation functions
test/ directory contains various files used in testing the networks
	*.csv files contain various time series data sets
	household_power_consumption.txt contains a time series data set in a similar format as the *.csv files (but delimited with semicolons)
		In these data files, the first row contains the feature name and the corresponding column contains the time series data for that feature
	*TestElman.jl and *TestJordan.jl files are scripts which train a network on the corresponding data set, test it, and print out the results
	helper.jl contains helper functions needed in the test files
	TestResults.txt contains a compiled list of results with various parameters

Execution
=========
Once Julia is installed and in your path, the test files can be run directly.
For example, "julia exchangerTestElman.jl" will train an Elman network on the exchanger data set, evaluate it, and print out the results.
Parameters for the network can be set (there are examples in the test files)