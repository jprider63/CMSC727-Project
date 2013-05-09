require("../src/RNN.jl")
require("../src/JordanRNN.jl")
require("helper.jl")
using RNN
using JordanRNN

srand(63)

data = importData("ice")
dataOutput = data[:,1]
dataInput = zeros(length(dataOutput))

trainingIndices = 1:100
testIndices = 101:219

trainingInput = TimeSeriesSample( dataInput[trainingIndices])
trainingInputs = TimeSeriesSamples( [trainingInput])
trainingOutput = TimeSeriesSample( dataOutput[trainingIndices])
trainingOutputs = TimeSeriesSamples( [trainingOutput])

testInput = TimeSeriesSample( dataInput[testIndices])
testInputs = TimeSeriesSamples( [testInput])
testOutput = TimeSeriesSample( dataOutput[testIndices])
testOutputs = TimeSeriesSamples( [testOutput])

net = JordanNetwork(1, 10, 1)
net.mu = .3
net.eta = .5
net.errorThreshold = .001
JordanTrain!(net, trainingInputs, trainingOutputs)

#print(net)

target = JordanEvaluate( net, testInput)

#print( target - testOutput.sample)
print(norm(target - testOutput.sample))