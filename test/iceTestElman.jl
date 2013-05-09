require("../src/RNN.jl")
require("../src/ElmanRNN.jl")
require("helper.jl")
using RNN
using ElmanRNN

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

net = ElmanNetwork(1, 10, 1)
net.mu = .3
net.eta = .5
net.errorThreshold = .001
ElmanTrain!(net, trainingInputs, trainingOutputs)

#print(net)

target = ElmanEvaluate( net, testInput)

#print( target - testOutput.sample)
print(norm(target - testOutput.sample))