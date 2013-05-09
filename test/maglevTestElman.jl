require("../src/RNN.jl")
require("../src/ElmanRNN.jl")
require("helper.jl")
using RNN
using ElmanRNN

srand(63)

data = importData("maglev")
dataInput = data[:,1]
dataOutput = data[:,2]

trainingIndices = 1:2000
testIndices = 2001:4001

trainingInput = TimeSeriesSample( dataInput[trainingIndices])
trainingInputs = TimeSeriesSamples( [trainingInput])
trainingOutput = TimeSeriesSample( dataOutput[trainingIndices])
trainingOutputs = TimeSeriesSamples( [trainingOutput])

testInput = TimeSeriesSample( dataInput[testIndices])
testInputs = TimeSeriesSamples( [testInput])
testOutput = TimeSeriesSample( dataOutput[testIndices])
testOutputs = TimeSeriesSamples( [testOutput])

net = ElmanNetwork(1, 30, 1)
net.mu = .1
net.eta = .3
net.errorThreshold = .01
numEpochs, lastTrainingError = ElmanTrain!(net, trainingInputs, trainingOutputs)

target = ElmanEvaluate( net, testInput)
testError = norm(target - testOutput.sample)

print("Num Epochs: $numEpochs\n")
print("Last Training Error: $lastTrainingError\n")
print("Testing Error: $testError\n")