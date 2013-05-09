require("../src/RNN.jl")
require("../src/ElmanRNN.jl")
require("helper.jl")
using RNN
using ElmanRNN

srand(63)

data = importData("laser")
dataInput = data[:,1]
dataOutput = map(x -> if (x >= mean(dataInput)) 1.0 else 0.0 end, dataInput)

trainingIndices = 1:5047
testIndices = 5048:10093

trainingInput = TimeSeriesSample( dataInput[trainingIndices])
trainingInputs = TimeSeriesSamples( [trainingInput])
trainingOutput = TimeSeriesSample( dataOutput[trainingIndices])
trainingOutputs = TimeSeriesSamples( [trainingOutput])

testInput = TimeSeriesSample( dataInput[testIndices])
testInputs = TimeSeriesSamples( [testInput])
testOutput = TimeSeriesSample( dataOutput[testIndices])
testOutputs = TimeSeriesSamples( [testOutput])

net = ElmanNetwork(1, 10, 1)
net.mu = 0
net.eta = .3
net.errorThreshold = .01
numEpochs, lastTrainingError = ElmanTrain!(net, trainingInputs, trainingOutputs)

target = ElmanEvaluate( net, testInput)
testError = norm(target - testOutput.sample)

print("Num Epochs: $numEpochs\n")
print("Last Training Error: $lastTrainingError\n")
print("Testing Error: $testError\n")