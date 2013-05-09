require("../src/RNN.jl")
require("../src/JordanRNN.jl")
require("helper.jl")
using RNN
using JordanRNN

srand(63)

data = importData("chickenpox")
dataInput = data[:,1]
dataOutput = map(x -> if (x >= mean(dataInput)) 1.0 else 0.0 end, dataInput)

trainingIndices = 1:249
testIndices = 250:498

trainingInput = TimeSeriesSample( dataInput[trainingIndices])
trainingInputs = TimeSeriesSamples( [trainingInput])
trainingOutput = TimeSeriesSample( dataOutput[trainingIndices])
trainingOutputs = TimeSeriesSamples( [trainingOutput])

testInput = TimeSeriesSample( dataInput[testIndices])
testInputs = TimeSeriesSamples( [testInput])
testOutput = TimeSeriesSample( dataOutput[testIndices])
testOutputs = TimeSeriesSamples( [testOutput])

net = JordanNetwork(1, 20, 1)
net.mu = 0
net.eta = .3
net.errorThreshold = .01
numEpochs, lastTrainingError = JordanTrain!(net, trainingInputs, trainingOutputs)

target = JordanEvaluate( net, testInput)
testError = norm(target - testOutput.sample)

print("Num Epochs: $numEpochs\n")
print("Last Training Error: $lastTrainingError\n")
print("Testing Error: $testError\n")