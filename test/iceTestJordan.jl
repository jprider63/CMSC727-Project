require("../src/RNN.jl")
require("../src/JordanRNN.jl")
require("helper.jl")
using RNN
using JordanRNN

srand(63)

data = importData("ice")
dataOutput = data[:,1]
dataInput = zeros(length(dataOutput))
sampleInput = TimeSeriesSample(dataInput)
sampleOutput = TimeSeriesSample(dataOutput)
vectorInput = [sampleInput]
vectorOutput = [sampleOutput]
samplesInput = TimeSeriesSamples(vectorInput)
samplesOutput = TimeSeriesSamples(vectorOutput)

net = JordanNetwork(1, 10, 1)
net.mu = .3
net.eta = .4
net.errorThreshold = .001
JordanTrain!(net, samplesInput, samplesOutput)

#print(net)

target = JordanEvaluate( net, sampleInput)

print( target - dataOutput')
