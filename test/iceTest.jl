require("../src/RNN.jl")
require("../src/ElmanRNN.jl")
require("helper.jl")
using RNN
using ElmanRNN

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

net = ElmanNetwork(1, 10, 1)
net.mu = .3
net.eta = .4
net.errorThreshold = .001
ElmanTrain!(net, samplesInput, samplesOutput)

#print(net)

target = ElmanEvaluate( net, sampleInput)

print( target - dataOutput')
