module ElmanRNN
using RNN

export ElmanNetwork, ElmanTrain!, ElmanEvaluate

# Define the structure of an Elman RNN.
type ElmanNetwork
	# Weights between each layer.
	weightsHI::Matrix{Float64} # Weights from input to hidden layer.
	weightsHC::Matrix{Float64} # Weights from context to hidden layer.
	weightsOH::Matrix{Float64} # Weights from hidden to output layer.

	# Parameters
	eta::Float64 # Learning rate.

	# Constants
	maxEpochs::Uint # The maximum number of epochs.
	errorThreshold::Float64 # The error threshold to stop training at.

	# Functions
	activationRule::ActivationRule # Activation rule.
	errorFunction::ErrorFunction # Error function.

	# Initialize the network with random weights from [-.5.5).
	function ElmanNetwork( inputSize::Int, hiddenSize::Int, outputSize::Int)
		wHI = rand( hiddenSize, inputSize) - .5
		wHC = rand( hiddenSize, hiddenSize) - .5
		wOH = rand( outputSize, hiddenSize) - .5

		new( wHI, wHC, wOH, defaultLearningRate, defaultMaxEpochs, defaultErrorThreshold, defaultActivationRule(), defaultErrorFunction())
	end
end

# Train the network in place with the given inputs and targets.
function ElmanTrain!( network::ElmanNetwork, inputs::TimeSeriesSamples, targets::TimeSeriesSamples)
	# Check dimensions.
	sizeContext = size( network.weightsHC, 1)
	sizeInput = size( network.weightHI, 2)
	sizeOutput = size( network.weightOH, 1)

	sizeTraining = size( inputs.samples, 1)

	# Check that the number of inputs and targets are equal.
	if sizeTraining != size( targets.samples, 1)
		error( "The number of inputs and targets must be equal!")
	end

	# Check that the size of each input vector is equal to the size of the input layer.
	if sizeInput != inputs.sizeSample
		error( "The size of each input and the size input layer must be equal!")
	end

	# Check that the size of each target vector is equal to the size of the output layer.
	if sizeOutput != targets.sizeSample
		error( "The size of each target and the size of the output layer must be equal!")
	end
	
	# Iterate over each epoch.
	epoch = 0
	error = Inf
	while epoch < network.maxEpoch && error > network.errorThreshold
		# Iterate over each training pair.
		for p in 1:length(inputs.samples)
			error = 0 # As long as there are more than 1 time steps this should be fine. TODO: Do we need to check this?
			sample = inputs.samples[p]
			target = targets.samples[p]
			# initialize contextLayer to zero vector
			contextLayer = zeros( Float64, sizeContext)
			for t in 1:length(sample)
				# Feed forward input activation.
				inputActivation = sample[:,t]
				targetT = target[:,t]
				aO, aH = ElmanEvaluateHelper( network, inputActivation, contextLayer)

				# Back propagate hidden layer.
				deltaO = (targetT - aO) .* map(network.activationRule.activationDerivative, aO)
				network.weightsOH += network.eta * (deltaO * aH')

				# Back propagate input and context layers.
				deltaH = network.weightsOH' * deltaO
				network.weightsHI += network.eta * (deltaH * inputActivation')
				network.weightsHC += network.eta * (deltaH * contextLayer')

				# Update context layer based on output activation.
				contextLayer = network.mu * contextLayer + aO

				# Update error.
				error +=  network.errorFunction( aO, targetT)
			end

			# Compute the average error by dividing by the number of time steps.
			error = error / length(sample)
		end

		epoch = epoch + 1
	end

	# Return the number of epochs and the error.
	epoch, error
end

# Helper function to ElmanEvaluate, which is used internally to feed forward.
function ElmanEvaluateHelper( network::ElmanNetwork, input::Vector{Float64}, contextLayer::Vector{Float64})

	# TODO: iterate input, get activations
	inH = network.weightsHI * input + network.weightsHC * contextLayer
	aH = map(network.activationRule.activationFunction, inH)
	inO = network.weightsOH * aH
	aO = map(network.activationRule.activationFunction, inO)

	# Return the target activation and hidden activation.
	aO, aH
end

# Evaluate the Elman RNN with the given input vector.
function ElmanEvaluate( network::ElmanNetwork, input::Vector{Float64})
	# Check that the size of each input is equal to the size of the input layer.
	if size( network.weightHI, 2) != size( input)
		error( "The size of the input and the size input layer must be equal!")
	end

	# Initialize context layer to zero vector.
	contextLayer = zeros( Float64, sizeContext)

	# Evaluate the Elman RNN.
	target, _ = ElmanEvaluateHelper( network, input, contextLayer)

	# Return the target.
	target
end

end
