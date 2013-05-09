module ElmanRNN
using RNN

export ElmanNetwork, ElmanTrain!, ElmanEvaluate

# Define the structure of an Elman RNN.
type ElmanNetwork
	# Weights between each layer.
	weightsHI::Matrix{Float64} # Weights from input to hidden layer.
	weightsHC::Matrix{Float64} # Weights from context to hidden layer.
	weightsOH::Matrix{Float64} # Weights from hidden to output layer.
	weightsHB::Vector{Float64} # Weights from bias node to hidden layer.

	# Parameters
	eta::Float64 # Learning rate.
	mu::Float64 # Context decay rate

	# Constants
	maxEpochs::Uint # The maximum number of epochs.
	errorThreshold::Float64 # The error threshold to stop training at.

	# Functions
	activationRule::ActivationRule # Activation rule.
	errorFunction::ErrorFunction # Error function.

	# Scaling
	inputScale::Vector{Float64}
	inputShift::Vector{Float64}
	targetScale::Vector{Float64}
	targetShift::Vector{Float64}


	# Initialize the network with random weights from [-.5.5).
	function ElmanNetwork( inputSize::Int, hiddenSize::Int, outputSize::Int)
		wHI = rand( hiddenSize, inputSize) - .5
		wHC = rand( hiddenSize, hiddenSize) - .5
		wOH = rand( outputSize, hiddenSize) - .5
		wHB = rand( hiddenSize) - .5

		new( wHI, wHC, wOH, wHB, defaultLearningRate, defaultDecayRate, defaultMaxEpochs, defaultErrorThreshold, defaultActivationRule(), defaultErrorFunction(), fill(1.0,inputSize), fill(0.0,inputSize), fill(1.0,inputSize), fill(0.0,inputSize))
	end
end

# Train the network in place with the given inputs and targets.
function ElmanTrain!( network::ElmanNetwork, inputs::TimeSeriesSamples, targets::TimeSeriesSamples)
	# Check dimensions.
	sizeContext = size( network.weightsHC, 1)
	sizeInput = size( network.weightsHI, 2)
	sizeOutput = size( network.weightsOH, 1)

	sizeTraining = size( inputs.samples, 1)

	# Check that the number of inputs and targets are equal.
	if sizeTraining != size( targets.samples, 1)
		Base.error( "The number of inputs and targets must be equal!")
	end

	# Check that the size of each input vector is equal to the size of the input layer.
	if sizeInput != inputs.sizeSample
		Base.error( "The size of each input and the size input layer must be equal!")
	end

	# Check that the size of each target vector is equal to the size of the output layer.
	if sizeOutput != targets.sizeSample
		Base.error( "The size of each target and the size of the output layer must be equal!")
	end

	# Check that the size of the bias vector is equal to the size of the hidden layer.
	if length( network.weightsHB) != size( network.weightsHC, 1)
		Base.error( "The size of the bias vector and the size of the hidden layer must be equal!")
	end

	# Normalize input and target data.
	inputs, network.inputScale, network.inputShift = normalize( inputs)
	targets, network.targetScale, network.targetShift = normalize( targets)
	
	# Iterate over each epoch.
	epoch = 0
	totalError = Inf
	while epoch < network.maxEpochs && totalError > network.errorThreshold
		epochErrors = Array( Float64, length(inputs.samples))

		# Iterate over each training pair.
		for p in 1:length(inputs.samples)
			error = 0 # As long as there are more than 1 time steps this should be fine. TODO: Do we need to check this?
			sample = inputs.samples[p].sample
			target = targets.samples[p].sample
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

				# Back propagate input layer, context layer, and bias vector.
				deltaH = network.weightsOH' * deltaO .* map(network.activationRule.activationDerivative, aH)
				network.weightsHI += network.eta * (deltaH * inputActivation')
				network.weightsHC += network.eta * (deltaH * contextLayer')
				network.weightsHB += network.eta * deltaH

				# Update context layer based on output activation.
				contextLayer = network.mu * contextLayer + aH

				# Update error.
				error +=  network.errorFunction( aO, targetT)
			end

			# Compute the average error by dividing by the number of time steps.
			epochErrors[p] = error / length(sample)
		end

		totalError = network.errorFunction( epochErrors, zeros( length(inputs.samples)))
		print( string( totalError)*"\n")
		epoch = epoch + 1
	end

	# Return the number of epochs and the error.
	epoch, totalError
end

# Helper function to ElmanEvaluate, which is used internally to feed forward.
function ElmanEvaluateHelper( network::ElmanNetwork, input::Vector{Float64}, contextLayer::Vector{Float64})
	# Feed forward to compute the hidden and output activations.
	inH = network.weightsHI * input + network.weightsHC * contextLayer + network.weightsHB
	aH = map(network.activationRule.activationFunction, inH)
	inO = network.weightsOH * aH
	aO = map(network.activationRule.activationFunction, inO)

	# Return the target activation and hidden activation.
	aO, aH
end

# Evaluate the Elman RNN with the given input vector.
function ElmanEvaluate( network::ElmanNetwork, input::TimeSeriesSample)
	# Check that the size of each input is equal to the size of the input layer.
	if size( network.weightsHI, 2) != size( input.sample, 1)
		Base.error( "The size of the input and the size input layer must be equal!")
	end

	# Normalize input.
	input = deepcopy( input)
	for i in 1:size( input.sample, 1)
		input.sample[i,:] = (input.sample[i,:] - network.inputShift[i]) / network.inputScale[i]
	end

	# Initialize context layer to zero vector.
	contextLayer = zeros( Float64, size( network.weightsHI, 1))

	# Evaluate the Elman RNN.
	target = Array( Float64, size( network.weightsOH, 1), size( input.sample, 2))
	for i in 1:size( input.sample, 2)
		target[:,i], aH = ElmanEvaluateHelper( network, input.sample[:,i], contextLayer)

		# Update context layer based on output activation.
		contextLayer = network.mu * contextLayer + aH
	end

	# Scale and shift back.
	for i in 1:length( network.targetScale)
		target[i,:] = target[i,:] * network.targetScale[i] + network.targetShift[i]
	end

	# Return the target.
	target
end

end
