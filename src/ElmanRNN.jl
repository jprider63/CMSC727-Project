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
		activationRule::RNN.ActivationRule # Activation rule.
		errorFunction::Function

		# functions (w/ constants like eta, mu included in closures) for error function, context, etc?

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
		error = Inf # TODO: or just compute?
		while epoch < network.maxEpoch && error > network.errorThreshold
			# Iterate over each training pair.
			for p in 1:length(inputs.samples)
				sample = inputs.samples[p]
				target = targets.samples[p]
				# initialize contextLayer to zero vector
				contextLayer = zeros( Float64, sizeContext)
				for t in 1:length(sample)
					# Propogate input activation forward.
					inputActivation = sample[:,t]
					targetT = target[:,t]
					targetActivation, hiddenActivation = ElmanEvaluateHelper( network, inputActivation, contextLayer)

					#TODO: EBP, update weights
					deltaO = (targetT - aO) .* map(network.activationRule.activationDerivative, aO)
					network.weightsOH += network.eta * (deltaO * aH')

					deltaH = network.weightsOH' * deltaO
					network.weightsHI += network.eta * (deltaH * inputActivation')
					network.weightsHC += network.eta * (deltaH * contextLayer')

					#TODO: Copy output back into context nodes
					contextLayer = network.mu * contextLayer + aO

					#TODO: Calculate mse
				end
			end

			epoch = epoch + 1
			# check if finished (threshold or error?)
		end

		# Return the number of epochs and the error.
		epoch, error
	end

	# Helper function to ElmanEvaluate, which is used internally during training.
	function ElmanEvaluateHelper( network::ElmanNetwork, input::Vector{Float64}, contextLayer::Vector{Float})

		# TODO: iterate input, get activations
		inH = network.weightsHI * input + network.weightsHC * contextLayer
		aH = map(network.activationRule.activationFunction, inH)
		inO = network.weightsOH * aH
		aO = map(network.activationRule.activationFunction, inO)

		# Return the target activation, hidden activation.
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
		target, _, error = ElmanEvaluateHelper( network, input, contextLayer)

		# Return the target and error.
		target, error
	end

end
