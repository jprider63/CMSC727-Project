module ElmanRNN

export ElmanNetwork, ElmanTrain!, ElmanEvaluate

# Define the structure of an Elman RNN.
type ElmanNetwork
	# Weights between each layer.
	weightsHI::Matrix{Float} # Weights from input to hidden layer.
	weightsHC::Matrix{Float} # Weights from context to hidden layer.
	weightsOH::Matrix{Float} # Weights from hidden to output layer.

	# Parameters
	eta::Float # Learning rate. The default is .01.

	# Constants
	maxEpochs::Uint # The maximum number of epochs. The default is typemax( Uint).
	errorThreshold::Float # The error threshold to stop training at. The default is .0005.

	# Functions
	activationRule::Function # Activation rule. The default is the sigmoid activation rule. This function must take TODO...

	# functions (w/ constants like eta, mu included in closures) for error function, context, etc?

	# Initialize the network with random weights from [-.5.5).
	function ElmanNetwork( inputSize::Int, hiddenSize::Int, outputSize::Int)
		wHI = rand( hiddenSize, inputSize) - .5
		wHC = rand( hiddenSize, hiddenSize) - .5
		wOH = rand( outputSize, hiddenSize) - .5

		new( wHI, wHC, wOH, .01, typemax( Uint), .0005)
	end
end

# Train the network in place with the given inputs and targets.
function ElmanTrain!( network::ElmanNetwork, inputs::Matrix{Float}, targets::Matrix{Float})
	# Check dimensions.
	sizeContext = size( network.weightsHC, 1)
	sizeInput = size( network.weightHI, 2)
	sizeOutput = size( network.weightOH, 1)

	sizeTraining = size( inputs, 1)
	lengthInput = size( inputs, 2)
	lengthTarget = size( targets, 2)

	# Check that the number of inputs and targets are equal.
	if sizeTraining != size( targets, 1)
		error( "The number of inputs and targets must be equal!")
	end

	# Check that the size of each input is equal to the size of the input layer.
	if sizeInput != lengthInput
		error( "The size of each input and the size input layer must be equal!")
	end

	# Check that the size of each target is equal to the size of the output layer.
	if sizeOutput != lengthTarget
		error( "The size of each target and the size of the output layer must be equal!")
	end
	
	# Initialize context layer to zero vector.
	contextLayer = zeros( Float, sizeContext)

	# Iterate over each epoch.
	epoch = 0
	error = Inf # TODO: or just compute?
	while epoch < network.maxEpoch
		# TODO: reinitialize contextLayer here instead?


		# Iterate over each training pair.
		for i in 1:sizeTraining
			# Propogate input activation forward.
			inputActivation = inputs[i,:]
			targetActivation, hiddenActivation, error = ElmanEvaluateHelper( network, inputActivation, contextLayer)


			#TODO: EBP, update weights


		end

		epoch = epoch + 1
		# check if finished (threshold or error?)
	end

	# Return the number of epochs and the error.
	epoch, error
end

# Helper function to ElmanEvaluate, which is used internally during training.
function ElmanEvaluateHelper( network::ElmanNetwork, input::Vector{Float}, contextLayer::Vector{Float})

	# TODO: iterate input, get activations


	# Return the target activation, hidden activation, and error.

end

# Evaluate the Elman RNN with the given input vector.
function ElmanEvaluate( network::ElmanNetwork, input::Vector{Float})
	# Check that the size of each input is equal to the size of the input layer.
	if size( network.weightHI, 2) != size( input)
		error( "The size of the input and the size input layer must be equal!")
	end

	# Initialize context layer to zero vector.
	contextLayer = zeros( Float, sizeContext)

	# Evaluate the Elman RNN.
	target, _, error = ElmanEvaluateHelper( network, input, contextLayer)

	# Return the target and error.
	target, error
end

end
