module ElmanRNN

# Define the structure of an Elman RNN.
type ElmanNetwork
	weightsHI::Matrix{Float} # Weights from input to hidden layer.
	weightsHC::Matrix{Float} # Weights from context to hidden layer.
	weightsOH::Matrix{Float} # Weights from hidden to output layer.
	# functions (w/ constants like eta, mu included in closures) for error function, context, etc?

	# Initialize the network with random weights from [-.5.5).
	function ElmanNetwork( inputSize::Int, hiddenSize::Int, outputSize::Int)
		wHI = rand( hiddenSize, inputSize) - .5
		wHC = rand( hiddenSize, hiddenSize) - .5
		wOH = rand( outputSize, hiddenSize) - .5

		new( wHI, wHC, wOH)
	end
end

# Train the network in place with the given inputs and targets.
function ElmanTrain!( network::ElmanNetwork, inputs::Matrix{Float}, targets::Vector{Float})
	# Check dimensions of inputs.
	sizeContext = size( weightsHC, 1)

	Initialize context layer to zero vector.
	contextLayer = zeros( Float, sizeContext)


	# return error, number of epochs?
end

function ElmanEvaluate( network::ElmanNetwork, input::Vector{Float})
	# Check dimensions of inputs.

end

end
