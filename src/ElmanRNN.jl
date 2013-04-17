module ElmanRNN

type ElmanNetwork
	weightsHI::Matrix{Float} # Weights from input to hidden layer.
	weightsHC::Matrix{Float} # Weights from context to hidden layer.
	weightsOH::Matrix{Float} # Weights from hidden to output layer.
	# functions (w/ constants like eta, mu included in closures) for error function, context, etc?


	function ElmanNetwork( inputSize::Int, hiddenSize::Int, outputSize::Int)
		wHI = rand( hiddenSize, inputSize)
		wHC = rand( hiddenSize, outputSize)
		wOH = rand( outputSize, hiddenSize)

		new( wHI, wHC, wOH)
	end
end

function ElmanTrain( network::ElmanNetwork, inputs::Matrix{Float}, target::Vector{Float})
	# Check dimensions of inputs.

	contextLayer = 


	# return error, number of epochs?
end

function ElmanEvaluate( network::ElmanNetwork, input::Vector{Float})
	# Check dimensions of inputs.

end

end
