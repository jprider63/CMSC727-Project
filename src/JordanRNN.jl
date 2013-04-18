module JordanRNN
	using RNN

	export JordanNetwork, JordanTrain!, JordanEvaluate

	# Define the structure of a Jordan RNN
	type JordanNetwork
		# Weights between layers
		weightsHI::Matrix{Float}
		weightsHC::Matrix{Float}
		weightsOH::Matrix{Float}

		# Parameters
		eta::Float

		# Constants
		maxEpochs::Uint
		errorThreshold::Float

		# Functions
		activationRule::RNN.ActivationRule

		function JordanNetwork(n_inputs::Int, n_outputs::Int, n_hidden::Int, )
			weightsHI = rand(n_hidden, n_input) - .5
			weightsHC = rand(n_hidden, n_output) - .5
			weightsOH = rand(n_output, n_hidden) - .5
			new(weightsHI, weightsHC, weightsOH, RNN.defaultLearningRate, RNN.defaultMaxEpochs, RNN.defaultActivationRule)
		end
	end

	function JordanTrain!(network::JordanNetwork, )

	end

	function JordanEvaluate(network::JordanNetwork, input::Vector{Float})

	end
end
