module RNN

export Float
export TimeSeriesSample, TimeSeriesSamples, ActivationRule, defaultLearningRate, defaultActivationRule, logisticActivationRule

#if WORD_SIZE == 64	 # Float not generalized like Ints
if true
	typealias Float Float64
else
	typealias Float Float32
end

immutable TimeSeriesSample
	sample::Matrix{Float} # |Input or Output| x time Matrix
end

immutable TimeSeriesSamples
	samples::Vector{TimeSeriesSample} # Vector of samples.
		sizeSample::Int # The |Input or Output| of each sample.

		function TimeSeriesSamples( samples::Vector{TimeSeriesSample})
			if length( samples) == 0
				return new( samples, 0)
			end

			# Check that the sizes of each sample's Input or Output are equal.
			sizeSample = size( samples[1], 1)
			for sample in samples
				if size( sample, 1) != sizeSample
					error( "The sizes of each sample's Input or Output must be equal.")
				end
			end

			new( samples, sizeSample)
		end
end

immutable ActivationRule
	activationFunction::Function
	activationDerivative::Function
end

const defaultLearningRate = .05
const defaultMaxEpochs = typemax(Uint)
const defaultErrorThreshold = .0005

function defaultActivationRule()
	logisticActivationRule(1)
end

function logisticActivationRule(s)
	f(x) = 1/(1 + exp(-s * x))
	df(x) = s * x * (1 - x)
	ActivationRule(f, df)
end

function defaultErrorFunction()
	meanSquaredError
end

function meanSquaredError( output::Vector{Float}, target::Vector{Float})
	# Check that the two vectors of are equal lengths.
	if length( output) != length( target)
		error( "The length of the two vectors must be equal.")
	end

	sqrt( mapreduce(x->x^2, +, output - target))
end

end
