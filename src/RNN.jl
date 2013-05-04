module RNN

export TimeSeriesSample, TimeSeriesSamples, ActivationRule, defaultLearningRate, defaultActivationRule, logisticActivationRule, ErrorFunction

immutable TimeSeriesSample
	sample::Matrix{Float64} # |Input or Output| x time Matrix
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

# Define the type of error function. Both vectors must have equal lengths.
# Vector{Float64} * Vector{Float64} -> Float64
# TODO: Once Julia supports it, enforce this type on the function.
typealias ErrorFunction Function

immutable ActivationRule
	activationFunction::Function # TODO: Once Julia supports it, enforce the type on the function.
	activationDerivative::Function # TODO: Once Julia supports it, enforce the type on the function.
end

const defaultLearningRate = .05
const defaultMaxEpochs = typemax(Uint)
const defaultErrorThreshold = .5

function defaultActivationRule()
	logisticActivationRule(1)
end

function logisticActivationRule(s)
	f(x) = 1/(1 + exp(-s * x))
	df(x) = s * x * (1 - x)
	ActivationRule(f, df)
end

function defaultErrorFunction()
	L2NormError
end

# Compute the L2 norm error.
function L2NormError( output::Vector{Float64}, target::Vector{Float64})
	# Check that the two vectors of are equal lengths.
	if length( output) != length( target)
		error( "The length of the two vectors must be equal.")
	end

	sqrt( mapreduce(x->x^2, +, output - target))
end

end
