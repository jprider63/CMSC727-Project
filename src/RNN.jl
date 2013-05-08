module RNN

export TimeSeriesSample, TimeSeriesSamples, ActivationRule, defaultLearningRate, defaultMaxEpochs, defaultErrorThreshold, defaultActivationRule, logisticActivationRule, ErrorFunction, defaultErrorFunction, importData

# Current julia build on my machine doesn't support immutable (0.1.x)
# immutable TimeSeriesSample
type TimeSeriesSample
	sample::Matrix{Float64} # |Input or Output| x time Matrix
end

# Current julia build on my machine doesn't support immutable (0.1.x)
# immutable TimeSeriesSamples
type TimeSeriesSamples
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

# Current julia build on my machine doesn't support immutable (0.1.x)
# immutable ActivationRule
type ActivationRule
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

# Imports a specified time series dataset
# Other than the household power dataset, all are in same format
function importData(datasetName)
	filesep = "/"
	if OS_NAME != ":Windows"
		filesep = "\\"
	end
	testPath = pwd() * filesep * ".." * filesep * "test" * filesep

	if datasetName == "household"
		csvPath = testPath * "household_power_consumption.txt"
		data = readdlm(csvPath, ';', Any)
		# Strips non float values (1st and 2nd column)
		data = data[:, 3:]
	else
		csvPath = testPath * datasetName * ".csv"
		data = readcsv(csvPath, Float64)
	end

	data = data[2:, :]	# Strips column names (1st row)
	return data
end

end
