module RNN

export TimeSeriesSample, TimeSeriesSamples, ActivationRule, defaultLearningRate, defaultActivationRule, logisticActivationRule

type TimeSeriesSample
    sample::Matrix{Float}
end

type TimeSeriesSamples
    samples::Vector{TimeSeriesSample}
end

type ActivationRule
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

end
