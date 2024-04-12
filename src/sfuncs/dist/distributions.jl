import Distributions

export DistributionsSF

struct DistributionsSF{D <: Distributions.Distribution, O} <: Dist{O}
    dist::D
    function DistributionsSF(dist::D) where {D <: Distributions.Distribution}
        O = eltype(D)
        return new{D, O}(dist)
    end
end

@impl begin
  function expectation(sf::DistributionsSF, i::Tuple{})
      return Distributions.mean(sf.dist)
  end
end

@impl begin
    function sample(sf::DistributionsSF, i::Tuple{})
        return Distributions.rand(sf.dist)
    end
end

@impl begin
    function variance(sf::DistributionsSF{D}, i::Tuple{}) where {D <: Distributions.ContinuousDistribution}
        return Distributions.std(sf.dist)^2
    end
end

# See https://juliastats.org/Distributions.jl/stable/convolution/
ConvSupported = Union{Distributions.Bernoulli,
                      Distributions.Binomial,
                      Distributions.NegativeBinomial,
                      Distributions.Geometric,
                      Distributions.Poisson,
                      Distributions.Normal,
                      Distributions.Cauchy,
                      Distributions.Chisq,
                      Distributions.Exponential,
                      Distributions.Gamma,
                      Distributions.MvNormal}

@impl begin
    function sumsfs(fs::NTuple{N, DistributionsSF{SubSF}}) where {N, SubSF <: ConvSupported}
        # Return an SFunc representing g(x) = f1(x) + f2(x) + ...
        # I.e. convolution of the respective densities
        dists = tuple((f.dist for f in fs)...)
        return DistributionsSF(reduce(Distributions.convolve, dists))
    end
end
