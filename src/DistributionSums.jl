module DistributionSums

import Distributions
import Distributions.pdf
import Grid

type DistributionSum <: Distributions.Distribution
	interp::Grid.CoordInterpGrid{Float64,1,Grid.BCfill,Grid.InterpLinear,Tuple{FloatRange{Float64}}}
	truncatedprobability::Float64
end

function DistributionSum(numsamples::Int, bounds::Vector, dists...)
	@assert length(bounds) == 2
	xs = linspace(bounds[1], bounds[2], numsamples)
	pdfvals = Array(Float64, numsamples)
	convdist = Float64[]
	probability = 1.
	for d in dists
		map!(x->Distributions.pdf(d, x), pdfvals, xs)
		highprob = Distributions.cdf(d, bounds[2])
		lowprob = Distributions.cdf(d, bounds[1])
		sampleprob = sum(pdfvals) * ((bounds[2] - bounds[1]) / length(pdfvals))
		if isnan(highprob) || isnan(lowprob)
			probability *= sampleprob
		else
			probability *= (Distributions.cdf(d, bounds[2]) - Distributions.cdf(d, bounds[1]))
		end
		if abs(sampleprob - (highprob - lowprob)) > 0.001
			warn("samples may be insufficient error is $(100 * abs(sampleprob - (highprob - lowprob)))%")
		end
		if length(convdist) == 0
			convdist = copy(pdfvals)
		else
			convdist = conv(pdfvals, convdist)
		end
	end
	truncatedprobability = 1 - probability
	if truncatedprobability > 0.001
		warn("truncated probability is high: $truncatedprobability")
	end
	newbounds = length(dists) * bounds
	normfactor = (sum(convdist) - .5 * (convdist[1] + convdist[end])) * (newbounds[2] - newbounds[1]) / (length(convdist) - 1)
	scale!(convdist, 1 / normfactor)
	range = newbounds[1]:(newbounds[2] - newbounds[1]) / (length(convdist) - 1):newbounds[2]
	DistributionSum(Grid.CoordInterpGrid(range, convdist, 0., Grid.InterpLinear), truncatedprobability)
end

function pdf(d::DistributionSum, x::Real)
	max(0., d.interp[x])
end

end
