export
    Table
    
"""
mutable struct Table{NumInputs, I <: NTuple{NumInputs, Any}, J, K, O, S <: SFunc{J,O}} <: Conditional{I, J, K, O, S}

`Table` defines a `Conditional` whose function is given by a multi-dimensional table of type 
`Array{Q, NumInputs}`, where `Q` is the output type of the internal *sfunc* and `NumInputs`
is the count of incoming parent values.

See also:  [`Conditional`](@ref), [`DiscreteCPT`](@ref), [`CLG`](@ref)
"""
mutable struct Table{NumInputs, I <: NTuple{NumInputs, Any}, J, K, O, S <: SFunc{J,O}} <: Conditional{I, J, K, O, S}
    icombos :: Vector{I}
    iranges :: NTuple{NumInputs, Array{Any, 1}}
    isizes :: NTuple{NumInputs, Int}
    imults :: NTuple{NumInputs, Int}
    inversemaps :: NTuple{NumInputs, Dict{Any, Int}}
    sfs :: Array{S, NumInputs}
    """
        function Table(J, O, NumInputs::Int, paramdict, sfmaker:Function) where {J, O, S <: SFunc{J,O}}

    `Table` constructor.  
    
    # Arguments
    - `J` the type of inputs to the sfuncs in the table
    - 'O' the type of outputs from the sfuncs in the table
    - `NumInputs::Int`  the count of incoming parent values  
    - `paramdict` see [`DiscreteCPT`](@ref) and [`CLG`](@ref) for examples
    - `sfmaker` a function from Q to S
    """
    function Table(J, O, NumInputs::Int, paramdict::Dict{I,Q}, sfmaker::Function) where {I, Q}
        K = extend_tuple_type(I,J)
        icombos = keys(paramdict)
        iranges :: Array{Array{Any,1}} = [unique(collect([combo[k] for combo in icombos])) for k in 1:NumInputs]
        isizes = tuple([length(irange) for irange in iranges]...)
        m = 1
        ims = zeros(Int, NumInputs)
        for k in NumInputs:-1:1
            ims[k] = m
            m *= isizes[k]
        end
        imults = tuple(ims...)
        # TODO: Fix ordering of Dict
        inversemaps = tuple([Dict([x => i for (i,x) in enumerate(irange)]) for irange in iranges]...)
        sortedcombos = [tuple(p...) for p in cartesian_product(iranges)]
        S = SFunc{J,O}
        sfs = Array{S, NumInputs}(undef, isizes)
        for k in 1:length(sortedcombos)
            is = sortedcombos[k]
            q = paramdict[is] 
            sfs[k] = sfmaker(q)
        end
        new{NumInputs, I, J, K, O, S}(sortedcombos, tuple(iranges...), isizes, imults, inversemaps, sfs)
    end
end

#=
function dict2tableparams(sf::Table{NumInputs,I,J,K,O,S}, p::Dict{I,Q}) where {NumInputs,I,J,K,O,Q,S}
    k1 = collect(keys(p))[1]
    @assert NumInputs == length(k1)

    icombos = keys(p)
    iranges :: Array{Array{Any,1}} = [unique(collect([combo[k] for combo in icombos])) for k in 1:NumInputs]
    sortedcombos = [tuple(p...) for p in cartesian_product(iranges)]
    isizes = tuple([length(irange) for irange in iranges]...)
    array = Array{Q, NumInputs}(undef, isizes)
    for k in 1:length(sortedcombos)
        is = sortedcombos[k]
        array[k] = get(p, is, sf.default) # The input dict does not have to have all combos
    end
    array
end
=#

function gensf(t::Table{N,I,J,K,O,S}, parvals::NTuple{N,Any}) where {N,I,J,K,O,S}
    inds = tuple([t.inversemaps[k][parvals[k]] for k in 1:length(parvals)]...)
    i = 1
    for k in 1:N
        i += (inds[k] - 1) * t.imults[k]
    end
    return t.sfs[i]# Change this to just index on the tuple of indices rather than do the calculation ourselves
end
#=
function do_maximize_stats(t::Table{N,I,J,K,O,Q,S}, sfmaximizers) where {N,I,J,K,O,Q,S}
    result = Array{Q, N}(undef, t.isizes)
    # Since this is a table, the new parameters have to have an entry for each of the original parent values
    for k in 1:length(t.params)
        is = t.icombos[k]
        result[k] = get(sfmaximizers, is, t.default)
    end
    return result
end
=#
