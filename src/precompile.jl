function _precompile_()
   # Base.precompile(Tuple{typeof(Base.show), FixedEffectModel}) (not working)
   let fbody = try __lookup_kwbody__(which(reg, (DataFrame,FormulaTerm,CovarianceEstimator,))) catch missing end
       if !ismissing(fbody)
           precompile(fbody, (Dict{Symbol, Any},Nothing,Symbol,Symbol,Int64,Bool,Float64,Int64,Bool,Bool,Int64,Nothing,Bool,typeof(reg),DataFrame,FormulaTerm,CovarianceEstimator,))
       end
   end
   Base.precompile(Tuple{typeof(compute_linearly_independent), Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Symbol}})
   Base.precompile(Tuple{typeof(basis),AbstractVector{T} where T,Vararg{AbstractVector{T} where T, N} where N})
   Base.precompile(Tuple{typeof(undo_perm),Vector{Float64},Symmetric{Float64, Matrix{Float64}},Vector{Int64}})
   Base.precompile(Tuple{typeof(add_omitted_variables),Vector{Float64},Symmetric{Float64, Matrix{Float64}},BitVector})
   Base.precompile(Tuple{typeof(compute_Fkp_pkp),Vector{Matrix{Float64}},Vector{Matrix{Float64}},Vector{Matrix{Float64}},Int64,Int64,Vcov.SimpleCovariance,Bool})
   Base.precompile(Tuple{typeof(tss),Vector{Float64},Bool,UnitWeights{Int64}})
   Base.precompile(Tuple{typeof(apply_schema),FormulaTerm{Term, Tuple{InterceptTerm{true}, Term}},StatsModels.Schema,Type{FixedEffectModel},Bool})
   Base.precompile(Tuple{typeof(_parse_fixedeffect),DataFrame,InteractionTerm})
   Base.precompile(Tuple{typeof(_parse_fixedeffect),DataFrame,FormulaTerm})
   Base.precompile(Tuple{typeof(parse_fixedeffect),DataFrame,FormulaTerm})
   Base.precompile(Tuple{typeof(_multiply), DataFrame, Vector{Symbol}})
   Base.precompile(Tuple{typeof(Fstat), Vector{Float64}, Symmetric{Float64, Matrix{Float64}}, Bool})
   Base.precompile(Tuple{typeof(reg),DataFrame,FormulaTerm})
   Base.precompile(Tuple{typeof(crossprod),Vector{SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}})
   Base.precompile(Tuple{typeof(_helper),Any,CovarianceEstimator,CovarianceEstimator,FormulaTerm,FormulaTerm,Vector{Float64},UnitWeights{Int64},Bool,Int64,Vector{Float64},Matrix{Float64},Matrix{Float64},Matrix{Float64},Vector{Matrix{Float64}},Vector{Matrix{Float64}},Vector{Matrix{Float64}},BitVector,Vector{Float64},Float64,Int64,Vector{Symbol},Vector{FixedEffect},Nothing,Int64,Int64,Cholesky{Float64, Matrix{Float64}},Bool,Bool,Bool,Bool,Float64,Float64,Vector{Int64},Vector{String},String,Dict{Symbol, Any},Int64,Bool})
end
