### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ bba57b68-1806-11eb-38b2-490678df00bb
using DelimitedFiles

# ╔═╡ 2ff43afc-1809-11eb-0918-2986efca10fd
using Statistics

# ╔═╡ c1202eda-1806-11eb-23da-b55bcbfb53ac
M = readdlm("/home/phuonglh/courses/ifi/dat/wdbc.txt", ',')

# ╔═╡ c74e95f8-1806-11eb-1925-4d0136516b2f
y = Int.(M[:, 2] .+ 1)

# ╔═╡ d47b3fec-1806-11eb-2500-39c87a4aa8f7
X = M[:, 3:12]

# ╔═╡ 2a7ba8e0-1808-11eb-2654-9f27422dfb88
N = length(y)

# ╔═╡ 3aa005ae-1808-11eb-229c-27280762bb63
K = length(unique(y))

# ╔═╡ 624c2c5e-1808-11eb-06c5-31504e6d5716
prior = zeros(K)

# ╔═╡ b2be42e4-1808-11eb-2b56-cdd9acba33fb
prior[1] = sum(y .== 1)/N

# ╔═╡ 4fb5d496-1808-11eb-3790-53231cdb0101
prior[2] = sum(y .== 2)/N

# ╔═╡ f0753ebc-1808-11eb-096b-f1a5b0e0a58f
X_1 = X[y .== 1, :]

# ╔═╡ 37c75a98-1809-11eb-3f5a-131e622515a0
μ_1 = mean(X_1, dims=1)

# ╔═╡ 77c822c6-1809-11eb-1e57-bdd1de200566
σ_1 = std(X_1, dims=1)

# ╔═╡ ad2ceac0-1809-11eb-39ed-6ff49bf07aad
X_2 = X[y .== 2, :]

# ╔═╡ b7222a98-1809-11eb-1ca5-b76a95e46c53
μ_2 = mean(X_2, dims=1)

# ╔═╡ ca2580b8-1809-11eb-2b38-1fdbbeba3cba
σ_2 = std(X_2, dims=1)

# ╔═╡ d3ac8212-1809-11eb-3fcd-97bd66182489
function train(X, y)
	N, D = size(X)
	K = length(unique(y))
	θ = zeros(K)   # prior distribution
	μ = zeros(K,D) # mean matrix
	σ = zeros(K,D) # standard deviation matrix
	for k=1:K
		θ[k] = sum(y .== k)/N
		X_k = X[y .== k, :]
		μ[k,:] = mean(X_k, dims=1)
		σ[k,:] = std(X_k, dims=1)
	end
	(θ, μ, σ)
end

# ╔═╡ 156b95f2-180b-11eb-3aea-89186c9003b4
θ, μ, σ = train(X, y)

# ╔═╡ 2965ab74-180b-11eb-0f1a-f176dea1c768
function classify(x, θ, μ, σ)
	K = length(θ)
	posterior = zeros(K)
	for k=1:K
		u = (x - μ[k,:]) ./ σ[k,:]
		posterior[k] = -sum(log.(σ[k,:])) - u'*u/2 + log(θ[k])
	end
	argmax(posterior)
end

# ╔═╡ 8e1f1f44-180d-11eb-2d26-7b751f55b505
prediction_100 = classify(X[100,:], θ, μ, σ)

# ╔═╡ acbf2534-180d-11eb-1abf-0fa5a4f293ff
prediction_500 = classify(X[500,:], θ, μ, σ)

# ╔═╡ 32054f2a-180e-11eb-11a7-9b1791085097
function evaluate(X, y, θ, μ, σ)
	N = length(y)
	ŷ = zeros(N)
	for i=1:N
		ŷ[i] = classify(X[i,:], θ, μ, σ)
	end
	sum(ŷ .== y)/N
end

# ╔═╡ a5582c68-180e-11eb-1d08-1f967a2dd410
evaluate(X, y, θ, μ, σ)

# ╔═╡ b60a4d50-180d-11eb-02c0-9f56d855a06c


# ╔═╡ Cell order:
# ╠═bba57b68-1806-11eb-38b2-490678df00bb
# ╠═c1202eda-1806-11eb-23da-b55bcbfb53ac
# ╠═c74e95f8-1806-11eb-1925-4d0136516b2f
# ╠═d47b3fec-1806-11eb-2500-39c87a4aa8f7
# ╠═2a7ba8e0-1808-11eb-2654-9f27422dfb88
# ╠═3aa005ae-1808-11eb-229c-27280762bb63
# ╠═624c2c5e-1808-11eb-06c5-31504e6d5716
# ╠═b2be42e4-1808-11eb-2b56-cdd9acba33fb
# ╠═4fb5d496-1808-11eb-3790-53231cdb0101
# ╠═f0753ebc-1808-11eb-096b-f1a5b0e0a58f
# ╠═2ff43afc-1809-11eb-0918-2986efca10fd
# ╠═37c75a98-1809-11eb-3f5a-131e622515a0
# ╠═77c822c6-1809-11eb-1e57-bdd1de200566
# ╠═ad2ceac0-1809-11eb-39ed-6ff49bf07aad
# ╠═b7222a98-1809-11eb-1ca5-b76a95e46c53
# ╠═ca2580b8-1809-11eb-2b38-1fdbbeba3cba
# ╠═d3ac8212-1809-11eb-3fcd-97bd66182489
# ╠═156b95f2-180b-11eb-3aea-89186c9003b4
# ╠═2965ab74-180b-11eb-0f1a-f176dea1c768
# ╠═8e1f1f44-180d-11eb-2d26-7b751f55b505
# ╠═acbf2534-180d-11eb-1abf-0fa5a4f293ff
# ╠═32054f2a-180e-11eb-11a7-9b1791085097
# ╠═a5582c68-180e-11eb-1d08-1f967a2dd410
# ╠═b60a4d50-180d-11eb-02c0-9f56d855a06c
