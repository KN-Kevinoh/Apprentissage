### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ 0cd54946-3e86-11eb-0c80-7b48049817a1
using Flux

# ╔═╡ bc2869dc-3e86-11eb-031c-7bdc98277ef0
using Statistics

# ╔═╡ 3940bcf8-3e87-11eb-0abf-09e69556bd67
using Plots

# ╔═╡ 7c5df234-3e8a-11eb-08c9-f10b3cd5c766
using LinearAlgebra

# ╔═╡ 2961602c-3e86-11eb-3854-91c93f13c4b7
A = Flux.Data.Iris.features()

# ╔═╡ 5c848a42-3e86-11eb-017a-5de2e5965c1a
X = convert(Array{Float64,2}, transpose(A))

# ╔═╡ ccfe5b90-3e86-11eb-2461-7dc5c34a7c02
mean(X, dims=1)

# ╔═╡ 154f2148-3e87-11eb-3a96-bfb9f8f635b7
feature_order = [1, 3, 2, 4]

# ╔═╡ 21139f74-3e87-11eb-02d6-c99c98a6d9e5
X[:, feature_order]

# ╔═╡ 539083e0-3e87-11eb-1deb-3f9836905161
plot(X[:,1], X[:,2], st=:scatter, legend=false)

# ╔═╡ 735824d0-3e87-11eb-3aaa-c7757a84f0c8
y = Flux.Data.Iris.labels()

# ╔═╡ 145362f0-3e88-11eb-3e27-5b0c51ca561f
labels = unique(y)

# ╔═╡ 17c8c83a-3e88-11eb-2672-8716dd2b92eb
begin
	setosa = (y .== labels[1])
	versicolor = (y .== labels[2])
	virginica = (y .== labels[3])
end

# ╔═╡ 50d1b77c-3e88-11eb-2841-7bbcdb50084d
plot(X[setosa,1], X[setosa,2], st=:scatter, m=:+, c=:black, legend=false)

# ╔═╡ 83e61450-3e88-11eb-3ac1-75e524b3bdd2
plot!(X[versicolor,1], X[versicolor,2], st=:scatter, m=:*, c=:red)

# ╔═╡ a495ff30-3e88-11eb-36a0-9387ded9f7ea
plot!(X[virginica,1], X[virginica,2], st=:scatter, m=:o, c=:yellow)

# ╔═╡ da425c96-3e88-11eb-07fd-4b907926b23b
μ = mean(X, dims=1)

# ╔═╡ e38a5024-3e88-11eb-2abf-d1c086a70cad
X

# ╔═╡ ebd19f9e-3e88-11eb-0e7a-23159aaa50a7
u = [1 2 3]

# ╔═╡ 20bd9762-3e89-11eb-21c1-31d2bd8db721
repeat(u, 10, 1)

# ╔═╡ 29228946-3e89-11eb-22e2-b58ff9bcbcc7
M = repeat(μ, 150, 1)

# ╔═╡ 59379252-3e89-11eb-0fbe-995562a130db
X̄ = X - M

# ╔═╡ 729a0bba-3e89-11eb-0a84-a3efb2026cad
mean(X̄, dims=1)

# ╔═╡ 81036674-3e89-11eb-3a00-676f651129dc
B = X̄[:, feature_order]

# ╔═╡ a8e362d4-3e89-11eb-0246-d1471421fb3f
begin
	plot(B[setosa,1], B[setosa,2], st=:scatter, m=:+, c=:black, legend=false)
	plot!(B[versicolor,1], B[versicolor,2], st=:scatter, m=:*, c=:red)
	plot!(B[virginica,1], B[virginica,2], st=:scatter, m=:o, c=:yellow, xlabel="Sepal Length", ylabel="Petal Length")
end

# ╔═╡ da5b9d9a-3e89-11eb-166d-ebb1faff5535
Σ = cov(B)

# ╔═╡ 17f49b20-3e8a-11eb-33bb-7f982a29dafe
B'*B/(length(y) - 1)

# ╔═╡ 8608e818-3e8b-11eb-074d-e3afa92f8e14
values, vectors = eigen(Σ)

# ╔═╡ ac8d442a-3e8b-11eb-2aff-8bd97f1e3810
values

# ╔═╡ d8b826dc-3e8b-11eb-2d92-b1601802bbf1
vectors

# ╔═╡ 3a4ee1e0-3e8c-11eb-3386-67649493cc24
values

# ╔═╡ 4fe146e4-3e8c-11eb-1c1d-6b34d9985e5f
order = sortperm(values, rev=true)

# ╔═╡ 643f69ae-3e8c-11eb-2807-a1d2afc2605d
θ = vectors[:, order[1:2]]

# ╔═╡ 7fa41f0a-3e8c-11eb-359a-c7b2155b2e57
Z = B*θ

# ╔═╡ a712535e-3e8c-11eb-3e40-c1f31d9d6a6c
begin
	plot(Z[setosa,1], Z[setosa,2], st=:scatter, m=:+, c=:black, legend=false)
	plot!(Z[versicolor,1], Z[versicolor,2], st=:scatter, m=:*, c=:red)
	plot!(Z[virginica,1], Z[virginica,2], st=:scatter, m=:o, c=:yellow, xlabel="1st component", ylabel="1nd component")
end

# ╔═╡ Cell order:
# ╠═0cd54946-3e86-11eb-0c80-7b48049817a1
# ╠═2961602c-3e86-11eb-3854-91c93f13c4b7
# ╠═5c848a42-3e86-11eb-017a-5de2e5965c1a
# ╠═bc2869dc-3e86-11eb-031c-7bdc98277ef0
# ╠═ccfe5b90-3e86-11eb-2461-7dc5c34a7c02
# ╠═154f2148-3e87-11eb-3a96-bfb9f8f635b7
# ╠═21139f74-3e87-11eb-02d6-c99c98a6d9e5
# ╠═3940bcf8-3e87-11eb-0abf-09e69556bd67
# ╠═539083e0-3e87-11eb-1deb-3f9836905161
# ╠═735824d0-3e87-11eb-3aaa-c7757a84f0c8
# ╠═145362f0-3e88-11eb-3e27-5b0c51ca561f
# ╠═17c8c83a-3e88-11eb-2672-8716dd2b92eb
# ╠═50d1b77c-3e88-11eb-2841-7bbcdb50084d
# ╠═83e61450-3e88-11eb-3ac1-75e524b3bdd2
# ╠═a495ff30-3e88-11eb-36a0-9387ded9f7ea
# ╠═da425c96-3e88-11eb-07fd-4b907926b23b
# ╠═e38a5024-3e88-11eb-2abf-d1c086a70cad
# ╠═ebd19f9e-3e88-11eb-0e7a-23159aaa50a7
# ╠═20bd9762-3e89-11eb-21c1-31d2bd8db721
# ╠═29228946-3e89-11eb-22e2-b58ff9bcbcc7
# ╠═59379252-3e89-11eb-0fbe-995562a130db
# ╠═729a0bba-3e89-11eb-0a84-a3efb2026cad
# ╠═81036674-3e89-11eb-3a00-676f651129dc
# ╠═a8e362d4-3e89-11eb-0246-d1471421fb3f
# ╠═da5b9d9a-3e89-11eb-166d-ebb1faff5535
# ╠═17f49b20-3e8a-11eb-33bb-7f982a29dafe
# ╠═7c5df234-3e8a-11eb-08c9-f10b3cd5c766
# ╠═8608e818-3e8b-11eb-074d-e3afa92f8e14
# ╠═ac8d442a-3e8b-11eb-2aff-8bd97f1e3810
# ╠═d8b826dc-3e8b-11eb-2d92-b1601802bbf1
# ╠═3a4ee1e0-3e8c-11eb-3386-67649493cc24
# ╠═4fe146e4-3e8c-11eb-1c1d-6b34d9985e5f
# ╠═643f69ae-3e8c-11eb-2807-a1d2afc2605d
# ╠═7fa41f0a-3e8c-11eb-359a-c7b2155b2e57
# ╠═a712535e-3e8c-11eb-3e40-c1f31d9d6a6c
