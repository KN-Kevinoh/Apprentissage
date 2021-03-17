### A Pluto.jl notebook ###
# v0.12.16

using Markdown
using InteractiveUtils

# ╔═╡ 269e7c2a-32db-11eb-1287-5b2d54d033d1
begin
	using MLDatasets
	using ImageCore
end

# ╔═╡ 12987568-32dd-11eb-03d4-f9dd0aec968d
using Flux: onehotbatch

# ╔═╡ d995c874-32de-11eb-2c8e-611804211b65
using Flux

# ╔═╡ b6da123e-32e2-11eb-0ddf-b97a20610249
using Flux: @epochs

# ╔═╡ 684a12ce-32e3-11eb-07a2-ab9a667763ad
using PlutoUI

# ╔═╡ 8de08ba2-32e4-11eb-3151-39686f52379d
using Statistics

# ╔═╡ f69beab2-32d8-11eb-2229-ad6a513addda
# Flux for Julia
# BigDL for Java, Scala
# PyTorch, TensorFlow, MXNet, CNTK, etc... for Python

# ╔═╡ d49df6a6-32da-11eb-17cf-5dd4858aa997
# Today: MNIST with feed-forward ANN (instead of RNN, CNN,...) 

# ╔═╡ 3bf81516-32db-11eb-29b1-9d92f558bc0c
function readTrainingData(N)
	A = MNIST.traintensor(Float32, 1:N)
	X = Float32.(zeros(28*28, N))
	for i=1:N
		X[:,i] = vec(A[:,:,i])
	end
	y = MNIST.trainlabels(1:N) .+ 1
	(X, y)
end

# ╔═╡ 34271552-32dc-11eb-3090-898168278b47
N = 20000

# ╔═╡ 482c6cd2-32dc-11eb-06a1-abc7d03a0881
X_train, y_train = readTrainingData(N)

# ╔═╡ 50c94d24-32dc-11eb-2c47-dd816f361b32
function readTestData(N)
	A = MNIST.testtensor(Float32, 1:N)
	X = Float32.(zeros(28*28, N))
	for i=1:N
		X[:,i] = vec(A[:,:,i])
	end
	y = MNIST.testlabels(1:N) .+ 1
	(X, y)
end

# ╔═╡ 9c83b966-32dc-11eb-36f6-57d7ce99ec3c
M = 10000

# ╔═╡ b086beb8-32dc-11eb-268b-bdd08500be64
X_test, y_test = readTestData(M)

# ╔═╡ bb3d7de0-32dc-11eb-2767-3f68ce579450
first_x = MNIST.convert2image(MNIST.traintensor(1))

# ╔═╡ ff627838-32dc-11eb-2232-5f88fe7c1480
first_y = y_train[1]

# ╔═╡ 7335cfa8-32dd-11eb-07f5-4feb8ddb9ebb
y_train

# ╔═╡ 8d514c46-32dd-11eb-0da2-59b8a5c7e252
Y_train = onehotbatch(y_train, 1:10)

# ╔═╡ a650d0ca-32dd-11eb-2347-35a9d7daca19


# ╔═╡ 1fb9c1a8-32de-11eb-2817-d521facf7ab2
#model = Chain(Dense(784, 128, σ), Dense(128, 10), softmax)
#model = Chain(Dense(784, 10), softmax)
model = Chain(Dense(784, 128, σ), Dense(128, 32, σ), Dense(32,10), softmax)

# ╔═╡ e310bfe4-32de-11eb-3fc9-17d50e1c8edb
θ = Flux.params(model)

# ╔═╡ 241e91a2-32df-11eb-0bdc-117e8b017169
θ[1]

# ╔═╡ d534f4e2-32de-11eb-157b-654574fc67da
θ[2]

# ╔═╡ 55d459fa-32df-11eb-2a41-4137c9abf9ae
θ[3]

# ╔═╡ 641ae11c-32df-11eb-372e-bd9995210a84
θ[4]

# ╔═╡ 7aa65f62-32df-11eb-3a94-1315808275c7
first_prediction = model(X_train[:,1])

# ╔═╡ 0b4dd19e-32e0-11eb-0720-dfd9af599042
Flux.onecold(first_prediction)

# ╔═╡ 9bdc57da-32e0-11eb-1627-8d74c0fdff1a
loss(x, y) = Flux.crossentropy(model(x), y)

# ╔═╡ 0783fbbe-32e1-11eb-0e06-79e46b059388
initialLoss = loss(X_train, Y_train)

# ╔═╡ 7c8f08fe-32e1-11eb-2add-7d24e5c23f0f
model

# ╔═╡ 9933959c-32e1-11eb-242f-57a4da67a293
optimizer = ADAM()

# ╔═╡ ba93face-32e1-11eb-2eda-8b1e02bde31d
Flux.train!(loss, θ, [(X_train, Y_train)], optimizer)

# ╔═╡ 486f088e-32e2-11eb-3e5d-353d4192cc0d
θ[2]

# ╔═╡ 86f501bc-32e2-11eb-086a-b95ad56f59e0
θ[4]

# ╔═╡ 0d69278c-32e3-11eb-3717-0ddc31440b86
function train(numEpochs=20)
	with_terminal() do
		@epochs numEpochs Flux.train!(loss, θ, [(X_train, Y_train)], optimizer; cb = () -> println(loss(X_train[:,1:100], Y_train[:,1:100])))
	end
end

# ╔═╡ c5066cc4-32e3-11eb-1558-a9d2eb816fff
train(500)

# ╔═╡ d08ef908-32e3-11eb-19e2-a52b64f60b40
θ[1]

# ╔═╡ 0d3c79f2-32e4-11eb-22e2-4d3cd2fce143
θ[2]

# ╔═╡ 1043a5fa-32e4-11eb-1124-eba1cde2d888
θ[3]

# ╔═╡ 1841e51c-32e4-11eb-345a-2b2be8e27564
θ[4]

# ╔═╡ 1b17fbe6-32e4-11eb-37cf-35f0377b1991
initialLoss

# ╔═╡ 292f5f74-32e4-11eb-26cb-f9c2ce1f3d02
accuracy(x, y) = mean(Flux.onecold(model(x)) .== y)

# ╔═╡ 671c1a8e-32e4-11eb-118d-f1a18cef93f0
training_accuracy = accuracy(X_train, y_train)

# ╔═╡ 7f14d5ec-32e4-11eb-0c23-77f4e89afb77
test_accuracy = accuracy(X_test, y_test)

# ╔═╡ a8a2cef0-32e4-11eb-082b-1978bd182c81


# ╔═╡ Cell order:
# ╠═f69beab2-32d8-11eb-2229-ad6a513addda
# ╠═d49df6a6-32da-11eb-17cf-5dd4858aa997
# ╠═269e7c2a-32db-11eb-1287-5b2d54d033d1
# ╠═3bf81516-32db-11eb-29b1-9d92f558bc0c
# ╠═34271552-32dc-11eb-3090-898168278b47
# ╠═482c6cd2-32dc-11eb-06a1-abc7d03a0881
# ╠═50c94d24-32dc-11eb-2c47-dd816f361b32
# ╠═9c83b966-32dc-11eb-36f6-57d7ce99ec3c
# ╠═b086beb8-32dc-11eb-268b-bdd08500be64
# ╠═bb3d7de0-32dc-11eb-2767-3f68ce579450
# ╠═ff627838-32dc-11eb-2232-5f88fe7c1480
# ╠═12987568-32dd-11eb-03d4-f9dd0aec968d
# ╠═7335cfa8-32dd-11eb-07f5-4feb8ddb9ebb
# ╠═8d514c46-32dd-11eb-0da2-59b8a5c7e252
# ╠═a650d0ca-32dd-11eb-2347-35a9d7daca19
# ╠═d995c874-32de-11eb-2c8e-611804211b65
# ╠═1fb9c1a8-32de-11eb-2817-d521facf7ab2
# ╠═e310bfe4-32de-11eb-3fc9-17d50e1c8edb
# ╠═241e91a2-32df-11eb-0bdc-117e8b017169
# ╠═d534f4e2-32de-11eb-157b-654574fc67da
# ╠═55d459fa-32df-11eb-2a41-4137c9abf9ae
# ╠═641ae11c-32df-11eb-372e-bd9995210a84
# ╠═7aa65f62-32df-11eb-3a94-1315808275c7
# ╠═0b4dd19e-32e0-11eb-0720-dfd9af599042
# ╠═9bdc57da-32e0-11eb-1627-8d74c0fdff1a
# ╠═0783fbbe-32e1-11eb-0e06-79e46b059388
# ╠═7c8f08fe-32e1-11eb-2add-7d24e5c23f0f
# ╠═9933959c-32e1-11eb-242f-57a4da67a293
# ╠═ba93face-32e1-11eb-2eda-8b1e02bde31d
# ╠═486f088e-32e2-11eb-3e5d-353d4192cc0d
# ╠═86f501bc-32e2-11eb-086a-b95ad56f59e0
# ╠═b6da123e-32e2-11eb-0ddf-b97a20610249
# ╠═684a12ce-32e3-11eb-07a2-ab9a667763ad
# ╠═0d69278c-32e3-11eb-3717-0ddc31440b86
# ╠═c5066cc4-32e3-11eb-1558-a9d2eb816fff
# ╠═d08ef908-32e3-11eb-19e2-a52b64f60b40
# ╠═0d3c79f2-32e4-11eb-22e2-4d3cd2fce143
# ╠═1043a5fa-32e4-11eb-1124-eba1cde2d888
# ╠═1841e51c-32e4-11eb-345a-2b2be8e27564
# ╠═1b17fbe6-32e4-11eb-37cf-35f0377b1991
# ╠═8de08ba2-32e4-11eb-3151-39686f52379d
# ╠═292f5f74-32e4-11eb-26cb-f9c2ce1f3d02
# ╠═671c1a8e-32e4-11eb-118d-f1a18cef93f0
# ╠═7f14d5ec-32e4-11eb-0c23-77f4e89afb77
# ╠═a8a2cef0-32e4-11eb-082b-1978bd182c81
