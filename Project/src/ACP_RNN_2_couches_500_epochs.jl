### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 46b49206-58a4-11eb-0cdb-a7731f0116fc
# Importation des librairies

begin
	using DelimitedFiles
	using Plots
	using Statistics
	using LinearAlgebra
	using Flux
	
	using Flux: onehotbatch
	using Flux: @epochs
	using PlutoUI
end

# ╔═╡ bfe83b76-58a3-11eb-1e56-897ec0cd82c3
# Projet TP groupe 3
# PCA_ANN_2_couches

# ╔═╡ 15104d1a-58a4-11eb-1b25-718953011806
# Nous avons utilisé les étapes suivantes
# 1. Réduction de la dimension des variables avec PCA
# 2. Nous avons utilisé un ANN pour notre apprentissage

# ╔═╡ b7cade82-58a4-11eb-1f69-61cb2ac9488f
# Importation des données d'entrainement

# ╔═╡ cf0deee8-58a4-11eb-1f90-a78be058e0a0
begin
	path_dataset = "dat/"
	dataset_train = readdlm(path_dataset * "train/X_train_new.txt", ' ')
	X_train = dataset_train[:, 1:561]
	y_train = Int.(dataset_train[:,563])
end

# ╔═╡ ebb0f286-58a4-11eb-19c2-99b4e7c1cc7f
X_train

# ╔═╡ 00fcac48-58a5-11eb-0096-63ce72b46278
y_train

# ╔═╡ f1fab550-58a4-11eb-3029-93b4ac8d49f0
# Importation des données de test

# ╔═╡ 1730bdce-58a5-11eb-381e-1bf3405642ba
begin
	dataset_test = readdlm(path_dataset * "test/X_test_new.txt", ' ')
	X_test = dataset_test[:, 1:561]
	y_test = Int.(dataset_test[:,563])
end

# ╔═╡ 356c5b7c-58a5-11eb-0c5c-4d8597495fe9
X_test

# ╔═╡ 38f5c4c2-58a5-11eb-097f-c759bbebe968
y_test

# ╔═╡ 3b86d6b8-58a5-11eb-1332-d7e01814352a
md"#### Utilisation de PCA pour réduire la dimension des données

Algorithme :

1. Centrer les données
2. Construire la matrice de covariance Σ
3. Décomposer cette matrice en vecteur propres,valeur propres {vi, λi}
4. Ordonner les valeurs propres par ordre décroissant
5. Le sous-espace de dimension q qui représente 99% de la variance des données est utilisé pour le modèle ANN
"

# ╔═╡ 1d592672-58a6-11eb-2472-c133b36b24cc
# Calcul de la moyenne de chaque colonne
μ = mean(X_train, dims=1)

# ╔═╡ 38f683fc-58a6-11eb-3100-a70b8f1062da
M = repeat(μ, length(y_train), 1)

# ╔═╡ 474b3876-58a6-11eb-1bfe-1597ae5397fc
# Centrage des données
X̄ = X_train - M

# ╔═╡ 556b9e78-58a6-11eb-32e3-752ec655ac8d
mean(X̄, dims=1) # La moyenne de chaque colonne est bien proche de 0

# ╔═╡ 5e667b4c-58a6-11eb-1fc3-5f3311e2ed6a
# Construction la matrice de covariance Σ
Σ = cov(X̄) # Ou B'*B/(length(y_train) - 1)

# ╔═╡ 6f97f6fc-58a6-11eb-3102-6fb3e70764f9
# Décomposer cette matrice en vecteur propres,valeur propres {vi, λi}
λ, v = eigen(Σ)

# ╔═╡ 7e4c3d34-58a6-11eb-0441-3df026a63265
# Ordonner les valeurs propres par ordre décroissant
order = sortperm(λ, rev=true)

# ╔═╡ 87702420-58a6-11eb-2dd2-c900bcaef44b
# Recherche du nombre de composant permettant de conserver au moins 99% de la variance
begin
	eboli = float([0 for idx in 1:561])
	λ_drec = λ[order]
	total = sum(λ_drec)
	for i in 1:561
	    som_i = sum(λ_drec[1:i])
	    eboli[i] = som_i / total * 100
	end
end

# ╔═╡ a6b4b18e-58a6-11eb-0874-ad269ae76dee
plot(1:561, eboli, legend=false, xlabel="Numbre de composantes", ylabel="Pourcentage variance expliquée", title="Variance expliquée en fonction du nombre de composante")

# ╔═╡ d9296c06-58a6-11eb-1ef6-89fc58eb3979
begin
	print("Le pourcentage de la variance conservée avec 155 composantes est : ")
	print(eboli[155])
end

# Nous alons donc utiliser 155 composantes

# ╔═╡ f6124458-58a6-11eb-24e8-137e98ad3dc1
# Le sous-espace de dimension q qui représente 99 de la variance
θ = v[:, order[1:155]]

# ╔═╡ 05d50c4a-58a7-11eb-19f3-27fe881e279a
X_train_acp = X̄ * θ

# ╔═╡ 1eabc7fe-58a7-11eb-1394-dbe6f87c06bb
begin
    WALKING = (y_train .== 1)
    WALKING_UPSTAIRS = (y_train .== 2)
    WALKING_DOWNSTAIRS = (y_train .== 3)
    SITTING = (y_train .== 4)
    STANDING = (y_train .== 5)
    LAYING = (y_train .== 6)
end

# ╔═╡ 2b7932f0-58a7-11eb-0687-09328872cfbe
# Affichage des 2 premières composantes

begin
    θ_afich = v[:, order[1:4]]
    Z = X̄ * θ
    plot(Z[WALKING, 1], Z[WALKING, 2], st=:scatter, size =(750, 500), legend=true, label="Walking", m=:utriangle, c=:black, xlabel="1st component", ylabel="2nd component")
    plot!(Z[WALKING_UPSTAIRS, 1], Z[WALKING_UPSTAIRS, 2], st=:scatter, legend=true, label="Walking upstairs", m=:star7, c=:red)
    plot!(Z[WALKING_DOWNSTAIRS, 1], Z[WALKING_DOWNSTAIRS, 2], st=:scatter, legend=true, label="Walking downstair", m=:o, c=:yellow)
    plot!(Z[SITTING, 1], Z[SITTING, 2], st=:scatter, legend=true, label="Sitting", m=:heptagon, c=:blue)
    plot!(Z[STANDING, 1], Z[STANDING, 2], st=:scatter, legend=true, label="Standing", m=:diamond, c=:green)
    plot!(Z[LAYING, 1], Z[LAYING, 2], st=:scatter, legend=true, label="Laying", m=:a, c=:orange)
end

# ╔═╡ 501fa6f2-58a7-11eb-27f5-a5258f747210
# Affichage des composantes 3 et 4

begin
    plot(Z[WALKING, 3], Z[WALKING, 4], st=:scatter, size =(750, 500), legend=true, label="Walking", m=:utriangle, c=:black, xlabel="3th component", ylabel="4th component")
    plot!(Z[WALKING_UPSTAIRS, 3], Z[WALKING_UPSTAIRS, 4], st=:scatter, legend=true, label="Walking upstairs", m=:star7, c=:red)
    plot!(Z[WALKING_DOWNSTAIRS, 3], Z[WALKING_DOWNSTAIRS, 4], st=:scatter, legend=true, label="Walking downstair", m=:o, c=:yellow)
    plot!(Z[SITTING, 3], Z[SITTING, 4], st=:scatter, legend=true, label="Sitting", m=:heptagon, c=:blue)
    plot!(Z[STANDING, 3], Z[STANDING, 4], st=:scatter, legend=true, label="Standing", m=:diamond, c=:green)
    plot!(Z[LAYING, 3], Z[LAYING, 4], st=:scatter, legend=:bottomright, label="Laying", m=:a, c=:orange)
end

# ╔═╡ 9e6e3d32-58a7-11eb-2c8c-9bb8cdaaec1a
md"### Utilisation d'un modèle RNN"

# ╔═╡ 9d1b9290-58a7-11eb-3e88-1397b99cb5f3
# Application du OneHot encoding sur les labels d'apprentissage
Y_train = onehotbatch(y_train, 1:6)

# ╔═╡ de2a0820-58a7-11eb-263c-c38b71c5e274
# Création du modèle
model = Chain(Dense(155, 64, σ), Dense(64,6), softmax)

# ╔═╡ fc480456-58a7-11eb-238a-3bd6b26d2270
# Initialisation des paramètres
θ_rnn = Flux.params(model)

# ╔═╡ 20aec9ce-58a8-11eb-10d6-d5f7e5c38759
# Création de la fonction de perte (Entropie croisée)
loss(x, y) = Flux.crossentropy(model(x), y)

# ╔═╡ 3c1dbd0a-58a8-11eb-08e1-9b6c0b94b027
# Perte initiale
initialLoss = loss(X_train_acp', Y_train)

# ╔═╡ 5ec9c5ec-58a8-11eb-0c30-f11b0e50b069
# Création de l'optimizer (Adam)
optimizer = ADAM()

# ╔═╡ 6ef4fb6c-58a8-11eb-0aa6-5f8a01397d54
# Fonction d'entrainement
function train(numEpochs=20)
    with_terminal() do
        @epochs numEpochs Flux.train!(loss, θ_rnn, [(X_train_acp', Y_train)], optimizer; cb = () -> println(loss(X_train_acp'[:,1:100], Y_train[:,1:100])))
    end
end

# ╔═╡ eb67b2a0-58a8-11eb-18fb-5367f3b4c2b6
train(500)

# ╔═╡ 8ef0508c-58a9-11eb-0772-9d231c528cc0
accuracy(x, y) = mean(Flux.onecold(model(x)) .== y)

# ╔═╡ b1697e40-58a9-11eb-1fc9-21cf80e8c1d5
training_accuracy = accuracy(X_train_acp', y_train)

# ╔═╡ e72c5bec-58a9-11eb-0aba-53deb88892e9


# ╔═╡ 50830bfe-58aa-11eb-2e07-09d825294976
# Application de acp sur les données de test
begin
	M_test = repeat(μ, length(y_test), 1)
	X̄_test = X_test - M_test
	X_test_acp = X̄_test * θ
end

# ╔═╡ de0c8d6a-58aa-11eb-327a-53ce2995aca0
test_accuracy = accuracy(X_test_acp', y_test)

# ╔═╡ Cell order:
# ╠═bfe83b76-58a3-11eb-1e56-897ec0cd82c3
# ╠═15104d1a-58a4-11eb-1b25-718953011806
# ╠═46b49206-58a4-11eb-0cdb-a7731f0116fc
# ╠═b7cade82-58a4-11eb-1f69-61cb2ac9488f
# ╠═cf0deee8-58a4-11eb-1f90-a78be058e0a0
# ╠═ebb0f286-58a4-11eb-19c2-99b4e7c1cc7f
# ╠═00fcac48-58a5-11eb-0096-63ce72b46278
# ╠═f1fab550-58a4-11eb-3029-93b4ac8d49f0
# ╠═1730bdce-58a5-11eb-381e-1bf3405642ba
# ╠═356c5b7c-58a5-11eb-0c5c-4d8597495fe9
# ╠═38f5c4c2-58a5-11eb-097f-c759bbebe968
# ╠═3b86d6b8-58a5-11eb-1332-d7e01814352a
# ╠═1d592672-58a6-11eb-2472-c133b36b24cc
# ╠═38f683fc-58a6-11eb-3100-a70b8f1062da
# ╠═474b3876-58a6-11eb-1bfe-1597ae5397fc
# ╠═556b9e78-58a6-11eb-32e3-752ec655ac8d
# ╠═5e667b4c-58a6-11eb-1fc3-5f3311e2ed6a
# ╠═6f97f6fc-58a6-11eb-3102-6fb3e70764f9
# ╠═7e4c3d34-58a6-11eb-0441-3df026a63265
# ╠═87702420-58a6-11eb-2dd2-c900bcaef44b
# ╠═a6b4b18e-58a6-11eb-0874-ad269ae76dee
# ╠═d9296c06-58a6-11eb-1ef6-89fc58eb3979
# ╠═f6124458-58a6-11eb-24e8-137e98ad3dc1
# ╠═05d50c4a-58a7-11eb-19f3-27fe881e279a
# ╠═1eabc7fe-58a7-11eb-1394-dbe6f87c06bb
# ╠═2b7932f0-58a7-11eb-0687-09328872cfbe
# ╠═501fa6f2-58a7-11eb-27f5-a5258f747210
# ╠═9e6e3d32-58a7-11eb-2c8c-9bb8cdaaec1a
# ╠═9d1b9290-58a7-11eb-3e88-1397b99cb5f3
# ╠═de2a0820-58a7-11eb-263c-c38b71c5e274
# ╠═fc480456-58a7-11eb-238a-3bd6b26d2270
# ╠═20aec9ce-58a8-11eb-10d6-d5f7e5c38759
# ╠═3c1dbd0a-58a8-11eb-08e1-9b6c0b94b027
# ╠═5ec9c5ec-58a8-11eb-0c30-f11b0e50b069
# ╠═6ef4fb6c-58a8-11eb-0aa6-5f8a01397d54
# ╠═eb67b2a0-58a8-11eb-18fb-5367f3b4c2b6
# ╠═8ef0508c-58a9-11eb-0772-9d231c528cc0
# ╠═b1697e40-58a9-11eb-1fc9-21cf80e8c1d5
# ╠═e72c5bec-58a9-11eb-0aba-53deb88892e9
# ╠═50830bfe-58aa-11eb-2e07-09d825294976
# ╠═de0c8d6a-58aa-11eb-327a-53ce2995aca0
