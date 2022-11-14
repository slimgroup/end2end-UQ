# author: Ziyi Yin, ziyi.yin@gatech.edu 
## Calibration step

using DrWatson
@quickactivate "End-to-end-hint3"

using FNO4CO2
using PyPlot
using Flux, Random
using MAT, Statistics, LinearAlgebra
using ProgressMeter, JLD2
using JUDI
using SlimPlotting
using InvertibleNetworks
using BSON
using ImageQualityIndexes 
using UNet

Random.seed!(2022)
matplotlib.use("agg")

# load the network
JLD2.@load "../data/3D_FNO/batch_size=2_dt=0.02_ep=300_epochs=1000_learning_rate=0.0001_modes=4_nt=51_ntrain=1000_nvalid=100_s=1_width=20.jld2";
NN = deepcopy(NN_save);
Flux.testmode!(NN, true);

# load the CNF network

# Create summary network
device = cpu
sum_net = true

# Create conditional network
L = 3
K = 9 
n_hidden = 64
low = 0.5f0

Random.seed!(123);
model_path = "../data/K=9_L=3_batch_size=10_clipnorm_val=10.0_e=285_lr=0.004_n_hidden=64_n_in=1_n_train=1000_noise_lev_x=0.01_sum_net=false.jld2"# unet_model
Params_trained = JLD2.load(model_path)["Params"]
K = JLD2.load(model_path)["K"]
L = JLD2.load(model_path)["L"]
n_in = JLD2.load(model_path)["n_in"]
n_hidden = JLD2.load(model_path)["n_hidden"]
e = JLD2.load(model_path)["e"]

G = NetworkConditionalGlow(1, n_in, n_hidden,  L, K; rb_activation=ReLUlayer(), split_scales=true, activation=SigmoidLayer(low=low,high=1.0f0), logdet=false);

p_curr = get_params(G);
for i in 1:length(p_curr)
	p_curr[i].data = Params_trained[i].data
end
G = G |> device;

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")
grad_path = datadir("training-data", "nrec=960_nsample=1100_nsrc=32_nssample=4_ntrain=1000_nv=5_nvalid=100_snr=10.0_upsample=2.jld2")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/eqre95eqggqkdq2/'
        'perm_gridspacing15.0.mat -q -O $perm_path`)
end
if ~isfile(conc_path)
    run(`wget https://www.dropbox.com/s/b5zkp6cw60bd4lt/'
        'conc_gridspacing15.0.mat -q -O $conc_path`)
end

perm = matread(perm_path)["perm"];
conc = matread(conc_path)["conc"];
grad = JLD2.load(grad_path)["gset"];

function posterior_sampler(G, y, size_x; device=gpu, num_samples=1, batch_size=16)
    # make samples from posterior for train sample 
	X_forward = randn(Float32, size_x[1:end-1]...,batch_size) |> device
    Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
    Zy_fixed_train = G.forward(X_forward, Y_train_latent_repeat)[2]; #needs to set the proper sizes here

    X_post_train = zeros(Float32, size_x[1:end-1]...,num_samples)
    for i in 1:div(num_samples, batch_size)
    	ZX_noise_i = randn(Float32, size_x[1:end-1]...,batch_size)|> device
   		X_post_train[:,:,:, (i-1)*batch_size+1 : i*batch_size] = G.inverse(
        	ZX_noise_i,
        	Zy_fixed_train
    		)[1] |> cpu;
	end
	return X_post_train
end

mutable struct ReverseConditionNet
    G::InvertibleNetwork
    zy::Array{Float32, 4}
end

using Zygote: @adjoint
function (G::ReverseConditionNet)(zx::AbstractArray{Float32, 4})
    return reverse(G.G).forward(zx, G.zy)[1]
end
@adjoint function (G::ReverseConditionNet)(zx::AbstractArray{Float32, 4})
    X, Y = reverse(G.G).forward(zx, G.zy)
    return X, Δ -> (nothing, reverse(G.G).backward(Δ,X,Y)[1])
end
function (G::ReverseConditionNet)(zx::Matrix{Float32})
    return reverse(G.G).forward(reshape(zx, size(zx, 1), size(zx, 2), 1, 1), G.zy)[1]
end
@adjoint function (G::ReverseConditionNet)(zx::Matrix{Float32})
    X, Y = reverse(G.G).forward(reshape(zx, size(zx, 1), size(zx, 2), 1, 1), G.zy)
    return X, Δ -> (nothing, reverse(G.G).backward(Δ,X,Y)[1][:,:,1,1])
end

# physics grid
n1 = (64, 64)
d1 = 1f0 ./ n
nt = 51
dt = 0.02f0
grid = gen_grid(n1, d1, nt, dt)

# take a test sample
x_true = perm[:,:,ntrain+nvalid+1];  # take a test sample
y_true = conc[:,:,:,ntrain+nvalid+1];

# observation vintages
nv = 5
survey_indices = Int.(round.(range(1, stop=22, length=nv)))
sw_true = y_true[survey_indices,:,:]; # ground truth CO2 concentration at these vintages

# rock properties
vp = 3500 * ones(Float32,n1)     # p-wave
phi = 0.25f0 * ones(Float32,n1)  # porosity
rho = 2200 * ones(Float32,n1)    # density

## upsampling
upsample = 2
n = n1 .* upsample
d = (15f0, 15f0) ./ upsample
o = (0f0, 0f0)
snr = 1f1
u(x::Vector{Matrix{Float32}}) = [repeat(x[i], inner=(upsample,upsample)) for i = 1:length(x)]

### fluid-flow physics (FNO)
S(x::AbstractMatrix{Float32}) = permutedims(relu01(NN(perm_to_tensor(x, grid, AN)))[:,:,survey_indices,1], [3,1,2])

### rock physics
R(c::AbstractArray{Float32,3}) = Patchy(c,vp,rho,phi)[1]

### source subsampling
nssample = 4
nsrc = 32       # num of sources
nrec = 960      # num of receivers

function add_noise(d_obs, snr)
    ## add noise
    noise_ = deepcopy(d_obs)
    for i = 1:length(d_obs)
        for j = 1:d_obs[i].nsrc
            noise_[i].data[j] = randn(Float32, size(noise_[i].data[j]))
        end
    end
    snr = 10f0
    noise_ = noise_/norm(noise_) *  norm(d_obs) * 10f0^(-snr/20f0)
    d_obs = d_obs + noise_
    σ = Float32.(norm(noise_)/sqrt(length(vcat(noise_...))))
    return d_obs
end

# where to take the gradient
x_init = mean(perm[:,:,1:ntrain], dims=3)[:,:,1]

# set up rock physics
vp_stack = Patchy(sw_true,vp,rho,phi)[1]   # time-varying vp
vp_stack_up = u(vp_stack)

extentx = (n[1]-1)*d[1] # width of model
extentz = (n[2]-1)*d[2] # depth of model

model = [Model(n, d, o, (1f3 ./ vp_stack_up[i]).^2f0; nb = 160) for i = 1:nv]   # wave model

timeS = timeR = 750f0               # recording time
dtS = dtR = 1f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples

# source locations -- half at the left hand side of the model, half on top
xsrc = convertToCell(range(d[1],stop=d[1],length=nsrc))
ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
zsrc = convertToCell(range(d[2],stop=(n[2]-1)*d[2],length=nsrc))

# receiver locations -- half at the right hand side of the model, half on top
xrec = range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=nrec)
yrec = 0f0
zrec = range(d[2],stop=(n[2]-1)*d[2],length=nrec)

# set up src/rec geometry
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.05f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# set up simulation operators
Ftrue = [judiModeling(model[i], srcGeometry, recGeometry) for i = 1:nv] # acoustic wave equation solver

rand_ns = [jitter(nsrc, nsrc) for i = 1:nv]                             # select random source idx for each vintage
q_sub = [q[rand_ns[i]] for i = 1:nv]                                        # set-up source
F_sub = [Ftrue[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator

### wave physics
function F(v::Vector{Matrix{Float32}})
    m = [vec(1f3./v[i]).^2f0 for i = 1:nv]
    return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
end

function fwd(x)
    c = S(x); v = R(c); v_up = u(v); dpred = F(v_up);
    return dpred
end
@time d_obs = add_noise(F(u(R(sw_true))), snr)
sigma = 0.035657972f0

#####  physics correction/calibration

rand_ns = [jitter(nsrc, nssample) for i = 1:nv]                             # select random source idx for each vintage
q_sub = [q[rand_ns[i]] for i = 1:nv]                                        # set-up source
F_sub = [Ftrue[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator
dobs_g = [d_obs[i][rand_ns[i]] for i = 1:nv]        # subsample dobs to calculate gradient

### wave physics
function F(v::Vector{Matrix{Float32}})
    m = [vec(1f3./v[i]).^2f0 for i = 1:nv]
    return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
end
    
function fwd(x)
    c = S(x); v = R(c); v_up = u(v); dpred = F(v_up);
    return dpred
end

function f(x)
    global fval = .5f0 * norm(fwd(x)-dobs_g)^2f0
    return fval
end

@time g = gradient(()->f(x_init), Flux.params(x_init)).grads[x_init]

Zy_fixed_g = G.forward(randn(Float32, n1[1], n1[2], 1, 1), reshape(g, n1[1], n1[2], 1, 1))[2];
G1 = ReverseConditionNet(G, Zy_fixed_g);

#### start correction
niterations = 400          # 50 data pass

### Define result directory
sim_name = "UQ"
exp_name = "correction"
plot_path = plotsdir(sim_name, exp_name)
save_path = datadir(sim_name, exp_name)

### track iterations
hisloss = zeros(Float32, niterations+1)
hismisfit = zeros(Float32, niterations+1)
hisprior = zeros(Float32, niterations+1)
hislogdet = zeros(Float32, niterations+1)
prog = Progress(niterations)

## weighting
λ = 1f0;

# Define network mapping its latent space to the pretrained model's latent space.
n1 = (64, 64)
b = zeros(Float32, prod(n1))
s = ones(Float32, prod(n1))
C = vcat(s,b) #correction Layer
θ = Flux.params(C)

# ADAM-W algorithm
lr = 1f-3
lr_step = 2
num_batches = cld(nsrc, nssample)
opt = Flux.Optimiser(
        Flux.ExpDecay(lr, 0.9f0, num_batches * lr_step, 1.0f-6),
        Flux.ADAM(lr),
    )

for iter=1:niterations
    rand_ns = [jitter(nsrc, nssample) for i = 1:nv]                             # select random source idx for each vintage
    q_sub = [q[rand_ns[i]] for i = 1:nv]                                        # set-up source
    F_sub = [Ftrue[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator
    dobs = [d_obs[i][rand_ns[i]] for i = 1:nv]                                  # subsampled seismic dataset from the selected sources

    ### wave physics
    function F(v::Vector{Matrix{Float32}})
        m = [vec(1f3./v[i]).^2f0 for i = 1:nv]
        return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
    end

    # function value
    function f(z)
        x = 120f0 * G1(z)[:,:,1,1]; c = S(x); v = R(c); v_up = u(v); dpred = F(v_up);
        global misfit = .5f0/sigma^2f0 * norm(dpred-dobs)^2f0/length(vcat(dobs...))
        #global prior = .5f0 * norm(z)^2f0/length(z)
        #global logdet = -log(abs(prod(C[1])))/length(C[1])
        global prior = 0
        global logdet = 0
        global fval = misfit + prior + logdet
        @show misfit, prior, logdet, fval
        return fval
    end

    ## AD by Flux
    @time grads = gradient(()->f(reshape(C[1:prod(n1)].*randn(Float32, prod(n1)).+C[prod(n1)+1:end], n1)), θ)
    #@time grads = gradient(()->mean([f(C[1:prod(n1)].*z[i].+C[prod(n1)+1:end])/n_particles for i = 1:n_particles]), θ)
    for p in θ
        Flux.Optimise.update!(opt, p, grads[p])
    end

    hisloss[iter] = fval
    hismisfit[iter] = misfit
    hisprior[iter] = prior
    hislogdet[iter] = logdet

    ProgressMeter.next!(prog; showvalues = [(:loss, fval), (:misfit, misfit), (:prior, prior), (:iter, iter), (:stepsize, step)])

    ### save intermediate results
    save_dict = @strdict iter snr nssample λ rand_ns step niterations nv nsrc nrec survey_indices hisloss hismisfit hisprior lr lr_step
    @tagsave(
        joinpath(save_path, savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict iter snr nssample niterations nv nsrc nrec survey_indices lr lr_step

    generative_samples = [120f0 * G1(randn(Float32, n1))[:,:,1,1] for k = 1:100]
    post_mean = mean(generative_samples)
    post_std = std(generative_samples)
    ssim_i = round(assess_ssim(post_mean, x_true),digits=2)
    mse_i = round(norm(x_true'-mean(generative_samples)')^2, digits=2)
    fig = figure(figsize=(20, 12)); 
    subplot(2,4,1); imshow(120f0 * G1(reshape(C[1:prod(n1)].*randn(Float32, prod(n1)).+C[prod(n1)+1:end], n1))[:,:,1,1]', interpolation="none", vmin=20, vmax=120)
    axis("off");  colorbar(fraction=0.046, pad=0.04);title("generative samples")
    subplot(2,4,2); imshow(120f0 * G1(reshape(C[1:prod(n1)].*randn(Float32, prod(n1)).+C[prod(n1)+1:end], n1))[:,:,1,1]', interpolation="none", vmin=20, vmax=120)
    axis("off");  colorbar(fraction=0.046, pad=0.04);title("generative samples")
    subplot(2,4,3); imshow(120f0 * G1(reshape(C[1:prod(n1)].*randn(Float32, prod(n1)).+C[prod(n1)+1:end], n1))[:,:,1,1]', interpolation="none", vmin=20, vmax=120)
    axis("off");  colorbar(fraction=0.046, pad=0.04);title("generative samples")
    subplot(2,4,4); imshow(120f0 * G1(reshape(C[1:prod(n1)].*randn(Float32, prod(n1)).+C[prod(n1)+1:end], n1))[:,:,1,1]', interpolation="none", vmin=20, vmax=120)
    axis("off");  colorbar(fraction=0.046, pad=0.04);title("generative samples")
    subplot(2,4,5); imshow(x_true', interpolation="none", vmin=20, vmax=120);
    axis("off");  colorbar(fraction=0.046, pad=0.04);title("ground truth")
    subplot(2,4,6); imshow(post_mean', interpolation="none", vmin=20, vmax=120)
    axis("off");  colorbar(fraction=0.046, pad=0.04);title("posterior mean")
    subplot(2,4,7); imshow(x_true'-mean(generative_samples)', interpolation="none", cmap="magma")
    axis("off");title("Plot: Absolute error | MSE="*string(mse_i)) ; cb = colorbar(fraction=0.046, pad=0.04)
    subplot(2,4,8); imshow(post_std', interpolation="none", cmap="magma")
    axis("off");  colorbar(fraction=0.046, pad=0.04);title("posterior standard deviation")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_cnf_correction.png"), fig); close(fig)
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    subplot(4,1,1);
    plot(hisloss[1:iter]);title("loss=$(hisloss[iter])");
    subplot(4,1,2);
    plot(hismisfit[1:iter]);title("misfit=$(hismisfit[iter])");
    subplot(4,1,3);
    plot(hisprior[1:iter]);title("prior=$(hisprior[iter])");
    subplot(4,1,4);
    plot(hislogdet[1:iter]);title("logdet=$(hislogdet[iter])");
    suptitle("Loss at iter $iter, seismic data snr=$snr")
    tight_layout()
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

end