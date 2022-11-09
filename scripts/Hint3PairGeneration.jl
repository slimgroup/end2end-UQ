# author: Ziyi Yin, ziyi.yin@gatech.edu 
## Date: Oct, 2022
## this script produces training pairs for HINT3 -- with (K, grad K)
using DrWatson
@quickactivate "FNO4CO2"

using FNO4CO2
using Flux, Random
using MAT, Statistics, LinearAlgebra
using JLD2
using JUDI
using JSON
using PyPlot
using FNO4CO2
using JUDI, Flux, Distributed, LinearAlgebra, Statistics, FFTW, JLD2
JLD2.@load "../data/3D_FNO/batch_size=2_dt=0.02_ep=300_epochs=1000_learning_rate=0.0001_modes=4_nt=51_ntrain=1000_nvalid=100_s=1_width=20.jld2";

Random.seed!(2022)
NN = deepcopy(NN_save);
Flux.testmode!(NN, true);

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")

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

ntrain = 1000
nvalid = 100

# physics grid
n1 = (64, 64)
d1 = 1f0 ./ n
nt = 51
dt = 0.02f0
grid = gen_grid(n1, d1, nt, dt)

# observation vintages
  nv = 5

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
    return d_obs
end

# where to take the gradient
x_init = mean(perm[:,:,1:ntrain], dims=3)[:,:,1]

 survey_indices = Int.(round.(range(1, stop=22, length=nv)))
 function pair_sample(i, x_true, y_true, x_init)

    survey_indices = Int.(round.(range(1, stop=22, length=nv))) # observation vintages

    sw_true = y_true[survey_indices,:,:]; # ground truth CO2 concentration at these vintages

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

    rand_ns = [jitter(nsrc, nssample) for i = 1:nv]                             # select random source idx for each vintage
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
    function f(x)
        global fval = .5f0 * norm(fwd(x)-d_obs)^2f0
        return fval
    end

    @time g = gradient(()->f(x_init), Flux.params(x_init)).grads[x_init]
   
    return g
    
end

Base.flush(stdout)

nsample = ntrain+nvalid

gset = zeros(Float32, n1[1], n1[2], nsample)
for i = 1:nsample
    Base.flush(Base.stdout)
    println("sample $i")
    gset[:,:,i] = pair_sample(i, perm[:,:,i], conc[:,:,:,i], x_init)
end

### save gradient results

save_dict = @strdict gset upsample snr nssample nv nsrc nrec survey_indices ntrain nvalid nsample
@tagsave(
    joinpath(datadir("gradients"), savename(save_dict, "jld2"; digits=6)),
    save_dict;
    safe=true
)

#=
for i = 1:nsample
    figure(figsize=(15,15));
    subplot(2,2,1);imshow(x_init', vmin=20, vmax=120);colorbar();title("initial (where we take the gradient at)");
    subplot(2,2,2);imshow(perm[:,:,i]', vmin=20, vmax=120);colorbar();title("true permeability");
    subplot(2,2,3);imshow(gset[:,:,i]');colorbar();title("gradient");
    subplot(2,2,4);imshow(x_init'-perm[:,:,i]');colorbar();title("initial-true");
    savefig("upsample$(upsample)_nv$(nv)_hint3samples$i.png", bbox_inches="tight", dpi=300)
end
=#
