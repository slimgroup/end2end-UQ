#CUDA_VISIBLE_DEVICES=7 nohup julia scripts/train_cond_nf.jl & 
using DrWatson
@quickactivate "End-to-end-hint3"
import Pkg; Pkg.instantiate()

#using Pkg;Pkg.add(PackageSpec(url="https://github.com/slimgroup/InvertibleNetworks.jl", rev="joint-training"))
using PyPlot
using InvertibleNetworks, Flux
using LinearAlgebra
using Random
using JLD2,BSON, MAT
using Statistics
using ImageQualityIndexes 
using UNet

function posterior_sampler(G, y, size_x; device=gpu, num_samples=1, batch_size=16)
    # make samples from posterior for train sample 
	X_forward = randn(Float32, size_x[1:end-1]...,batch_size) |> device
    Y_train_latent_repeat = repeat(y |>cpu, 1, 1, 1, batch_size) |> device
    _, Zy_fixed_train, _ = G.forward(X_forward, Y_train_latent_repeat); #needs to set the proper sizes here

    X_post_train = zeros(Float32, size_x[1:end-1]...,num_samples)
    for i in 1:div(num_samples, batch_size)
    	ZX_noise_i = randn(Float32, size_x[1:end-1]...,batch_size)|> device
   		X_post_train[:,:,:, (i-1)*batch_size+1 : i*batch_size] = G.inverse(
        	ZX_noise_i,
        	Zy_fixed_train
    		) |> cpu;
	end
	X_post_train
end

function get_cm_l2_ssim(G, X_batch, Y_batch; device=gpu, num_samples=1)
	    num_test = size(Y_batch)[end]
	    l2_total = 0 
	    ssim_total = 0 
	    #get cm for each element in batch
	    for i in 1:num_test
	    	y_i = Y_batch[:,:,:,i:i]
	    	x_i = X_batch[:,:,:,i:i]
	    	X_post_test = posterior_sampler(G, y_i, size(x_i); device=device, num_samples=num_samples, batch_size=batch_size)
	    	X_post_mean_test = mean(X_post_test;dims=4)
	    	ssim_total += assess_ssim(X_post_mean_test[:,:,1,1], x_i[:,:,1,1]|> cpu)
			l2_total   += norm(X_post_mean_test[:,:,1,1]- (x_i[:,:,1,1]|> cpu))^2
		end
	return l2_total / num_test, ssim_total / num_test
end

function get_loss(G, X_batch, Y_batch; device=gpu, batch_size=16)
	num_test = size(Y_batch)[end]
	l2_total = 0 
	logdet_total = 0 
	num_batches = div(num_test, batch_size)
	for i in 1:num_batches
		x_i = X_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size] 
    	y_i = Y_batch[:,:,:,(i-1)*batch_size+1 : i*batch_size] 

    	x_i .+= noise_lev_x*randn(Float32, size(x_i)); 
		y_i .+= noise_lev_y*randn(Float32, size(y_i)); 

    	Zx, Zy, lgdet = G.forward(x_i|> device, y_i|> device) |> cpu;
    	l2_total     += norm(Zx)^2 / (N*batch_size)
		logdet_total += lgdet / N
	end

	return l2_total / (num_batches), logdet_total / (num_batches)
end

plot_path = plotsdir("cond-nf")

# Define raw data directory
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")
grad_path = datadir("training-data", "nrec=960_nsample=1100_nsrc=32_nssample=4_ntrain=1000_nv=5_nvalid=100_snr=10.0_upsample=2.jld2")

perm = matread(perm_path)["perm"];
conc = matread(conc_path)["conc"];
grad = JLD2.load(grad_path)["gset"];

n_train = 1000
n_val   = 50
nx = 64
N = nx^2

#Normalize X!!! I think that the output needs to be 01 because of a sigmoid at end of unet.
#Maybe you dont need to normalize if you do the conditional network, since the putput of the unet still goes through layers 
max_val_x = maximum(perm)
X_train = reshape(perm[:,:,1:n_train], size(perm)[1:2]...,1,n_train) ./ max_val_x
Y_train = reshape(grad[:,:,1:n_train], size(grad)[1:2]...,1,n_train)

X_val = reshape(perm[:,:,(n_train+1):(n_train+n_val)], size(perm)[1:2]...,1,n_val) ./ max_val_x
Y_val = reshape(grad[:,:,(n_train+1):(n_train+n_val)], size(grad)[1:2]...,1,n_val)

vmax = maximum(X_train)
vmin = minimum(X_train)

# Training hyperparameters 
device = gpu
lr = 4f-3
lr_step   = 30
lr_rate = 0.75f0
clipnorm_val = 2.5f0
noise_lev_x  = 0.005f0
noise_lev_y  = 0.005f0
batch_size   = 20
n_batches    = cld(n_train, batch_size)
n_epochs     = 500

save_every   = 5
plot_every   = 1
sample_viz = 1

# Number of samples to test conditional mean quality metric on.
n_condmean = 25
posterior_samples = 128

# Create summary network
sum_net = true
h1 = nothing
n_in = 1
unet_levels = 3
if sum_net
	h1 = Unet(1, 1, unet_levels);
	trainmode!(h1, true)
	h1 = FluxBlock(Chain(h1))
	h1 = h1 |> device
end

# Create conditional network
K = 6
L = 6
n_hidden = 32
low = 0.5f0

Random.seed!(123);
G = NetworkConditionalGlow(1, n_in, n_hidden,  L, K; rb_activation=ReLUlayer(), summary_net=h1, split_scales=true, activation=SigmoidLayer(low=low,high=1.0f0));
G = G |> device;

# Optimizer
opt = Flux.Optimiser(ExpDecay(lr, lr_rate, n_batches*lr_step, 1f-6), ClipNorm(clipnorm_val), ADAM(lr))

# Training logs 
loss   = [];
logdet = [];
ssim   = [];
l2_cm  = [];

loss_val   = [];
logdet_val = [];
ssim_val   = [];
l2_cm_val  = [];

for e=1:n_epochs
	idx_e = reshape(randperm(n_train), batch_size, n_batches) 
    for b = 1:n_batches 
    	@time begin
	        X = X_train[:, :, :, idx_e[:,b]];
	        Y = Y_train[:, :, :, idx_e[:,b]];
	        X .+= noise_lev_x*randn(Float32, size(X));
			Y .+= noise_lev_y*randn(Float32, size(Y));
      
	        # Forward pass of normalizing flow
	        Zx, Zy, lgdet = G.forward(X|> device, Y|> device)

	        # Loss function is l2 norm 
	        append!(loss, norm(Zx)^2 / (N*batch_size))  # normalize by image size and batch size
	        append!(logdet, -lgdet / N) # logdet is internally normalized by batch size

	        # Set gradients of flow and summary network
	        G.backward(Zx / batch_size, Zx, Zy; C_save=Y|> device)

	        for p in get_params(G) 
	          Flux.update!(opt,p.data,p.grad)
	        end
	        clear_grad!(G)

	        print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
	            "; f l2 = ",  loss[end], 
	            "; lgdet = ", logdet[end], "; f = ", loss[end] + logdet[end], "\n")

	        Base.flush(Base.stdout)
    	end
    end

    # get objective mean metrics over testing batch  
    @time l2_val, lgdet_val  = get_loss(G, X_val, Y_val; device=device, batch_size=batch_size)
    append!(logdet_val, -lgdet_val)
    append!(loss_val, l2_val)

    # get conditional mean metrics over training batch (takes a bit since you have to do posterior sample but worth it in order to catch overfitting)
    @time cm_l2_train, cm_ssim_train = get_cm_l2_ssim(G, X_train[:,:,:,1:n_condmean], Y_train[:,:,:,1:n_condmean]; device=device, num_samples=posterior_samples )
    append!(ssim, cm_ssim_train)
    append!(l2_cm, cm_l2_train)

    # get conditional mean metrics over testing batch (takes a bit since you have to do posterior sample but worth it in order to catch overfitting)
    @time cm_l2_val, cm_ssim_val  = get_cm_l2_ssim(G, X_val[:,:,:,1:n_condmean], Y_val[:,:,:,1:n_condmean]; device=device, num_samples=posterior_samples )
    append!(ssim_val, cm_ssim_val)
    append!(l2_cm_val, cm_l2_val)

    if(mod(e,plot_every)==0) 
	    #testmode!(h1, true)
	    
	    x      = X_val[:,:,:,sample_viz:sample_viz];
	    y      = Y_val[:,:,:,sample_viz:sample_viz];

	    # make samples from posterior for train sample 
	   	X_post = posterior_sampler(G,  y, size(x); device=device, num_samples=posterior_samples)
	   	X_post = X_post |> cpu
	    X_post_mean = mean(X_post,dims=4)
	    X_post_std  = std(X_post, dims=4)
	    error_mean = abs.(X_post_mean[:,:,1,1]-x[:,:,1,1])
	    ssim_i = round(assess_ssim(X_post_mean[:,:,1,1], x[:,:,1,1]),digits=2)
	    mse_i = round(norm(error_mean)^2,digits=2)

	    fig = figure(figsize=(20, 10)); 
	    subplot(2,4,1); imshow(y[:,:,1,1]', interpolation="none", cmap="gray")
		axis("off");  colorbar(fraction=0.046, pad=0.04);title(L"$\hat \mathbf{y}$  (gradient)")

	    subplot(2,4,2); imshow(X_post[:,:,1,1]',vmax=vmax,vmin=vmin, interpolation="none", cmap="gray")
		axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample 1")

		subplot(2,4,3); imshow(X_post[:,:,1,2]', vmax=vmax,vmin=vmin,interpolation="none", cmap="gray")
		axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample 2")

		subplot(2,4,4); imshow(X_post[:,:,1,3]',vmax=vmax,vmin=vmin, interpolation="none", cmap="gray")
		axis("off");  colorbar(fraction=0.046, pad=0.04);title("Posterior sample 3")

		subplot(2,4,5); imshow(x[:,:,1,1]', vmax=vmax,vmin=vmin, interpolation="none", cmap="gray")
		axis("off"); title(L"$\mathbf{x_{gt}}$") ; colorbar(fraction=0.046, pad=0.04)

		subplot(2,4,6); imshow(X_post_mean[:,:,1,1]', vmax=vmax,vmin=vmin,  interpolation="none", cmap="gray")
		axis("off"); title("Posterior mean SSIM="*string(ssim_i)) ; colorbar(fraction=0.046, pad=0.04)

		subplot(2,4,7); imshow(error_mean', interpolation="none", cmap="magma")
		axis("off");title("Plot: Absolute error | MSE="*string(mse_i)) ; cb = colorbar(fraction=0.046, pad=0.04)

		subplot(2,4,8); imshow(X_post_std[:,:,1,1]',interpolation="none", cmap="magma")
		axis("off"); title("Posterior standard deviation") ;cb =colorbar(fraction=0.046, pad=0.04)

		tight_layout()
		fig_name = @strdict sum_net unet_levels posterior_samples clipnorm_val noise_lev_x noise_lev_y n_train e lr lr_step lr_rate n_hidden L K batch_size lr_step 
		safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_nf_sol_val.png"), fig); close(fig)
			
	    ############# Training metric logs
		sum = loss + logdet
		sum_val = loss_val + logdet_val

		fig = figure(figsize=(10,12))
		subplot(5,1,1); title("L2 Term: Train="*string(loss[end])*" Validation="*string(loss_val[end]))
		plot(loss, label="Train");
		plot(n_batches:n_batches:n_batches*e, loss_val, label="Validation"); 
		axhline(y=1,color="red",linestyle="--",label="Normal noise")
		ylim(top=1.5)
		ylim(bottom=0)
		xlabel("Parameter Update"); legend();

		subplot(5,1,2); title("Logdet Term: Train="*string(logdet[end])*" Validation="*string(logdet_val[end]))
		plot(logdet);
		plot(n_batches:n_batches:n_batches*e, logdet_val);
		xlabel("Parameter Update") ;

		subplot(5,1,3); title("Total Objective: Train="*string(sum[end])*" Validation="*string(sum_val[end]))
		plot(sum); 
		plot(n_batches:n_batches:n_batches*e, sum_val); 
		xlabel("Parameter Update") ;

		subplot(5,1,4); title("Posterior mean SSIM: Train=$(ssim[end]) Validation=$(ssim_val[end])")
	    plot(1:n_batches:n_batches*e, ssim); 
	    plot(1:n_batches:n_batches*e, ssim_val); 
	    xlabel("Parameter Update") 

	    subplot(5,1,5); title("Posterior mean MSE: Train=$(l2_cm[end]) Validation=$(l2_cm_val[end])")
	    plot(1:n_batches:n_batches*e, l2_cm); 
	    plot(1:n_batches:n_batches*e, l2_cm_val); 
	    xlabel("Parameter Update") 

		tight_layout()
		fig_name = @strdict sum_net unet_levels posterior_samples clipnorm_val noise_lev_x noise_lev_y n_train e lr lr_step lr_rate n_hidden L K batch_size lr_step
		safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_trainin_log.png"), fig); close(fig)
	end

	#save params every 4 epochs
    if(mod(e,save_every)==0) 
         # Saving parameters and logs
     	unet_model = G.summary_net.model;
        unet_model = unet_model |> cpu;
        G_save = deepcopy(G);
        reset!(G_save.summary_net); # clear params to not save twice

		Params = get_params(G_save) |> cpu;
		save_dict = @strdict unet_model unet_levels n_in sum_net clipnorm_val n_train e noise_lev_x noise_lev_y lr lr_step lr_rate n_hidden L K Params loss logdet l2_cm ssim loss_val logdet_val l2_cm_val ssim_val batch_size;
	
		@tagsave(
			joinpath(datadir(), savename(save_dict, "bson"; digits=6)),
			save_dict;
			safe=true
		);
    end

  #   if(mod(e,save_every)==0) 
  #        # Saving parameters and logs
		# Params = get_params(G) |> cpu
		# save_dict = @strdict n_in sum_net clipnorm_val n_train e noise_lev_x lr n_hidden L K Params loss logdet l2_cm ssim loss_val logdet_val l2_cm_val ssim_val batch_size lr_step
		# @tagsave(
		# 	joinpath(datadir(), savename(save_dict, "jld2"; digits=6)),
		# 	save_dict;
		# 	safe=true
		# )
  #   end
end

