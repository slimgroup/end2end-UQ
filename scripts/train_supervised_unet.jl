using DrWatson
@quickactivate "End-to-end-hint3"

using MAT, JLD2
using Flux, UNet, Zygote
using Random, LinearAlgebra
using PyPlot, ImageQualityIndexes

plot_path = plotsdir("supervised")

# Define raw data directory
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")
grad_path = datadir("training-data", "nrec=960_nsample=1100_nsrc=32_nssample=4_ntrain=1000_nv=5_nvalid=100_snr=10.0_upsample=2.jld2")

perm = matread(perm_path)["perm"];
conc = matread(conc_path)["conc"];
grad = JLD2.load(grad_path)["gset"];

n_train = 1000
n_val   = 50

#Normalize X!!! I think that the output needs to be 01 because of a sigmoid at end of unet. 
max_val_x = maximum(perm)
X_train = reshape(perm[:,:,1:n_train], size(perm)[1:2]...,1,n_train) ./ max_val_x
Y_train = reshape(grad[:,:,1:n_train], size(grad)[1:2]...,1,n_train)

X_val = reshape(perm[:,:,(n_train+1):(n_train+n_val)], size(perm)[1:2]...,1,n_val) ./ max_val_x
Y_val = reshape(grad[:,:,(n_train+1):(n_train+n_val)], size(grad)[1:2]...,1,n_val)

# Training parameters
device = gpu
unet_levels = 3
n_epochs = 40
batch_size = 10
lr = 1f-3
y_noise = 0.005f0 

save_every = n_epochs
plot_every = 4
sample_viz = 1

# Make UNET. NOTE display doesnt work so always use semicolon
unet = Unet(1, 1, unet_levels);
unet = unet |> device;
ps = Flux.params(unet);

function loss_unet(unet, X, Y;)
    ΔX = unet(Y) - X 
    norm(ΔX)^2 / prod(size(ΔX))
end

# Setup training loop
n_batches     = cld(n_train, batch_size)
n_batches_val = cld(n_val, batch_size)

# Optimizer
opt = ADAM(lr)

# Training logs 
loss     = [];
loss_val = [];

# Training loop
trainmode!(unet, true);
for e=1:n_epochs # epoch loop
    @time begin
    idx_e = reshape(randperm(n_train), batch_size, n_batches) 
    for b = 1:n_batches # batch loop
        X = X_train[:, :, :, idx_e[:,b]];
        Y = Y_train[:, :, :, idx_e[:,b]];

        # Additive noise for data augmentation
        Y += y_noise*randn(Float32,size(Y))

        #f = loss_unet(unet, X|> device, Y|> device)
        f, back = Zygote.pullback(() -> loss_unet(unet, X|> device, Y|> device;), ps)
        append!(loss, f) 
        gs = back(one(f))
        Flux.update!(opt, ps, gs)

        print("Iter: epoch=", e, "/", n_epochs, ", batch=", b, "/", n_batches, 
            "; f l2 = ",  loss[end], "\n")
        Base.flush(Base.stdout)
        
    end
    end

    idx_e_val = reshape(1:n_val, batch_size, n_batches_val) 
    f_val_total = 0
    for b = 1:n_batches_val # batch loop
        X = X_val[:, :, :, idx_e_val[:,b]];
        Y = Y_val[:, :, :, idx_e_val[:,b]];

        f_val_total += loss_unet(unet, X |> device, Y |> device)
    end
    append!(loss_val, f_val_total / n_batches_val)

    if mod(e, plot_every) == 0
        X  = X_val[:,:,:,sample_viz]
        Y  = Y_val[:,:,:,sample_viz:sample_viz]

        # Plot current performance
        X_hat = unet(Y |> device)[:,:,1,1] |> cpu

        ssim_test = round(assess_ssim(X_hat, X) ,digits=2)
        psnr_test = round(assess_psnr(X_hat, X) ,digits=2)

        # Test 
        fig = figure(figsize=(10,6))
        subplot(1,3,1); title("Validation y");
        imshow(Y[:,:,1,1]', cmap="gray", interpolation="none",)
        cb = colorbar(fraction=0.046, pad=0.04);

        subplot(1,3,3); title("Validation x");
        imshow(X[:,:,1,1]', cmap="gray", interpolation="none",)
        cb = colorbar(fraction=0.046, pad=0.04);

        subplot(1,3,2); title("Validation h(y) \n PSNR=$(psnr_test) SSIM=$(ssim_test)");
        imshow(X_hat[:,:,1,1]',cmap="gray", interpolation="none")
        cb = colorbar(fraction=0.046, pad=0.04);

        tight_layout()
        fig_name = @strdict n_train unet_levels batch_size lr e y_noise
        safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_sol_val.png"), fig); close(fig)
        
        ############# Training metric logs
        fig = figure(); title("Training objective: train=$(loss[end]) val=$(loss_val[end])")
        plot(loss; label="Training MSE")
        plot(n_batches:n_batches:n_batches*e, loss_val; label="Validation MSE")
        xlabel("Parameter Update"); legend(); tight_layout()

        fig_name = @strdict n_train unet_levels batch_size lr  e y_noise
        safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_log.png"), fig); close(fig)
    end

    #NOTE SAVING IS A LITTLE IFFY. THIS WORKED FOR ME BUT MAKE SURE IT WORKS BEFORE YOU DO ANYTHIGN IMPORTANT
    # if(mod(e,save_every)==0) 
    #     global unet = unet |> cpu
    #     save_dict = @strdict   n_train e n_epochs lr loss loss_val batch_size  
    #     safesave(joinpath(datadir(), savename(save_dict, "bson"; digits=6)),save_dict;)
    #     global unet = unet |> gpu
    # end
end
