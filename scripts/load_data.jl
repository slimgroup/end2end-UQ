using DrWatson
@quickactivate "End-to-end-hint3"
using MAT, JLD2

# Define raw data directory
mkpath(datadir("training-data"))
perm_path = datadir("training-data", "perm_gridspacing15.0.mat")
conc_path = datadir("training-data", "conc_gridspacing15.0.mat")
grad_path = datadir("training-data", "nrec=960_nsample=1100_nsrc=32_nssample=4_ntrain=1000_nv=5_nvalid=100_snr=10.0_upsample=2.jld2")

# Download the dataset into the data directory if it does not exist
if ~isfile(perm_path)
    run(`wget https://www.dropbox.com/s/o35wvnlnkca9r8k/'
        'perm_gridspacing15.0.mat -q -O $perm_path`)
end
if ~isfile(conc_path)
    run(`wget https://www.dropbox.com/s/mzi0xgr0z3l553a/'
        'conc_gridspacing15.0.mat -q -O $conc_path`)
end
if ~isfile(conc_path)
    run(`wget https://www.dropbox.com/s/ckb6o0ywcfaamzg/'
        'nrec=960_nsample=1100_nsrc=32_nssample=4_ntrain=1000_nv=5_nvalid=100_snr=10.0_upsample=2.jld2 -q -O $grad_path`)
end

perm = matread(perm_path)["perm"];
conc = matread(conc_path)["conc"];
grad = JLD2.load(grad_path)["gset"];