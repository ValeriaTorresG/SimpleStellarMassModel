#? Write data
# python write_data.py --fits_path <fits_file_path> --bgs_path <probabgs_file_path>
python ../src/write_data.py --fits_path '../data/BGS_ANY_N_clustering.dat.fits' --bgs_path '../data/BGS_ANY_full.provabgs.sv3.v0.hdf5'

#? Run linear model
#python ../src/linear/main.py --train_rosettes <rosettes for training> --test_rosettes <(optional)rosettes to test> --test_size <(optional, default=0.8) test size to evaluate> --plot
#use --plot as a store_true value to plot
python ../src/linear/main.py --train_rosettes '3,6,7' --test_rosettes '13' --test_size 0.8 #--plot

#? Run Random Forest model
python main.py --train_rosettes '12' --test_rosettes '19' --plot --shap --optimize --change_data 'z' --add_rand_col --feat_imp