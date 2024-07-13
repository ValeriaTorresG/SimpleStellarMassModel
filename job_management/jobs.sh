#? Write data
# python write_data.py --fits_path <fits_file_path> --bgs_path <probabgs_file_path>
python job_management/write_data.py --fits_path '../data/BGS_ANY_N_clustering.dat.fits' --bgs_path '../data/BGS_ANY_full.provabgs.sv3.v0.hdf5'

#? Run linear model
#python ../src/linear/main.py --train_rosettes <rosettes for training>
        #--test_rosettes <(optional)rosettes to test>
        #--test_size <(optional, default=0.8) test size to evaluate>
        #--plot <(optional) sets true for plotting predictions>
python src/linear/main.py --train_rosettes '3,6,7' --test_rosettes '13' --test_size 0.8 --plot

#? Run Random Forest model
#python ../src/random_forest/main.py --train_rosettes <rosettes for training>
        #--test_rosettes <(optional)rosettes to test>
        #--change_data <(optional) columns to take out of the model>
        #--plot <(optional) sets true for plotting predictions>
        #--shap <(optional) sets true for plotting predictions>
        #--optimize <(optional) sets true for optimizing hyperparameters>
        #--add_rand_col <(optional) sets true for adding a random column in the model>
        #--feat_imp <(optional) sets true for plotting feature importances>
python src/random_forest/main.py --train_rosettes '12,15,13' --test_rosettes '19,3,7,18' --plot --shap --optimize --change_data 'z' --add_rand_col --feat_imp