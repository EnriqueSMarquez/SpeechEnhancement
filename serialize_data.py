import data

data_folder = '/ssd/esm1g14/segan'
# data.downsample_16k(clean_folder_name='clean_testset_wav',noisy_folder_name='noisy_testset_wav',out_clean_fdrnm='test_clean_16k',out_noisy_fdrnm='test_noisy_16k')
data.process_and_serialize(clean_folder_16k='test_clean_16k',noisy_folder_16k='test_noisy_16k',output_folder='ser_test_data')
