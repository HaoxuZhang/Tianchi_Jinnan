### unzip the zip files
unzip jinnan2_round1_test_a_20190306.zip
unzip jinnan2_round1_test_b_20190326.zip
unzip jinnan2_round1_train_20190305.zip

### split datasets to training set and valid set
mkdir annotations
source activate jinnan
python split_datasets.py

### do data augmentation by overlaying the restricted objects from restricted images onto some normal image randomly
mkdir normal_aug
python select_normal_images.py
python data_augmentation.py

### merge the enhanced images and restricted images into one file
mkdir train_val
python merge.py
rm -r normal_aug

### convert the test datasets to json format, which will be used in inference processing
python test_json.py --file_path './jinnan2_round1_test_a_20190306' --json_path './annotations/test_a.json'
python test_json.py --file_path './jinnan2_round1_test_b_20190326' --json_path './annotations/test_b.json'