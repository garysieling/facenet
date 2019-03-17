set -e

python src/align_dataset_mtcnn.py \
./kids_tagged_and_video/ \
./kids_tagged_and_video_mtcnnpy_182 \
--image_size  182 \
--random_order \
--gpu_memory_fraction 0.25 \
--margin 44

python src/classifier.py TRAIN ./images/kids_tagged_and_video_mtcnnpy_182/ ./models/20180408-102900/20180408-102900.pb ./models/kids_classifier.pkl --batch_size 1000
# --image_size 224 \
#--random_flip \

python src/train_softmax.py \
 --data_dir ./kids_tagged_and_video/ \
 --optimizer ADAM \
 --learning_rate -1 \
 --max_nrof_epochs 150 \
--keep_probability 0.8 \
--random_crop \
--use_fixed_image_standardization \
--learning_rate_schedule_file data/learning_rate_schedule_classifier_casia.txt \
--weight_decay 5e-4 \
--embedding_size 512 \
--lfw_distance_metric 1 \
--lfw_use_flipped_images \
--lfw_subtract_mean \
--validation_set_split_ratio 0.05 \
--validate_every_n_epochs 5 \
--prelogits_norm_loss_factor 5e-4

# --models_base_dir /data/models/20180408-102900/20180408-102900.pb 
#/data/models/kids_classifier.pkl --batch_size 1000 



python src/classifier.py CLASSIFY ./kids_tagged_and_video/ ./models/20180408-102900/20180408-102900.pb ./models/kids_classifier.pkl --batch_size 1000
