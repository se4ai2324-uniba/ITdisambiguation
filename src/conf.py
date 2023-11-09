""" Module containing the configuration """

config = {
	"TRAIN_IMAGES_PATH" : "data/train_preprocessed_images/",
	"TRAIN_DATA" : "data/Train/resized_train.data.v1.txt",
	"TRAIN_TARGET_IMAGES" : "data/Train/resized_train.gold.v1.txt",
	"TRAIN_ORIGINAL_IMAGE_PATH" : "data/Train/resized_images_N/",
	"TEST_IMAGES_PATH" : "data/test_preprocessed_images/",
	"TEST_DATA" : "data/Test/en.test.data.v1.1.txt",
	"TEST_TARGET_IMAGES" : "data/Test/en.test.gold.v1.1.txt",
	"TEST_ORIGINAL_IMAGE_PATH" : "data/Test/resized_test_images_N/",
	"MODEL_FILE" : "models/model.pt",
	"METRICS_FOLDER" : "metrics/",
	"BATCH_SIZE": 8,
	"GRAD_ACC": 8,
	"EPOCHS": 2,
	"LEARNING_RATE": 2e-5
}
