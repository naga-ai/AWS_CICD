from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion

obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

# train_path = "artifacts/train.csv"
# test_path = "artifacts/test.csv"

preprocess = DataTransformation()

train_arr, test_arr,preprocessor_obj_file_path = preprocess.initiate_data_transformation(train_data_path, test_data_path)
print("data transformation completed, preporcessore is created")

model = ModelTrainer()
r2_square = model.initiate_model_trainer(train_arr, test_arr)

print("model created")
print("r2_square ", r2_square)