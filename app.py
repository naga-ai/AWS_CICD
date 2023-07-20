from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion


application=Flask(__name__)

app=application

## Route for a home page

# @app.route('/')
# def index():
#     return render_template('index.html') 

@app.route('/')
def home():
    return render_template('model.html') # change this line to reflect the new file name


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score')))
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

@app.route('/create_model', methods=['POST'])
def create_model():
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    preprocess = DataTransformation()

    train_arr, test_arr, preprocessor_obj_file_path = preprocess.initiate_data_transformation(train_data_path, test_data_path)
    print("Data transformation completed, preprocessor is created")

    model = ModelTrainer()
    r2_square = model.initiate_model_trainer(train_arr, test_arr)

    print("Model created")
    print("r2_square ", r2_square)

    return "Model Created Successfully"
  

if __name__=="__main__":
    # app.run(host="0.0.0.0",port=8080)        
    app.run(host='0.0.0.0', port=8080)        


