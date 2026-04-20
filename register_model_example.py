import os
import mlflow


EXPERIMENT_NAME = 'churn_vds'
RUN_NAME = "model_0_registry"
REGISTRY_MODEL_NAME = "churn_model_vds"


os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net" #endpoint бакета от YandexCloud
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") # получаем id ключа бакета, к которому подключён MLFlow, из .env
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") # получаем ключ бакета, к которому подключён MLFlow, из .env

# ваш код здесь


pip_requirements = '../requirements.txt'
signature = mlflow.models.infer_signature(X_test, prediction)
input_example = X_test[:10]
metadata = {'model_type': 'monthly'}


experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    # ваш код здесь
    model_info = mlflow.catboost.log_model( 
            cb_model=model,
            artifact_path='models',
            registered_model_name=REGISTRY_MODEL_NAME,
            pip_requirements=pip_requirements,
            signature=signature,
            input_example=input_example,
            metadata=metadata,
            await_registration_for=60
		)

loaded_model = mlflow.catboost.load_model(model_uri=model_info.model_uri)
model_predictions = loaded_model.predict(X_test)

assert model_predictions.dtype == int

print(model_predictions[:10])