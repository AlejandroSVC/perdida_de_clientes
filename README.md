# Predicción de la pérdida de clientes mediante XGBoost y PySpark en AWS

Este script permite construir y gestionar un pipeline robusto de clasificación binaria usando XGBoost en AWS SageMaker, aprovechando PySpark para el preprocesamiento distribuido. Se carga datos en formato Parquet, se les procesa con PySpark y se entrena un modelo escalable en SageMaker, usando recursos gestionados de AWS.

## Antecedentes

La investigación sobre la pérdida de clientes (customer churn) ha identificado diversas variables predictoras, entre las que se encuentran las siguientes:

1. Valor del tiempo de Vida del Cliente (VLC).

2. Disminución del uso o la Interacción con el producto o servicio.

3. Antigüedad del cliente.

4. Interacciones con el servicio de atención al cliente.

5. Detalles contractuales o de suscripción.

6. Información sociodemográfica y psicográfica.

7. Nivel de ingreso.

8. Número de quejas.

9. Historial de pagos o problemas de facturación.

10. Método de pago.

11. Nivel de satisfacción del cliente.

## 1. Requisitos Previos

- Instalar: `pip install pyspark sagemaker boto3 pandas scikit-learn`
- Tener permisos y configuración de AWS (rol, bucket S3, acceso a SageMaker).
- Archivo Parquet local con los datos de clientes y la columna objetivo `churn`.

## 2. Configuración Inicial y Carga de Librerías

En esta sección se adapta el flujo para ejecutar el entrenamiento y predicción
de XGBoost sobre AWS SageMaker, aprovechando su capacidad gestionada.
El preprocesamiento sigue usando PySpark local/distribuido, pero el modelo
se entrena y despliega usando SageMaker. Se debe tener los permisos y
configuraciones necesarios en AWS (rol, bucket S3, etc.).

python

### Importar bibliotecas

```
import os  # Para trabajar con rutas y variables de entorno
import pandas as pd  # Manipulación de datos
import boto3  # Cliente AWS
import sagemaker  # SDK de SageMaker
from sagemaker.inputs import TrainingInput  # Para definir input de entrenamiento
from sagemaker.xgboost.estimator import XGBoost  # Contenedor gestionado por SageMaker
from pyspark.sql import SparkSession  # Para iniciar Spark
from pyspark.sql.functions import col  # Manipulación de columnas
from pyspark.ml.feature import VectorAssembler, StringIndexer  # Preprocesamiento
from pyspark.ml import Pipeline  # Pipeline de Spark
```
### Configuración inicial de Spark y SageMaker.
```
spark = SparkSession.builder.appName("ChurnPredictionSageMaker").getOrCreate()  # Inicia sesión Spark
parquet_path = "./clientes_churn.parquet"  # Ruta local del Parquet

sagemaker_session = sagemaker.Session()  # Crea sesión de SageMaker
bucket = sagemaker_session.default_bucket()  # Usa el bucket por defecto de SageMaker
role = sagemaker.get_execution_role()  # Obtiene el rol asociado
```
## 3. Carga y Preprocesamiento de Datos con PySpark

Cargar el archivo Parquet y preprocesar los datos:
- Convertit la columna objetivo a numérica.
- Ensamblar las columnas de entrada en un vector de features.
- Convertir el DataFrame de Spark a Pandas, que es el formato requerido por SageMaker para entrenamiento con CSV.
```
df = spark.read.parquet(parquet_path)  # Carga el Parquet
row_count = df.count()  # Cuenta filas
threshold = 100_000  # Umbral para Spark vs Pandas

target_col = "churn"  # Columna objetivo
feature_cols = [col for col in df.columns if col != target_col]  # Features
```
### Preprocesamiento: Indexador y ensamblador de features
```
indexer = StringIndexer(inputCol=target_col, outputCol="label")  # Convierte la etiqueta a numérica
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")  # Crea vector de features
pipeline = Pipeline(stages=[indexer, assembler])  # Pipeline con etapas
df_transformed = pipeline.fit(df).transform(df)  # Aplica pipeline
```
### Convertir el DataFrame de Spark a Pandas (para SageMaker)
```
df_pd = df_transformed.select("label", "features").toPandas()  # Convierte a Pandas
features_df = pd.DataFrame(df_pd["features"].tolist())  # Expande el vector
final_df = pd.concat([df_pd["label"], features_df], axis=1)  # Junta label y features
```
## 4. División de los Datos y Envío a S3

Dividir los datos en entrenamiento y test (80/20), guardar en CSV,
y subir ambos archivos a S3 para consumo de SageMaker.
```
from sklearn.model_selection import train_test_split  # Para separar sets
train, test = train_test_split(final_df, test_size=0.2, random_state=42)  # Split

train_file = "train.csv"  # Nombre archivo train
test_file = "test.csv"  # Nombre archivo test
train.to_csv(train_file, header=False, index=False)  # Guarda train
test.to_csv(test_file, header=False, index=False)  # Guarda test
```
### Subir los datos a S3 para SageMaker
```
prefix = "churn-xgboost"  # Prefijo carpeta en S3
train_s3_uri = sagemaker_session.upload_data(train_file, bucket=bucket, key_prefix=f"{prefix}/data")  # Sube train
test_s3_uri = sagemaker_session.upload_data(test_file, bucket=bucket, key_prefix=f"{prefix}/data")  # Sube test
```
## 5. Entrenamiento del Modelo XGBoost en SageMaker

Entrenar el modelo XGBoost en SageMaker usando el contenedor gestionado.
Se puede ajustar los hiperparámetros según las necesidades y recursos.
```
xgboost_estimator = XGBoost(
    entry_point=None,  # No se requiere script personalizado para entrenamiento básico
    framework_version="1.7-1",  # Versión del contenedor gestionado XGBoost
    instance_type="ml.m5.xlarge",  # Tipo de instancia (ajusta según recursos)
    instance_count=1,  # Número de nodos (puedes escalar horizontalmente)
    role=role,  # Rol de ejecución de SageMaker
    sagemaker_session=sagemaker_session,  # Sesión actual
    objective="binary:logistic",  # Clasificación binaria
    max_depth=6,  # Profundidad máxima
    num_round=100,  # Número de iteraciones
    learning_rate=0.1,  # Tasa de aprendizaje
    eval_metric="auc",  # Métrica de evaluación
    early_stopping_rounds=10,  # Regularización
    verbosity=2,  # Verbosidad
)

xgboost_estimator.fit(
    {
        "train": TrainingInput(train_s3_uri, content_type="csv"),  # Input de entrenamiento
        "validation": TrainingInput(test_s3_uri, content_type="csv"),  # Input de validación
    }
)
```
## 6. Despliegue y Predicción en Tiempo Real (Opcional)

Desplegar el modelo entrenado como endpoint para predicción en tiempo real.
Realizar una predicción de ejemplo y eliminar el endpoint cuando termines.
```
predictor = xgboost_estimator.deploy(
    initial_instance_count=1,  # Número de instancias para el endpoint
    instance_type="ml.m5.large",  # Tipo de instancia para el endpoint
)

import numpy as np  # Para manipulación de arrays

sample = test.iloc[0, 1:].values.astype(np.float32).reshape(1, -1)  # Toma una muestra del test set
prediction = predictor.predict(sample)  # Predice la probabilidad de churn
print(f"Predicción (probabilidad de churn): {prediction}")  # Muestra resultado

predictor.delete_endpoint()  # Elimina el endpoint si ya no se necesita
```

## Resumen y Recomendaciones

- Este pipeline utiliza PySpark para el preprocesamiento y SageMaker para entrenamiento y despliegue, lo que facilita la escalabilidad y robustez en AWS.
- Los datos se suben preprocesados a S3 y se entrena el modelo usando el contenedor gestionado de XGBoost.
- Es posible desplegar el modelo para predicción en tiempo real y liberar el endpoint cuando no se lo esté usando para evitar cargos.
- Ajustar los tipos de instancia y parámetros de entrenamiento según el tamaño de los datos y el presupuesto disponible.
- Para personalizaciones avanzadas (ingeniería de features, tuning, scripts de entrenamiento propios), revisar la [documentación de SageMaker XGBoost](https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html) y [PySpark](https://spark.apache.org/docs/latest/).

## Fuentes consultadas

- [AWS SageMaker XGBoost](https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html)  
- [PySpark (Spark) Documentation](https://spark.apache.org/docs/latest/)  
- [AWS SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)
