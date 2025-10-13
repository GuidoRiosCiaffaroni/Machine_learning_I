# Proyecto: Exploración y Propuesta de Problema con Machine Learning no supervisado

 **Objetivo de esta primera etapa**:
Identificar una fuente de datos confiable y proponer un problema relevante que pueda ser resuelto utilizando técnicas de Machine Learning no supervisado, justificando su importancia y viabilidad.

## Parte 1: Exploración de datasets y problemas

**Paso 1**: Investigar fuentes de datos abiertos
Explora al menos dos sitios de datasets reconocidos para Deep Learning:

https://www.kaggle.com/datasets

https://huggingface.co/datasets

https://data.gov/ (para datos públicos y gubernamentales)

https://archive.ics.uci.edu/ml/index.php (UCI Machine Learning Repository)

https://paperswithcode.com/datasets

Seleccion de Datos

Imagenes

https://www.kaggle.com/datasets/emmarex/plantdisease

https://www.kaggle.com/datasets/akshitmadan/eyes-open-or-closed

https://www.kaggle.com/datasets/akshitmadan/eyes-open-or-closed


Audio

https://www.kaggle.com/datasets/warcoder/cats-vs-dogs-vs-birds-audio-classification

CSV

https://www.kaggle.com/code/erimsaholut/student-depression-dataset

https://www.kaggle.com/code/mahdimashayekhi/predicting-disease-risk-from-daily-habits/input

https://www.kaggle.com/datasets/athirags/car-data

TXT

https://www.kaggle.com/datasets/tentotheminus9/religious-and-philosophical-texts



# Resumen 

# ** https://www.kaggle.com/datasets/emmarex/plantdisease **

1. Resumen del dataset

El conjunto de datos “PlantVillage / PlantDisease” contiene imágenes de hojas de plantas, con distintas enfermedades o estados saludables. Cada imagen está etiquetada con la clase correspondiente (por ejemplo, planta saludable, o enfermedad específica). Las clases incluyen múltiples tipos de patologías foliares en cultivos comunes (por ejemplo, tomate, papa, etc.) y también clases de hojas sanas. El dataset cuenta con un número considerable de instancias (varias decenas de miles de imágenes) y múltiples clases categóricas. Dado que se trata de un dataset visual, las variables son pixeles (canales RGB) y las clases son etiquetas nominales. El propósito original suele estar orientado al reconocimiento automático de enfermedades en plantas mediante modelos de clasificación supervisada (visión por computador).

2. Problemas del mundo real que podrías abordar con aprendizaje no supervisado

Aunque el dataset fue concebido con un enfoque supervisado (diagnóstico de enfermedades), hay varias aplicaciones interesantes para aprendizaje no supervisado:

Agrupamiento de patrones foliares: detectar clústeres de hojas con características visuales similares, posiblemente descubriendo subtipos de enfermedades no etiquetadas o variantes emergentes.

Detección de anomalías / outliers: identificar imágenes que no encajan en ningún grupo conocido (por ejemplo, hojas con síntomas nuevos o errores en captura de imagen).

Reducción de dimensionalidad / visualización: proyectar las imágenes (o sus embeddings) en espacios de menor dimensión para inspección visual interactiva de la estructura latente de los datos.

Aprendizaje de representaciones no etiquetadas: usar autoencoders o técnicas de embedding para aprender representaciones latentes de las imágenes que luego pueden servir como insumo para tareas supervisadas con pocas etiquetas (aprendizaje semi-supervisado).

Estos enfoques pueden ayudar a los investigadores o agricultores a descubrir patrones desconocidos, diagnosticar síntomas emergentes o preparar sistemas de monitoreo que no dependan enteramente de etiquetas humanas.

3. Técnicas de aprendizaje no supervisado recomendadas y su justificación

Dadas las características del dataset (datos de imagen, alta dimensionalidad, clases múltiples, posibilidad de nuevos patrones), las siguientes técnicas serían apropiadas:

Autoencoders (redes neuronales): entrenar un autoencoder convolucional para reconstrucción de imágenes. La capa latente (embedding) puede servir para agrupar imágenes similares. Es especialmente útil porque maneja bien alta dimensionalidad de entrada.

Variational Autoencoders (VAEs) o Autoencoders Denoising: generan una distribución estructurada en el espacio latente y pueden ayudar a identificar regiones poco densas (anomalías).

Clustering sobre embeddings: una vez que tienes embeddings (por ejemplo, la capa latente del autoencoder), aplicar métodos como K-means, DBSCAN o HDBSCAN en ese espacio de menor dimensión para agrupar instancias con características similares.

t-SNE / UMAP / PCA para reducción de dimensionalidad exploratoria: permiten visualizar las relaciones entre instancias en 2D o 3D, para inspección manual de grupos latentes o separaciones de clases naturales.

Clustering jerárquico (agglomerativo) sobre embeddings o sobre vectores de características extraídas (por ejemplo, saliencia, histograma de color, texturas) puede revelar subgrupos dentro de cada clase diagnóstica.

Justificación:

Las imágenes tienen una dimensionalidad muy alta (muchos pixeles), por lo que aplicar directamente clustering en el espacio original puede no ser efectivo ni computacionalmente viable.

Usar autoencoders (o técnicas de embeddings) permite reducir la dimensionalidad preservando las estructuras latentes relevantes.

Algoritmos como DBSCAN o HDBSCAN permiten encontrar grupos de densidad arbitraria (útil si los grupos no son esféricos ni balanceados).

t-SNE / UMAP son herramientas excelentes para visualización de alta dimensión en espacios bajos, facilitando la interpretación del espacio latente.

4. Desafíos potenciales

Al trabajar con este dataset en un contexto no supervisado, surgen varios desafíos que es importante anticipar:

Alta dimensionalidad / curse of dimensionality
Las imágenes tienen miles o millones de dimensiones (pixeles), lo que complica que los algoritmos de clustering funcionen directamente en ese espacio (distancias pueden volverse poco informativas). Se requiere reducción de dimensionalidad o extracción de características antes de aplicar clustering.

Datos faltantes / imágenes corruptas
Algunas imágenes pueden estar dañadas, incompletas o con artefactos. Esto puede afectar la calidad de embeddings o provocar errores en entrenamiento de autoencoders. Será necesario filtrar, limpiar o imputar estos casos.

Desbalance de clases / desigualdad de frecuencias
Si algunas enfermedades tienen muchas más imágenes que otras, los clusters dominantes pueden estar sesgados hacia clases mayoritarias, dejando grupos menores menos bien capturados. Esto puede llevar a que agrupaciones de enfermedades raras sean absorbidas por clusters mayores.

Selección de número de clusters y parámetros
Métodos como K-means requieren decidir k. Elegir el número óptimo de clusters en un entorno no supervisado es un desafío (usar métricas de validación como silhouette, Davies–Bouldin, etc.). En DBSCAN, elegir los parámetros de densidad (ε, min_samples) es crítico y dependiente del dataset.

Interpretabilidad de los clusters
Una vez obtenidos clusters, interpretar qué representan (¿subtipos de enfermedad? ¿errores de imagen?) no es trivial. Puede requerir inspección manual experta y validación biológica.

Complejidad computacional / recursos de hardware
El entrenamiento de autoencoders convolucionales o de grandes modelos de embedding puede exigir GPU, memoria considerable y tiempo de cómputo. Si el dataset es muy grande, puede requerir estrategias de muestreo, mini-batching o técnicas de entrenamiento escalable.

Ruido y variabilidad en imagen
Las imágenes pueden tener variaciones de iluminación, ángulos de captura, fondo, escala o rotación. Ese ruido puede dificultar la convergencia del modelo de representación latente o inducir clusters basados en artefactos en lugar de en características patológicas reales.

Sesgo de muestreo / generalización
El dataset puede estar sesgado hacia ciertas regiones geográficas, condiciones de cultivo o especies específicas. Los clusters latentes pueden capturar sesgos en lugar de patrones universales.


