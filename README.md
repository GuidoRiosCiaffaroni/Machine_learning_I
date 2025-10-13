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

## https://www.kaggle.com/datasets/emmarex/plantdisease 

## 1. Resumen del dataset

El conjunto de datos “PlantVillage / PlantDisease” contiene imágenes de hojas de plantas, con distintas enfermedades o estados saludables. Cada imagen está etiquetada con la clase correspondiente (por ejemplo, planta saludable, o enfermedad específica). Las clases incluyen múltiples tipos de patologías foliares en cultivos comunes (por ejemplo, tomate, papa, etc.) y también clases de hojas sanas. El dataset cuenta con un número considerable de instancias (varias decenas de miles de imágenes) y múltiples clases categóricas. Dado que se trata de un dataset visual, las variables son pixeles (canales RGB) y las clases son etiquetas nominales. El propósito original suele estar orientado al reconocimiento automático de enfermedades en plantas mediante modelos de clasificación supervisada (visión por computador).

## 2. Problemas del mundo real que podrías abordar con aprendizaje no supervisado

Aunque el dataset fue concebido con un enfoque supervisado (diagnóstico de enfermedades), hay varias aplicaciones interesantes para aprendizaje no supervisado:

- Agrupamiento de patrones foliares: detectar clústeres de hojas con características visuales similares, posiblemente descubriendo subtipos de enfermedades no etiquetadas o variantes emergentes.

- Detección de anomalías / outliers: identificar imágenes que no encajan en ningún grupo conocido (por ejemplo, hojas con síntomas nuevos o errores en captura de imagen).

- Reducción de dimensionalidad / visualización: proyectar las imágenes (o sus embeddings) en espacios de menor dimensión para inspección visual interactiva de la estructura latente de los datos.

- Aprendizaje de representaciones no etiquetadas: usar autoencoders o técnicas de embedding para aprender representaciones latentes de las imágenes que luego pueden servir como insumo para tareas supervisadas con pocas etiquetas (aprendizaje semi-supervisado).

Estos enfoques pueden ayudar a los investigadores o agricultores a descubrir patrones desconocidos, diagnosticar síntomas emergentes o preparar sistemas de monitoreo que no dependan enteramente de etiquetas humanas.

## 3. Técnicas de aprendizaje no supervisado recomendadas y su justificación

Dadas las características del dataset (datos de imagen, alta dimensionalidad, clases múltiples, posibilidad de nuevos patrones), las siguientes técnicas serían apropiadas:

- Autoencoders (redes neuronales): entrenar un autoencoder convolucional para reconstrucción de imágenes. La capa latente (embedding) puede servir para agrupar imágenes similares. Es especialmente útil porque maneja bien alta dimensionalidad de entrada.

- Variational Autoencoders (VAEs) o Autoencoders Denoising: generan una distribución estructurada en el espacio latente y pueden ayudar a identificar regiones poco densas (anomalías).

- Clustering sobre embeddings: una vez que tienes embeddings (por ejemplo, la capa latente del autoencoder), aplicar métodos como K-means, DBSCAN o HDBSCAN en ese espacio de menor dimensión para agrupar instancias con características similares.

- t-SNE / UMAP / PCA para reducción de dimensionalidad exploratoria: permiten visualizar las relaciones entre instancias en 2D o 3D, para inspección manual de grupos latentes o separaciones de clases naturales.

- Clustering jerárquico (agglomerativo) sobre embeddings o sobre vectores de características extraídas (por ejemplo, saliencia, histograma de color, texturas) puede revelar subgrupos dentro de cada clase diagnóstica.

Justificación:

- Las imágenes tienen una dimensionalidad muy alta (muchos pixeles), por lo que aplicar directamente clustering en el espacio original puede no ser efectivo ni computacionalmente viable.

- Usar autoencoders (o técnicas de embeddings) permite reducir la dimensionalidad preservando las estructuras latentes relevantes.

- Algoritmos como DBSCAN o HDBSCAN permiten encontrar grupos de densidad arbitraria (útil si los grupos no son esféricos ni balanceados).

- t-SNE / UMAP son herramientas excelentes para visualización de alta dimensión en espacios bajos, facilitando la interpretación del espacio latente.

## 4. Desafíos potenciales

Al trabajar con este dataset en un contexto no supervisado, surgen varios desafíos que es importante anticipar:

### Alta dimensionalidad / curse of dimensionality
- Las imágenes tienen miles o millones de dimensiones (pixeles), lo que complica que los algoritmos de clustering funcionen directamente en ese espacio (distancias pueden volverse poco informativas). Se requiere reducción de dimensionalidad o extracción de características antes de aplicar clustering.

### Datos faltantes / imágenes corruptas
- Algunas imágenes pueden estar dañadas, incompletas o con artefactos. Esto puede afectar la calidad de embeddings o provocar errores en entrenamiento de autoencoders. Será necesario filtrar, limpiar o imputar estos casos.

### Desbalance de clases / desigualdad de frecuencias
- Si algunas enfermedades tienen muchas más imágenes que otras, los clusters dominantes pueden estar sesgados hacia clases mayoritarias, dejando grupos menores menos bien capturados. Esto puede llevar a que agrupaciones de enfermedades raras sean absorbidas por clusters mayores.

### Selección de número de clusters y parámetros
- Métodos como K-means requieren decidir k. Elegir el número óptimo de clusters en un entorno no supervisado es un desafío (usar métricas de validación como silhouette, Davies–Bouldin, etc.). En DBSCAN, elegir los parámetros de densidad (ε, min_samples) es crítico y dependiente del dataset.

### Interpretabilidad de los clusters
- Una vez obtenidos clusters, interpretar qué representan (¿subtipos de enfermedad? ¿errores de imagen?) no es trivial. Puede requerir inspección manual experta y validación biológica.

### Complejidad computacional / recursos de hardware
- El entrenamiento de autoencoders convolucionales o de grandes modelos de embedding puede exigir GPU, memoria considerable y tiempo de cómputo. Si el dataset es muy grande, puede requerir estrategias de muestreo, mini-batching o técnicas de entrenamiento escalable.

### Ruido y variabilidad en imagen
- Las imágenes pueden tener variaciones de iluminación, ángulos de captura, fondo, escala o rotación. Ese ruido puede dificultar la convergencia del modelo de representación latente o inducir clusters basados en artefactos en lugar de en características patológicas reales.

### Sesgo de muestreo / generalización
- El dataset puede estar sesgado hacia ciertas regiones geográficas, condiciones de cultivo o especies específicas. Los clusters latentes pueden capturar sesgos en lugar de patrones universales.


## https://www.kaggle.com/datasets/akshitmadan/eyes-open-or-closed

## 1. Resumen del dataset

El dataset “Eyes-Open or Closed” contiene imágenes de ojos humanos en dos estados: abiertos (open) o cerrados (closed). Está orientado al reconocimiento del estado del ojo, con aplicaciones como la detección de somnolencia, parpadeo o atención visual. Cada instancia es una imagen (probablemente en escala de grises o color; tamaño fijo tras el preprocesamiento), y la etiqueta es binaria (“open” vs “closed”). El dataset puede tener decenas de miles de imágenes distribuidas entre los dos estados. Así, las variables son los pixeles de cada imagen (posiblemente preprocesadas en una dimensión estándar) y la clase binaria del estado del ojo. El propósito principal suele ser entrenar modelos supervisados de clasificación binaria, pero aquí lo consideraremos desde la perspectiva no supervisada.

## 2. Problemas del mundo real que podrías abordar con aprendizaje no supervisado

Usando aprendizaje no supervisado con este dataset, se pueden explorar y resolver varias situaciones reales:

- Detección de anomalías en el estado del ojo: por ejemplo, imágenes que no correspondan claramente a “abierto” ni “cerrado”, que podrían indicar errores de captura, parpadeos parciales, o condiciones atípicas (ojos medio cerrados) no contempladas por las clases originales.

- Clustering de subestados visuales: agrupar imágenes dentro de “open” o “closed” para descubrir subtipos visuales debido a iluminación, ángulo, pliegue del párpado, reflejos, etc. Esto puede ayudar a segmentar variabilidad intra-clase que un modelo supervisado podría obviar.

- Análisis exploratorio de estructura visual: mediante reducción de dimensionalidad, identificar si las imágenes “open” y “closed” separan naturalmente en el espacio latente, o si hay regiones de solapamiento.

- Preprocesamiento no supervisado para clasificación con pocas etiquetas: usar representación latente obtenida por métodos no supervisados para entrenar modelos supervisados con menor necesidad de datos etiquetados (aprendizaje semi-supervisado).

- Detección de parpadeos o transición continua: si se tuvieran secuencias, podría detectarse el cambio del estado open → closed a través de agrupamientos temporales o clustering secuencial.

Estos enfoques pueden ser útiles en sistemas de monitoreo de fatiga, interfaces cerebro-computadora (BCI), vigilancia de conductor (detectar somnolencia), etc.

## 3. Técnicas de aprendizaje no supervisado recomendadas y su justificación

Dadas las características del dataset (imágenes binarias de ojos, alta dimensionalidad, posibilidad de variabilidad de iluminación/ángulo), las siguientes técnicas serían adecuadas:

- Autoencoders / Autoencoders Convolucionales:
Construir un autoencoder que reconstruya la imagen original obliga a aprender una representación latente comprimida. Esa representación (vector latente) puede capturar los rasgos esenciales que distinguen estados “open” y “closed”. Luego, sobre ese espacio latente, se pueden aplicar clustering.

- Variational Autoencoder (VAE) o Denoising Autoencoder:
Estas variantes permiten estructurar el espacio latente (distribución gaussiana en VAE) o ser robusto al ruido (denoising). Un VAE puede ayudar a identificar regiones densas vs regiones escasas (posibles anomalías) en el espacio latente.

- Clustering sobre embeddings:
Una vez que se dispone de los vectores latentes, métodos como K-means, DBSCAN, HDBSCAN o Gaussian Mixture Models (GMM) pueden agrupar instancias. Por ejemplo:

- - K-means si se supone que hay dos clusters principales (open vs closed) o inclusive subdivisiones intermedias.

- - DBSCAN/HDBSCAN para detectar agrupamientos densos y separar outliers sin forzar una cantidad fija de clusters.

- - GMM si se sospecha que los clusters no son esféricos o tienen densidades distintas.

- PCA / t-SNE / UMAP para reducción de dimensionalidad exploratoria:
Aplicar PCA para reducir dimensiones linealmente, o usar t-SNE / UMAP para proyectar en 2D/3D con preservación de la estructura local, para visualizar la separación entre estados y detectar zonas de solapamiento o anomalías.

Clustering jerárquico (aglomerativo) sobre embeddings o características extraídas puede revelar jerarquías de variabilidad (por ejemplo, subgrupos dentro de “open” según ángulo de mirada, iluminación, tamaño de ojo).

Justificación técnica:

- Usar embeddings o autoencoders es esencial para reducir la dimensionalidad y evitar que el clustering en espacio de pixeles puro sea inefectivo o dominado por ruido.

- Métodos de clustering robustos como DBSCAN o HDBSCAN pueden manejar formas arbitrarias y detectar puntos atípicos (outliers).

- PCA / t-SNE / UMAP permiten inspección visual y diagnóstico de la calidad de separación en el espacio latente.

- Las técnicas probabilísticas (GMM) pueden modelar densidades subyacentes de cada grupo.

## 4. Desafíos potenciales

Al aplicar aprendizaje no supervisado a este dataset, se enfrentarán diversos retos:

### Alta dimensionalidad de las imágenes
Cada imagen puede tener cientos o miles de pixeles. Sin reducción o codificación previa, los algoritmos de clustering pueden no funcionar bien por el “curse of dimensionality”.

### Variabilidad de iluminación, ángulo, contraste y ruido
Las diferencias en condiciones de captura (sombras, reflejos, ojo parcialmente oscurecido) pueden introducir ruido visual significativo que distraiga los modelos.

### Superposición visual entre clases
Algunas imágenes de ojos “semi cerrados” o estados intermedios pueden mezclarse visualmente entre “open” y “closed”, provocando zonas de solapamiento en el espacio latente, lo que dificulta la separación clara.

### Selección de hiperparámetros de clustering
Escoger el número de clusters (en K-means), parámetros de densidad (ε, min_samples en DBSCAN), covarianzas (en GMM), etc., es no trivial. Se requerirá experimentación y métricas de evaluación no supervisadas (silhouette, Davies–Bouldin) para guiar la elección.

### Interpretabilidad de los clusters
Aunque los clusters se obtengan exitosamente, interpretar qué variaciones capturan (iluminación, ángulo, pliegue de párpado) puede requerir esfuerzo manual y expertos en visión.

### Limitaciones de datos atípicos o imágenes corruptas
Algunas imágenes podrían estar mal recortadas, parcialmente ocultas o con artefactos. Debe realizarse limpieza previa, filtrado o detección de anomalías antes del entrenamiento.

### Sesgo de captura / distribución desequilibrada
Si hay más imágenes de un estado (open) que del otro, puede haber sesgo en la representación latente hacia la clase mayoritaria. Además, puede haber sesgos según demografía (ojos de ciertas tonalidades, etnias, edades) que afecten la generalización.

### Complejidad computacional / recursos
Entrenar autoencoders CNN, optimizar hiperparámetros de clustering y proyectar en t-SNE/UMAP puede demandar GPU, memoria de GPU/CPU y tiempo significativo, especialmente si el dataset es voluminoso.

### Validación de resultados en ausencia de etiquetas supervisadas
En aprendizaje no supervisado la evaluación es más complicada: no tienes “ground truth” para comparar directamente. Deberás confiar en métricas intrínsecas (silhouette, cohesion/sepación) más revisiones cualitativas (inspección visual) para validar que los clusters son significativos.

## https://www.kaggle.com/datasets/warcoder/cats-vs-dogs-vs-birds-audio-classification

## 1. Resumen del dataset

El conjunto de datos “Cats vs Dogs vs Birds — Audio Classification” está compuesto por clips de audio que corresponden a sonidos producidos por gatos, perros o aves. Cada muestra de audio tiene asignada una de esas tres clases (cat / dog / bird). Las variables son, esencialmente, formas de onda de audio (por ejemplo, muestras en el dominio del tiempo, frecuencias o transformaciones espectrales como espectrogramas, MFCCs, etc.). El propósito clásico es entrenar modelos de clasificación de sonidos para identificar la especie que emite el sonido. Aunque la tarea típica es supervisada, aquí la analizaremos desde la óptica del aprendizaje no supervisado.

## 2. Posibles problemas del mundo real que podrías resolver con aprendizaje no supervisado

Al aplicar técnicas no supervisadas a este dataset, se pueden abordar diversos problemas prácticos:

- Descubrimiento de subtipos de sonidos: dentro de cada clase (gato, perro o ave), pueden existir variaciones acústicas (diferentes razas, entornos, tonalidades). El clustering puede revelar estos subgrupos sin necesidad de etiquetado detallado.

- Detección de anomalías acústicas: identificar grabaciones que no correspondan claramente a ninguna de las tres clases (ruido ambiental, interferencia, sonidos extraños) como outliers.

- Segmentación acústica no etiquetada: agrupar los clips de audio basándose en similitud espectral, lo cual puede permitir añadir nuevas categorías (por ejemplo, tipo de canto de aves o ladrido de distintos perros).

- Representación latente para uso posterior: aprender representaciones compactas (embeddings) del audio que luego puedan emplearse en tareas supervisadas con escasísimas etiquetas (aprendizaje semi-supervisado).

- Análisis exploratorio de la estructura del espacio acústico: mediante reducción dimensional, visualizar cómo se disponen los sonidos de las tres clases si las fronteras naturales emergen o si hay solapamiento.

Estos usos pueden servir en ecología (identificación de especies por sonido), vigilancia sonora urbana (detectar presencia de gatos o aves en zonas residenciales), monitoreo de fauna o detección de fauna urbana no deseada.

## 3. Técnicas de aprendizaje no supervisado recomendadas y su justificación

Dado que estamos tratando con señales de audio (señales en el dominio del tiempo o frecuencias), y que la dimensionalidad puede ser elevada, estas técnicas serían apropiadas:

- Autoencoders / Autoencoders convolucionales / Autoencoders recurrentes
Para el dominio del audio, se puede diseñar un autoencoder que trabaje sobre espectrogramas (por ejemplo con capas convolucionales) o directamente sobre secuencias (con redes recurrentes). El objetivo es aprender una representación latente comprimida que capture la esencia del sonido.

- Variational Autoencoders (VAEs) o Autoencoders Denoising
Un VAE puede imponer estructura probabilística al espacio latente, facilitando la detección de anomalías (lugares de baja densidad). Un autoencoder denoising ayuda a que el modelo sea robusto frente a ruido ambiental presente en grabaciones.

- Clustering sobre embeddings
Una vez que cada clip de audio se representa como un vector latente, aplicar K-means, GMM, DBSCAN, HDBSCAN es viable para agrupar muestras acústicamente similares. Por ejemplo:

- - K-means si creemos que hay tres grandes clusters (gato / perro / ave), o incluso subdivisiones dentro de cada clase.

- - GMM para modelar clusters con diferentes formas de densidad o covarianza.

- - DBSCAN/HDBSCAN para agrupar densidades arbitrarias y detectar outliers (grabaciones extrañas).

- Reducción de dimensionalidad exploratoria: PCA / t-SNE / UMAP
Aplicar PCA sobre los embeddings para ver cuánta varianza capturan las primeras componentes, o usar t-SNE / UMAP para proyectar los datos en 2D o 3D y visualizar la agrupación y separación de las clases acústicas.

- Clustering jerárquico (aglomerativo)
En el espacio latente, usar clustering jerárquico puede revelar relaciones jerárquicas (por ejemplo, dentro de aves, subdivisiones por especie, tipo de canto, frecuencia dominante).

Justificación técnica:

- El dominio del audio tiene alta dimensionalidad y estructuras locales (temporal / espectral), por lo que usar embeddings mediante autoencoders ayuda a extraer características latentes relevantes.

- Métodos de clustering aplicados en el espacio reducido (embedding) suelen tener mejor comportamiento que directamente en el espacio original de señales.

- Técnicas de densidad (como DBSCAN) permiten manejar clusters de forma no esférica y detectar grabaciones no conformes como anomalías.

- PCA / t-SNE / UMAP son útiles para interpretación visual y diagnóstico de la separación latente.

## 4. Desafíos potenciales

Al trabajar con este dataset en un contexto no supervisado, es necesario anticipar los siguientes retos:

### Alta dimensionalidad de las señales / transformaciones espectrales
Las señales de audio, transformadas en espectrogramas (por ejemplo, STFT), pueden resultar en matrices de alta dimensión (tiempo × frecuencia). Sin una reducción previa, los algoritmos de clustering no funcionarán bien.

### Ruido ambiental y capturas imperfectas
Grabaciones pueden contener ruido de fondo, ecos, interferencias o recortes abruptos, lo que complica que el modelo de representación latente capture solo la “parte relevante” del sonido de mascota.

### Variabilidad en el volumen, micrófono y entorno
Diferencias en equipos de captura, acústica del lugar, distancia al micrófono, volumen del sonido pueden generar variabilidad no deseada en los embeddings.

### Desbalance de clases
Si hay muchas más muestras de una clase (por ejemplo, muchos sonidos de perros respecto a aves), el clustering puede sesgarse a agrupar predominantemente los casos abundantes y pasar por alto las clases menos representadas.

### Determinación del número de clusters / parámetros de clustering
Elegir el número óptimo de clusters (en K-means), parámetros de densidad (en DBSCAN), o número de componentes latentes (en autoencoder) es un desafío en el entorno no supervisado. Deberás apoyarte en métricas internas (silhouette, Davies–Bouldin, cohesion/separación) y validaciones visuales.

### Interpretabilidad de los clusters acústicos
Identificar qué características distinguen un cluster u otro (por ejemplo, frecuencia dominante, duración, patrón espectral) requerirá análisis acústico o inspección de espectrogramas, posiblemente con expertos en audio.

### Limitaciones de datos corruptos o clips fallidos
Algunos archivos de audio pueden estar dañados, tener cortes abruptos o estar incompletos. Es necesario filtrarlos, limpiarlos o descartarlos antes del procesamiento.

### Complejidad computacional
Entrenar autoencoders sobre grandes conjuntos de espectrogramas, hacer clustering sobre grandes volúmenes de datos, y proyectar en t-SNE / UMAP puede consumir mucha memoria y tiempo de cómputo o requerir GPU.

### Validación sin etiquetas
La evaluación del desempeño del clustering es menos directa que en supervisión: no hay ground truth confiable para comparar. Se necesitarán métricas internas de calidad del clustering y validación cualitativa (inspección acústica de muestras representativas de cada cluster).