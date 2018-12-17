Proceso a seguir para los mapas:

1- Mediante un script (filter.py) con pyspark se filtra el fichero original risks.csv dando como resultado el fichero risks-f.csv
con las columnas que nos interesan

2- Mediante otro script (refined-filter.py) pasándole por parámetro aquel factor de riesgo que queremos filtrar realizamos transformaciones 
hasta tener un fichero .csv con dos columnas (abreviatura del estado, probabilidad de infarto)

3- Abrimos jupyter-notebook abiendo importado previamente las librerías pandas y folium y vamos ejecutando celda a celda 
el notebook (Heart Disease Map.ipynb) y nos generará archivos .html que contienen los mapas creados.
