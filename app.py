import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Leer datos desde un archivo CSV
df = pd.read_csv('TERRENO_VENTA.csv')

# Convertir la columna 'ubicacion' a minúsculas para asegurar consistencia
df['ubicacion'] = df['ubicacion'].str.lower()

# Separar variables independientes y dependientes
X = df[['area', 'ubicacion', 'tipo']]
y = df['precio']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un pipeline para la transformación y el modelo
pipeline = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['area']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['ubicacion', 'tipo'])
        ])),
    ('model', LinearRegression())
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

def predecir_precio(area, ubicacion, tipo):
    # Crear un DataFrame para la predicción
    datos_nuevos = pd.DataFrame({
        'area': [area],
        'ubicacion': [ubicacion.lower()],  # Convertir la entrada a minúsculas
        'tipo': [tipo]
    })
    
    # Realizar la predicción
    precio_predicho = pipeline.predict(datos_nuevos)
    return precio_predicho[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    precio = None
    if request.method == 'POST':
        try:
            area = float(request.form['area'])
            ubicacion = request.form['ubicacion']
            tipo = request.form['tipo']
            precio = predecir_precio(area, ubicacion, tipo)
        except Exception as e:
            print(f"Error: {e}")  # Imprimir el error en la consola para depuración
            precio = None
    
    # Formatear el precio aquí
    precio_formateado = f"{precio:.2f}" if precio is not None else None

    return render_template('index.html', precio=precio_formateado)

if __name__ == "__main__":
    app.run(debug=True)