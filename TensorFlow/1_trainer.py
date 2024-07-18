##############   Guia inicial de TensorFlow 2.0 para principiantes ##########
# Importa TensorFlow en tu programa
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
# Carga y prepara el conjunto de datos MNIST. Convierte los ejemplos de numeros enteros a numeros de punto flotante:
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Construye un modelo tf.keras.Sequential apilando capas. Escoge un optimizador y una funcion de perdida para
# el entrenamiento de tu modelo:

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrena y evalua el modelo:

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)