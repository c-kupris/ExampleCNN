# imports
from datasets import load_dataset
import keras
import numpy  # Use numpy version 1.26.4

# Import training data.
binding_affinity_dataset = load_dataset("jglaser/binding_affinity", split="train")  # 1.84M rows

# Define hyperparameters.
batch_size = 1
epochs = 10
filters = 1
kernel_size = 1
pool_size = 1

# make a Keras Sequential model.
model = keras.Sequential()

# Create the layers for the model.
input_layer = keras.layers.InputLayer(shape=(1, 1, ), batch_size=batch_size)

# Layer one
layer_one_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_one_batch_normalization_layer = keras.layers.BatchNormalization()
layer_one_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Layer two
layer_two_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_two_batch_normalization_layer = keras.layers.BatchNormalization()
layer_two_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Layer three
layer_three_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_three_batch_normalization_layer = keras.layers.BatchNormalization()
layer_three_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Layer four
layer_four_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_four_batch_normalization_layer = keras.layers.BatchNormalization()
layer_four_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Layer five
layer_five_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_five_batch_normalization_layer = keras.layers.BatchNormalization()
layer_five_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Layer six
layer_six_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_six_batch_normalization_layer = keras.layers.BatchNormalization()
layer_six_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Layer seven
layer_seven_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_seven_batch_normalization_layer = keras.layers.BatchNormalization()
layer_seven_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Layer eight
layer_eight_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_eight_batch_normalization_layer = keras.layers.BatchNormalization()
layer_eight_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Layer nine
layer_nine_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_nine_batch_normalization_layer = keras.layers.BatchNormalization()
layer_nine_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Layer ten
layer_ten_conv_layer = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)
layer_ten_batch_normalization_layer = keras.layers.BatchNormalization()
layer_ten_pooling_layer = keras.layers.AveragePooling1D(pool_size=pool_size)

# Add the layers to the model.
model.add(input_layer)
model.add(layer_one_conv_layer)
model.add(layer_one_batch_normalization_layer)
model.add(layer_one_pooling_layer)
model.add(layer_two_conv_layer)
model.add(layer_two_batch_normalization_layer)
model.add(layer_two_pooling_layer)
model.add(layer_three_conv_layer)
model.add(layer_three_batch_normalization_layer)
model.add(layer_three_pooling_layer)
model.add(layer_four_conv_layer)
model.add(layer_four_batch_normalization_layer)
model.add(layer_four_pooling_layer)
model.add(layer_five_conv_layer)
model.add(layer_five_batch_normalization_layer)
model.add(layer_five_pooling_layer)
model.add(layer_six_conv_layer)
model.add(layer_six_batch_normalization_layer)
model.add(layer_six_pooling_layer)
model.add(layer_seven_conv_layer)
model.add(layer_seven_batch_normalization_layer)
model.add(layer_seven_pooling_layer)
model.add(layer_eight_conv_layer)
model.add(layer_eight_batch_normalization_layer)
model.add(layer_eight_pooling_layer)
model.add(layer_nine_conv_layer)
model.add(layer_nine_batch_normalization_layer)
model.add(layer_nine_pooling_layer)
model.add(layer_ten_conv_layer)
model.add(layer_ten_batch_normalization_layer)
model.add(layer_ten_pooling_layer)

# Choose a loss function for the model.
loss = keras.losses.CategoricalCrossentropy()

# Choose an optimizer for the model.
optimizer = keras.optimizers.SGD()

# Choose metrics for the model.
metrics = keras.metrics.CategoricalAccuracy()

# Summarize the model.
model.summary()

# Compile the model.
model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])

# Split the data into x_train and y_train.
x_train = binding_affinity_dataset['seq']
y_train = binding_affinity_dataset['affinity']

# Preprocess the data.
# x_train is a collection of strings and y_train is a collection of floats.
# Turn strings in x_train to floats.
'''number_of_occurrences = []
unique_proteins = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7,
                   'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14,
                   'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20, 'B': 21,
                   'Z': 22}
unique_protein_scores = {'A': [], 'R': [], 'N': [], 'D': [], 'C': [], 'Q': [], 'E': [],
                         'G': [], 'H': [], 'I': [], 'L': [], 'K': [], 'M': [], 'F': [],
                         'P': [], 'S': [], 'T': [], 'W': [], 'Y': [], 'V': [], 'B': [],
                         'Z': []}
for x in range(len(x_train)):
    if unique_proteins.keys().__contains__(x_train[x]):
        if unique_protein_scores.keys().__eq__(x_train[x]):
            number_of_occurrences += 1
            unique_protein_scores.update(dict["number of occurrences": number_of_occurrences])'''

# Make a Vectorization layer
vectorize = keras.layers.TextVectorization()

vectorized_x_train_as_numpy_array = x_train[~numpy.any(numpy.isnan(x_train), axis=1)]

# Adapt the Vectorization layer to the data.
vectorize.adapt(vectorized_x_train_as_numpy_array)

# Vectorize the data.
vectorized_x_train = vectorize(vectorized_x_train_as_numpy_array)

# Make sure the x_train data was vectorized.
print("vectorized x_train data: " + vectorized_x_train.__str__())

#  Make sure the type of the x_train data is correct.
print("vectorized_x_train type: " + vectorized_x_train_as_numpy_array.__str__())

# Make sure the vectorized_x_train_as_numpy_array is the right size before flattening it.
print("vectorized_x_train_as_numpy_array size: " + vectorized_x_train_as_numpy_array.__str__())


# Make sure the x_train data is flattened.
print("flattened x_train shape: " + vectorized_x_train_as_numpy_array.__str__())

print('x_train type: ' + vectorized_x_train_as_numpy_array.__str__())

# Train the model.
model.fit(x=vectorized_x_train,
          epochs=epochs)
