# MNIST, using 4 classifiers: Random Forest, XGBoost, Neural Network, and Convolutional Neural Network
Using Sklearn, TensorFlow w. Keras, and Python 3.  

# Building

## Prerequisites (To run the NoteBook)
1. Python 3
2. Keras
3. TensorFlow
4. Numpy
5. Matplot
6. pandas
7. Sklearn

## Instructions
1. Clone the repository
2. Run the Jupyter Notebook: '/nbs/MNIST_Classification_Keras.ipynb'

## Prerequisites (To see results, and potentially load models)
1. Python 3
2. Numpy
3. Keras
4. TensorFlow

## Instructions
1. Clone the repository
2. Navigate to 'Main.py'
3. Run it

### To load the models
- Neural Networks: <br>
```model = keras.models.load_model(model_path)```
Example: ```cnn_model = keras.models.load_model('models/MNIST_cnn_optim.hdf5)```
- Random Forest & XGBoost
```model = pickle.load(open(full_path, "rb"))```
Example: ```loaded_model_rf = pickle.load(open("../models/MNIST_RF.pickle.dat", "rb"))```

### NOTE: 
.csv files too big for GitHub
Can be found here:
`https://pjreddie.com/projects/mnist-in-csv/`
Remember to store them in "data/" folder
