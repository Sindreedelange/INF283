# MNIST
## using 4 classifiers: Random Forest, XGBoost, Neural Network, and Convolutional Neural Network 

# Project Organization
 ------------
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── Main.py            <- Run to print the (important) dictionaries
    │
    ├── dictionaries            <- Dictionaries stores as numpy arrays:
    │                          classifiers_dict.npy - Score for each classifier
    │                          grid_search_cnn.npy -  
    │                          optim_classifier_cnn_params.npy -
    │                          optim_classifier_dict.npy - 
    │
    ├── nbs                <- Notebooks
    │                           MNIST_Keras - The One Notebook To Rule Them All
    │
    ├── reports            <- PDf-file answer to task sheet
    │
    ├── models            <- Stored classifiers
    │                          MNIST_RF.pickle.dat - Trained Random Forest Classifier
    │                          MNIST_XGB.pickle.dat - Trained XGBoost Classifier 
    │                          MNIST_cnn.hdf5 - Trained Convolutional Neural Network
    │                          MNIST_cnn_optim.hdf5 - Trained Convolutional Neural Network, using Grid Search
    │                          MNIST_nn.hdf5 - Trained Neural Network
    │
    ├── utils              <- *.py* files to prettify the Notebook
    │                           Data.py - get, transform, and store data
    │                           Models.py - get, train, and store models
    │                           Plots.py - make pyplots

------------

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
2. Download the data from `https://pjreddie.com/projects/mnist-in-csv/`,
and store them in `/data/`:
------------
    ├── data         
    │
    ├── handwritten_digits_images.csv        <- Images
    ├── handwritten_digits_labels            <- Labels
------------

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
Example: <br> ```cnn_model = keras.models.load_model('models/MNIST_cnn_optim.hdf5)```
- Random Forest & XGBoost
```model = pickle.load(open(full_path, "rb"))```
Example: ```loaded_model_rf = pickle.load(open("../models/MNIST_RF.pickle.dat", "rb"))```

### NOTE: 
.csv files too big for GitHub
Can be found here:
`https://pjreddie.com/projects/mnist-in-csv/`
Remember to store them in "data/" folder
