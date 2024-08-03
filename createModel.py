import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # removed a warning about floating point problems
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

randomSeed = 8 # this can be any number

# get paths for dataset and model
datasetName = 'chordDataset.csv'
root = os.path.dirname(__file__)
datasetPath = os.path.join(root, datasetName)


def main():
     # get number of chord classes
    class_dict, numClasses, chords = count_classes(datasetPath)

    # prepare data
    X_dataset = np.loadtxt(datasetPath, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 2)))
    Y_dataset = np.loadtxt(datasetPath, delimiter=',', dtype='str', usecols=(0))
    Y_dataset = [class_dict[classifier] for classifier in Y_dataset] # convert classifiers into ints, since strings weren't working properly

    # create training and testing sets + put them into numpy arrays
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, Y_dataset, train_size=0.75, random_state=randomSeed)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((43, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(numClasses, activation='softmax')
    ])
    
    # compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # fit model
    model.fit(
        X_train,
        y_train,
        epochs=500,
        validation_data=(X_test, y_test)
    )

    # save model to a file
    model.save(f'Models/{chords}.keras') # this will use all the chords in the current dataset
    
    
def count_classes(df_path):
    df = pd.read_csv(df_path)
    classes = df.iloc[:,0].unique() 
    class_dict = {}
    reverse_dict = {}
    for index, classifier in enumerate(classes):
        class_dict[classifier] = index
        reverse_dict[index] = classifier

    # send the reverse_dict for decoding the outputs later
    chords = ''.join(classes) 
    with open(f'Decoding JSON/{chords}.json', 'w') as f:
        json.dump(reverse_dict, f)

    return class_dict, len(classes), chords


if __name__ == '__main__':
    main()