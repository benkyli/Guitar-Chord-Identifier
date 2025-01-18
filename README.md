# Guitar Chord Identifier 
#### This program allows you to create a dataset of hand landmarkers to represent guitar chords. You can create your own class identifier using this dataset, which can then be used to identify chord shapes in real time. A webcam is needed to use the dataset creation script and the chord identifier script.

## Creating a dataset 

##### Start by running the dataset creation script:
```python
python createDataset.py
```
##### You will be given the following prompt:
```
Enter the chord you want to sample:
```
###### The names you input are case-sensitive, so use the same capitalization if you want to provide multiple samples of the same chord.

### Collecting samples
##### You will be given 5 seconds before data collection begins. During collection, make the shape of the guitar chord with your fretting hand. The script will collect 400 frames from your webcam. 

### Considerations
##### To prevent overfitting and misidentification, consider the following tips: 
* Show different angles of your hand when fretting the chord. 
* Try to include varying positions for unimportant fingers of the chord. 
    * For example, an open A chord does not use the thumb, so you could try to not include your thumb in the frame, or you could place your thumb in random positions to reduce its reliability as a marker. 
* Only include your fretting hand in the shot. Including your strumming hand can create unreliable datapoints. 

### Dataset description
##### The dataset will be placed in the _Datasets_ folder as a CSV file. The column descriptions are given below:
* Column 1: Chord name
* Column 2: Handedness (0 = left, 1 = right)
* Columns 3-44: Relative X and Y coordinates of hand landmarks (Odd columns = X, Even columns = Y)
###### Note: All coordinate values are based on the landmark's relative distance to the first hand landmark. This is why columns 3 and 4 are always zero, since these are the X and Y coordinates of the first landmark. Rather than relying on pixel distances, a relative distance accounts for differences in hand size and webcam resolution.

## Creating a model
##### Simply run the model creation script:
```
python createModel.py
```

##### The script uses the file _chordDataset.csv_ to create the model. If you wish to use a different dataset, change the name of the dataset path located on line 12 in _createModel.py_. I have provided a dataset, model, and decoding JSON in this repository. My model had about 0.7 accuracy, so feel free to create your own if you'd like.

##### The following section will explain the naming conventions of the models and decoding JSON files.

* <b>Models<b>: After running the model creation script, the model will be placed in the _Models_ folder. The file name of the model will consist of all chord names in the dataset. The specific order of these chords will correspond with the order in which you created the samples. Repeated samples of the same chord will not be added to the name again, since the script only counts unique chord names. Chord names are case-sensitive.
* <b>Decoding JSON<b>: The script will also create a JSON file that corresponds to the newly created model. The TensorFlow model that I used could only accept integers as classifiers, rather than strings. Therefore, I had to convert each class into an integer, which would then be decoded back into its original name during chord identification. 

## Using the chord identifier
##### Input the name of the model you want to use into line 18 of _chordIdentifier.py_. Then run the script:
```
python chordIdentifier.py
```
###### Press ESC to exit the program or by closing the window manually. 

## Limitations and future considerations

* There is no way to clear the dataset automatically. Since it is only a CSV file, you can just CTRL+A then BACKSPACE to clear the file. An additional implementation could be to use an actual database, which would make it easier to store different datasets for different models.
* Since the dataset consists of hand landmarks, this program could technically be used to identify any hand gesture. There is no specific part of the program that actually requires a guitar. A future addition could be to include the guitar neck in the data collection process. By including the guitar neck, you could make chord predictions based on where the fingers are positioned on the frets. This would allow you to identify barre chords other than the F chord. 
* When implementing this program, I thought it would be a good idea to include both hands in the identification process, since both hands would usually be present when playing guitar. However, it later occurred to me that there is no functional purpose for including the strumming hand if we are identifying chord shapes. Future implementations could have the user specify handedness when using the chord identifier. 
* When using the scripts, I often got unexpected warnings in my terminal regarding deprecated package modules. While these warnings did not impede the use of my program, they made it hard to see some of the prompts I included in the terminal. Future implementations would hopefully remove these warnings in a more eloquent fashion. 
