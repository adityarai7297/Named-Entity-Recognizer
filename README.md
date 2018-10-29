# Named Entity Recognizer
I have gone with an approach to build an entity recognizer with indian names from scratch using a convolutional neural network.

First i scraped the names (more than 30,000) then converted them into one hot encoded vectors (all names are converted to lowercase)

for example:

rachana

is converted to

[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

a 2d array of dimension 15x26 where each row represents the index of the alphabet used

the word is trimmed or padded to fit the length 15 for a uniform input to the model


Same is done with a corpora of commonly used english words (more than 20,000) 

These are combined and act as my Xtrain values.

A Label array is created which is "1" if the value at that index is an indian name and "0" if otherwise

this acts as my Ytrain values

The process mentioned above is done by running the word_vectorizer.py script

This produces 2 files Xtrain.pkl and Ytrain.pkl which is used for training the CNN.

After this the model_cnn.py script is run which takes the above 2 pickle files as input and trains the convolutional model (you can check the code to get details of the layers used).

After training is complete for about 50 epochs a training accuracy of 98.5% was reached and the model was saved as "trained_model.h5" file.

The predict script takes input from "Input_text.txt" file (this is the file where you will be giving the input) and returns the output on the terminal.

I have added a nice gui feature that highlights the Indian Name words in the output.

I have included all the files though you will only be needing "trained_model.h5" , "predict.py" and "Input_text.txt"

Execution:

I've taken additional efforts to make execution as simple as possible

activate the environment you have been working on
cd to the directory where these files are present
execute word_vectorizer.py to get Xtrain and Ytrain pickle files
execute model.py to train the model and save it in trained_model.py
write your text in Input_text.txt (limited to one text at a time for now)
execute predict.py and observe output on terminal

Libraries to be installed:

Tensorflow
Keras
Pickle
Pandas
NumPy
SKlearn
Nltk

