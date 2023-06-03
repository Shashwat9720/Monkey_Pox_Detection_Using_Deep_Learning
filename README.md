<p align="center">
Monkey Pox Detection using Deep Learning
</p>
## Motivation

As seen with the peculiar scenario of the covid-19 pandemic an earlier proactive reaction to the spread of  the virus during the infant stages of the pandemic could  have prevented the pandemic at the first place, learning  the lessons from the mistakes and observing the recent  surge in monkeypox cases and adding on to that the lack
of availability of a proper detection methodology for the  masses , deep learning based detection of monkey pox could prevent a future pandemic in the making. This same methodology could be extended to other diseases  in the future which could help in early diagnosis of such  cases and saving a lot of lives.

## Approach 

### System architecture 

•	The model uses EfficientNetB3, a pre-trained deep neural network, to create a custom image classification model for binary classification tasks.

•	The include_top parameter of the EfficientNetB3 model is set to False, allowing for the exclusion of the top layers of the model, as a new output layer will be added.
•	To fine-tune the model, the last 5 layers are trained while keeping the remaining layers frozen to prevent overfitting.

•	The inputs layer is created with a shape of (224, 224, 3), matching the input size of the EfficientNetB3 model.

•	The base_layer is defined as the output of the EfficientNetB3 model when given the inputs layer.

•	To prevent overfitting, a dropout layer with a rate of 0.4 is added to the base_layer.

•	The output of the dropout layer is flattened into a 1D vector using the Flatten () layer.

•	A dense layer with 256 units and ReLU activation is added, followed by a batch normalization layer to enhance training stability.

•	Another dropout layer with a rate of 0.6 is added to the model, followed by another batch normalization layer.

•	A dense layer with 128 units and ReLU activation is then included, followed by a dense output layer with 1 unit and sigmoid activation for binary classification.

•	The Model is then created with the inputs and outputs layers.
•	The model is compiled using the Adam optimizer with a learning rate of 0.001, binary cross-entropy loss, and accuracy metric.

•	Finally, the model is trained 5 times using a for loop, where it is fit to the train_data for 20 epochs and evaluated on the val_data.

•	The trained models are appended to model_list after each iteration, allowing for future evaluation or ensembling of the 5 trained models.

•	The accuracy and loss metrics are used to evaluate the performance of the model.

### Algorithms, Techniques

A custom image classification model for binary classification tasks is created in this work, with the EfficientNetB3 model serving as a base. 
The pre-trained weights of the EfficientNetB3 model are utilized through transfer learning to extract features from input images. To prevent overfitting,
the model is fine-tuned by training only the last 5 layers, while the remaining layers are frozen. The architecture of the model includes an input layer, 
a dropout layer, a flatten layer, two dense layers with batch normalization and dropout layers in between, and an output layer. The Adam optimizer is used
to compile the model with binary cross-entropy loss and accuracy metric.

Bagging: - The model is trained multiple times, specifically five times, in order to improve its accuracy. This is achieved through a for loop, where the model is fit to 
the train data for 20 epochs, and then evaluated on the value data. The trained models are then stored in a list called model list after each iteration. This list can
be utilized to employ bagging ensemble technique to further improve the accuracy of the model.

In conclusion, this code implementation demonstrates the use of transfer learning to create a custom image classification model for binary classification tasks using the EfficientNetB3 model as a base. The fine-tuned model architecture consists of an input layer, a dropout layer, a flatten layer, two dense layers with batch normalization and dropout layers in between, and an output layer. The model is trained using a 
for loop, and the trained models are appended to model_list for future evaluation or ensembling. The performance of the model can be evaluated using accuracy and
loss metrics.



## Tech Stack
1)scikit-learn

2)TensorFlow

3)Numpy

4)Pandas

5)Python


## Result

