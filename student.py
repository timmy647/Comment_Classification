#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

"""
Answer to the question:

As for the preprocessing, tokenise is used to split review into words and 
preprocessing is to remove all the meaningless characters except letters and words.
Alos, I turned all the lettter to lower case. I also found there are a lot of "http" 
in the review which is also meaningless, so I remove "http" from the training data.
Stopwords are those words which is meaningless to the content and emotion. Then, 
I converted it into words vector with 300 dimension, since I tested serveral
number of dimension and found out dimension 300 gave a good performance. As for 
postprocessing, I could do the batch normalization. But so far my model is not
overfitting so I chose not to do so.

As for the network, I used LSTM (long short-term memory) followed by a fully connected
layer with dropout 0.8, which is a good value after serveral attempts. I choose to use 
dropout feature to prevent overfiting model when I increase the trainValSplit. The reason
I used LSTM to handle the vaying input size and long-term dependencies of the reviews, 
which ensures extraneous padding from the batches would not affect the result. As for 
activation function, I used logsoftmax for output value and relu for hidden nodes. 
After I have tried different activation function, these gave best performance. This is
because relu work well in deep neural network and logsoftmax work well with cross
entropy loss to update the neural network for classifying multiple classes.

As for the loss function, I used CrossEntropyLoss to calculate the loss for both
ratingOutput and categoryOutput and return the mean of them. Because CrossEntropyLoss
work well with Log_softmax to classify multiple classes. I also have tried to return
different portions of ratingOutput and categoryOutput loss, where I increased the 
portion of category loss in the returned final loss, because normally the categoryOutput
had lower accuracy then ratingOutput. This increased my score weight by around 1
percent.

I increased the trainValSplit from default 0.8 to 0.9 for the purpose of training more
data I tried largely increase the epochs to 20 and 50, however this effectness is very
small. So I just remained the 10 epochs. The error usually become steady between 10
and 20 epochs. I decided to increase the batch size to 128 to prevent local minima.
I decided to use Adam instead of SGD beacuse it can converge faster with the small 
learning rate 0.005.

"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

import re

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

# convert the reviews into words list
def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

# remove the meaningless characters and convert letter into lower case
# only focusing on the words
def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    ans = []
    for word in sample:
        word.lower()
        word = re.sub(r'[^a-z]',"",word)
        word = re.sub(r'http\S+',"",word)
        if word != "":
            ans.append(word)
    
    return ans

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    return batch

# stopwords from NLTK library
stopWords = ["i", "me", "my", "myself", "we", "us","our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
            "she", "her", "hers", "herself", "it", "its", "it's","itself", "they", "them",
            "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the",
            "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by",
            "for", "with", "about", "against", "between", "into", "through", "during",
            "before", "after", "to", "from", "up", "down", "in", "out",
            "on", "off", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "both", "each",
            "more", "most", "other", "some", "such", "own",
            "so", "can", "will", "just",
            "should", "now"]

dimension = 300
wordVectors = GloVe(name='6B', dim=dimension)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################
def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    ratingOutput = torch.argmax(ratingOutput, -1)
    categoryOutput = torch.argmax(categoryOutput, -1)
    return (ratingOutput, categoryOutput)

################################################################################
###################### The following determines the model ######################
################################################################################

# LSTM network followed by a fully connected layer with 200 hidden nodes
# softmax is used for ratingOutput, relu is used for categoryOutput
class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        
        self.lstm_1 = tnn.LSTM(300, 100, num_layers=2, batch_first=True, bidirectional=True, dropout=0.8)
        self.lstm_2 = tnn.LSTM(300, 100, num_layers=2, batch_first=True, bidirectional=True, dropout=0.8)
        self.Linear_1 = tnn.Linear(200, 200)
        self.Linear_2 = tnn.Linear(200, 2)
        self.Linear_3 = tnn.Linear(200, 200)
        self.Linear_4 = tnn.Linear(200, 5)
        self.relu = tnn.ReLU(inplace=False)
        self.log_softmax = tnn.LogSoftmax(dim=1)

    def forward(self, input, length):
        out_1, hn_1 = self.lstm_1(input)
        out_1 = out_1[:,-1,:]
        out_1 = self.Linear_1(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.Linear_2(out_1)
        ratingOutput = self.log_softmax(out_1)

        out_2, hn_2 = self.lstm_2(input)
        out_2 = out_2[:,-1,:]
        out_2 = self.Linear_3(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.Linear_4(out_2)
        categoryOutput = self.log_softmax(out_2)
        return ratingOutput, categoryOutput

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.CrossEntropyLoss(reduction='mean')

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        ratingLoss = self.loss(ratingOutput, ratingTarget)
        categoryLoss = self.loss(categoryOutput, categoryTarget)
        return torch.mean(0.1*ratingLoss + 0.9*categoryLoss)

net = network() 
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.9
batchSize = 128
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.005)
