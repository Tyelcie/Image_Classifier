import torch
from torch import nn
import torch.nn.functional as F
import data_process as dp
from torchvision import models
import pandas as pd
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# define a class for classifier
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

# a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    classifier = Network(checkpoint['input_size'],
                         checkpoint['output_size'],
                         checkpoint['hidden_layers'],
                         drop_p = checkpoint['drop_p'])
    classifier.load_state_dict(checkpoint['model_state_dict'])
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = classifier
    model.class_to_idx = checkpoint['class_to_idx']
    return model

# predict an input image with the model
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    im = dp.process_image(image_path)
    with torch.no_grad():
        probs, idx = torch.exp(model(im.unsqueeze(0))).topk(5)
    probs = probs.detach().numpy().tolist()[0]
    idx = idx.detach().numpy().tolist()[0]
    cat = list(pd.DataFrame(list(model.class_to_idx.keys()))[0].map(cat_to_name))
    labels = list(model.class_to_idx.values())
    classes = []
    for i in idx:
        classes.append(cat[labels.index(i)])
    return probs, classes