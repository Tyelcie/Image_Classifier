import argparse
import model_class as mc
import data_process as dp

import torch
from torch import nn
from torch import optim
from torchvision import transforms, datasets, models

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type = str, help = 'must end with a backslash, eg. flowers/')
parser.add_argument('--save_dir', dest = 'save_dir', type = str, default = 'checkpoint.pth')
parser.add_argument('--arch', dest = 'arch', type = str, default = 'vgg16', help = 'currently only vgg16 and vgg13 are available')
parser.add_argument('--learnning_rate', dest = 'learning_rate', type = float, default = 0.001)
parser.add_argument('--hidden_units', dest = 'hidden_units', type = int, default = [4096, 1024], nargs = '+', help = 'if more than one layers please type --hidden_units h1 h2')
parser.add_argument('--epochs', dest = 'epochs', type = int, default = 5)
parser.add_argument('--gpu', dest = 'gpu', action = 'store_true', default = False)

args = parser.parse_args()

# Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(args.data_dir + x, transform = dp.data_transforms[x]) for x in dp.data_transforms.keys()}
# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle = True) for x in dp.data_transforms.keys()}

# pretrained model
if args.arch == 'vgg16':
    model = models.vgg16(pretrained = True)
elif args.arch == 'vgg13':
    model = models.vgg13(pretrained = True)

# froze parameters
for param in model.parameters():
    param.requires_grad = False

# build the classifier
drop_p = 0.1
classifier = mc.Network(25088, 102, args.hidden_units, drop_p = drop_p)
# replace the classifier
model.classifier = classifier

# train the model
epochs = args.epochs
step = 0
print_every = 30

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)

args.gpu = torch.cuda.is_available()
if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'
print('The model will be trained on {}'.format(device)) 

model.to(device)
for e in range(epochs):
    running_loss = 0
    for i, (inputs, labels) in enumerate(dataloaders['train']):
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if step % print_every == 0:
            model.eval()
            accuracy = 0
            valid_loss = 0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(dataloaders['valid']):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model.forward(inputs)
                    valid_loss += criterion(outputs, labels).item()
                    ps = torch.exp(outputs).data
                    equality = (labels.data == ps.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(dataloaders['valid'])),
                  "Valid Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))
            running_loss = 0
            model.train()

# Do validation on the test set    
correct = 0
total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(dataloaders['test']):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# save checkpoint
checkpoint = {
    'input_size': 25088,
    'output_size': 102,
    'epochs': epochs,
    'drop_p': drop_p,
    'hidden_layers': [e.out_features for e in model.classifier.hidden_layers],
    'model_state_dict': model.classifier.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx}
torch.save(checkpoint, args.save_dir)
print('The model checkpoint is saved in {}'.format(args.save_dir))