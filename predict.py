import argparse
import model_class as mc
import data_process as dp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb
import json
from PIL import Image
from torchvision import transforms, datasets, models

parser = argparse.ArgumentParser()
parser.add_argument('input', type = str, help = 'path of the flower image to be predicted')
parser.add_argument('checkpoint', type = str)
parser.add_argument('--top_k', dest = 'top_k', type = int, default = 5)
parser.add_argument('--category_names', dest = 'category_names', type = str, default = 'cat_to_name.json', help = 'path of a json file that stores the map from category labels to flower names')
parser.add_argument('--gpu', dest = 'gpu', action = 'store_true', default = False)

args = parser.parse_args()

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'
print('The model is running on {}'.format(device)) 

# read json labels
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# load the model
model = mc.load_checkpoint(args.checkpoint)

# define inputs

img_dir = args.input
name = img_dir.split('/')[-2]
dp.imshow(dp.process_image(img_dir));
plt.title(cat_to_name[name]);
plt.axis('off');

# predict
probs, classes = mc.predict(img_dir, model, args.top_k)
print('probs: {}'.format(probs))
print('classes: {}'.format(classes))


#  Display an image along with the top 5 classes
img = Image.open(img_dir).convert('RGB')
img_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
img = img_transform(img)

plt.figure(figsize = [4, 8])
plt.subplot(2, 1, 1)
plt.imshow(img);
plt.title(cat_to_name[name]);
plt.axis('off');

plt.subplot(2, 1, 2)
sb.barplot(x = probs, y = classes, color = 'pink');
plt.xlabel('');
plt.ylabel('');
