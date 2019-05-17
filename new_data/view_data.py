from utils import *
%matplotlib inline

save_path = '../../data/source_data/mnist_100'

train_imgs_ = load_pkls(save_path, 'train_images')
train_labels_ = load_pkls(save_path, 'train_labels')
square_grid_show_imgs(train_imgs_[:25], mode='L')
print(train_labels_[:25])