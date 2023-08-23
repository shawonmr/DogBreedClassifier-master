import os
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

# number of subprocesses to use for data loading
num_workers = 2
# how many samples per batch to load
batch_size = 10

# convert data to a normalized torch.FloatTensor
data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                     ])

#get the class labels from the dog_image direcotry
train_dir='/data/dog_images/train/'
classes=[]
for dirname in os.listdir(train_dir):
    s = str(dirname)
    if s !='.ipynb_checkpoints':
        classes.append(s)
        print(s)
# load training data set
train_data = datasets.ImageFolder(train_dir,transform=data_transform)
print('Number of train datasets',len(train_data))
# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)

#get test data
test_dir = '/data/dog_images/test/'
#load test data set
test_data = datasets.ImageFolder(test_dir,transform=data_transform)
print('Number of test datasets',len(test_data))
# prepare data loaders
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)

#get validation data
valid_dir = '/data/dog_images/valid/'
#load test data set
valid_data = datasets.ImageFolder(valid_dir,transform=data_transform)
print('Number of valid datasets',len(valid_data))
# prepare data loaders
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)

loader_scratch={}
loader_scratch['train']=train_loader
loader_scratch['test']=test_loader
loader_scratch['valid']=valid_loader

print('No of classes',len(classes))

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128* 14 * 14, 500)
        self.fc2 = nn.Linear(500, len(classes))
    
    
    def forward(self, x):
        ## Define forward behavior
        print('one', x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        print('two', x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        print('three', x.shape)
        x = self.pool(F.relu(self.conv3(x)))
        print('four',x.shape)
        x = self.pool(F.relu(self.conv4(x)))
        print('five', x.shape)
        x = x.view(-1, np.product(x.shape[1:]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
        

#-#-# You so NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()
#print(model_scratch)
# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
    
 import torch.optim as optim
from torch.autograd import Variable

### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()
#criterion_scratch = nn.MultiLabelMarginLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.01)

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        
        model_scratch.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
        #for batch_idx, info in enumerate(train_loader):
            #data, target = info
            target = target[:]-1
            print(data.shape,target.shape)
            print(target)
            #print(batch_idx,data,target)
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data, target
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
             # clear the gradients of all optimized variables
            optimizer_scratch.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model_scratch(data)
            print(output.shape)
            print(target.shape)
            # calculate the batch loss
            loss = criterion_scratch(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer_scratch.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model_scratch.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
        #for batch_idx, info in enumerate(valid_loader):
            #data, target = info
            # move to GPU
            target = target[:]-1
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data, target
            ## update the average validation loss
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model_scratch(data)
            # calculate the batch loss
            loss = criterion_scratch(output, target)
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)
        
        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model_scratch.pt')
            valid_loss_min = valid_loss   
    # return trained model
    return model


# train the model
model_scratch = train(100, loader_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))
