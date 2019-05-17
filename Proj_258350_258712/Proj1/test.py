import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import dlc_practical_prologue as prologue
import random
from torch import optim

#Configure the GPU usage
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('A GPU is available and will be used')
    
else:
    device = torch.device('cpu')
    #avoid to always have the same results when using CPU
    torch.manual_seed(random.randint(0,1000))
    print('No GPU available, CPU will be used')
    
#generate train and test data
[train_input,train_target,train_classes,
 test_input,test_target,test_classes]=prologue.generate_pair_sets(1000)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

train_input, train_target,train_classes = train_input.to(device), train_target.to(device),train_classes.to(device)
test_input, test_target,test_classes = test_input.to(device), test_target.to(device),test_classes.to(device)

#Simple architecture
#1 conv layer
class ConvNet1(nn.Module):
    def __init__(self):
        super(ConvNet1, self).__init__()
        nb_hidden=130
        self.conv1 = nn.Conv2d(2, 32, kernel_size=7)
        self.fc1 = nn.Linear(16* 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x=x.view(-1, 16*32)

        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
#2 conv layers
class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        nb_hidden=90
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(16* 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x=x.view(-1, 16*32)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x 
#3 conv layers
class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        nb_hidden=100
        self.conv1 = nn.Conv2d(2, 64, kernel_size=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2)
        self.fc1 = nn.Linear(4* 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.relu(self.conv3(x))
        x=x.view(-1, 4*32)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x   
    
# With weight sharing
#1 conv layer
class ConvNet1_ws(nn.Module):
    def __init__(self):
        super(ConvNet1_ws, self).__init__()
        nb_hidden=70
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7)
        self.fc1 = nn.Linear(32* 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x1=torch.zeros(mini_batch_size,1,14,14)   
        x2=torch.zeros(mini_batch_size,1,14,14) 
        
        #take the two digits apart
        x1[:,0,:,:]=x[:,0,:,:]
        x2[:,0,:,:]=x[:,1,:,:]
        x1 = F.max_pool2d(F.relu(self.conv1(x1)), kernel_size=2)
        x2 = F.max_pool2d(F.relu(F.conv2d(x2,self.conv1.weight)), kernel_size=2)
        x1=x1.view(-1, 16*32)
        x2=x2.view(-1, 16*32)
        x=torch.cat((x1,x2),1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
#2 conv layers    
class ConvNet2_ws(nn.Module):
    def __init__(self):
        super(ConvNet2_ws, self).__init__()
        nb_hidden=50
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(32* 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x1=torch.zeros(mini_batch_size,1,14,14)   
        x2=torch.zeros(mini_batch_size,1,14,14) 
        
        x1[:,0,:,:]=x[:,0,:,:]
        x2[:,0,:,:]=x[:,1,:,:]
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(F.conv2d(x2,self.conv1.weight))
        x1 = F.max_pool2d(F.relu(self.conv2(x1)), kernel_size=2)
        x2 = F.max_pool2d(F.relu(F.conv2d(x2,self.conv2.weight)), kernel_size=2)
        
        x1=x1.view(-1, 16*32)
        x2=x2.view(-1, 16*32)
        x=torch.cat((x1,x2),1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x  

 #3 conv layers   
class ConvNet3_ws(nn.Module):
    def __init__(self):
        super(ConvNet3_ws, self).__init__()
        nb_hidden=150
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2)
        
        self.fc1 = nn.Linear(8* 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x1=torch.zeros(mini_batch_size,1,14,14)   
        x2=torch.zeros(mini_batch_size,1,14,14) 
        
        x1[:,0,:,:]=x[:,0,:,:]
        x2[:,0,:,:]=x[:,1,:,:]
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(F.conv2d(x2,self.conv1.weight))
        
        x1 = F.max_pool2d(F.relu(self.conv2(x1)), kernel_size=2)
        x2 = F.max_pool2d(F.relu(F.conv2d(x2,self.conv2.weight)), kernel_size=2)
        
        x1 = F.relu(self.conv3(x1))
        x2 = F.relu(F.conv2d(x2,self.conv3.weight))
        x1=x1.view(-1, 4*32)
        x2=x2.view(-1, 4*32)
        x=torch.cat((x1,x2),1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x    
    
# with auxiliary losses
#2 conv layers
class ConvNet2_al(nn.Module):
    def __init__(self):
        super(ConvNet2_al, self).__init__()
        nb_hidden=80
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=9)
        
        self.lin_aux1 = nn.Linear(4* 32, 100)
        self.lin_aux2 = nn.Linear(100, 10)
                
        self.fc1 = nn.Linear(8* 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x1=torch.zeros(mini_batch_size,1,14,14)   
        x2=torch.zeros(mini_batch_size,1,14,14) 
        
        x1[:,0,:,:]=x[:,0,:,:]
        x2[:,0,:,:]=x[:,1,:,:]
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(F.conv2d(x2,self.conv1.weight))
        
        x1 = F.max_pool2d(F.relu(self.conv2(x1)), kernel_size=2)
        x2 = F.max_pool2d(F.relu(F.conv2d(x2,self.conv2.weight)), kernel_size=2)
        
        x1=x1.view(-1, 4*32)
        x2=x2.view(-1, 4*32)
        
        x1_aux=torch.sigmoid(self.lin_aux1(x1))
        x1_aux=self.lin_aux2(x1_aux)
        
        x2_aux=torch.sigmoid(self.lin_aux1(x2))
        x2_aux=self.lin_aux2(x2_aux)
        
        x = torch.cat((x1,x2),1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x,x1_aux,x2_aux
#3 conv layers
class ConvNet3_al(nn.Module):
    def __init__(self):
        super(ConvNet3_al, self).__init__()
        nb_hidden=120
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=9)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        
        self.lin_aux1 = nn.Linear(4* 32, 100)
        self.lin_aux2 = nn.Linear(100, 10)
                
        self.fc1 = nn.Linear(8* 32, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2)

    def forward(self, x):
        x1=torch.zeros(mini_batch_size,1,14,14)   
        x2=torch.zeros(mini_batch_size,1,14,14) 
        
        x1[:,0,:,:]=x[:,0,:,:]
        x2[:,0,:,:]=x[:,1,:,:]
        x1 = F.relu(self.conv1(x1))
        x2 = F.relu(F.conv2d(x2,self.conv1.weight))
        
        x1 = F.relu(self.conv2(x1))
        x2 = F.relu(F.conv2d(x2,self.conv2.weight))
        
        x1 = F.relu(self.conv3(x1))
        x2 = F.relu(F.conv2d(x2,self.conv3.weight))
        x1=x1.view(-1, 4*32)
        x2=x2.view(-1, 4*32)
        
        x1_aux=torch.sigmoid(self.lin_aux1(x1))
        x1_aux=self.lin_aux2(x1_aux)
        
        x2_aux=torch.sigmoid(self.lin_aux1(x2))
        x2_aux=self.lin_aux2(x2_aux)
        
        x = torch.cat((x1,x2),1)
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x,x1_aux,x2_aux
    
#Training and error computation functions
#For simple or weight sharing architecture
def train_model(model, train_input, train_target, mini_batch_size,nb_epochs):
    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to(device)

    optimizer = optim.SGD(model.parameters(),lr=1e-2)
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            model.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))   
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))           
            loss.backward()
            optimizer.step()

def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if target.data[b + k]!= predicted_classes[k]:
                nb_errors = nb_errors + 1
    return nb_errors

#For auxiliary losses architectures

def train_model_al(model, train_input, train_target, mini_batch_size,nb_epochs):
    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to(device)

    optimizer = optim.SGD(model.parameters(),lr=1e-2)
    
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            model.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size)) 
            #loss on the final result
            loss = criterion(output[0], train_target.narrow(0, b, mini_batch_size))
            #loss on the two digits
            loss1=criterion(output[1], train_classes[:,0].narrow(0, b, mini_batch_size))
            loss2=criterion(output[2], train_classes[:,1].narrow(0, b, mini_batch_size))
            #we give more importance to the final result
            loss_tot=4*loss+loss1+loss2
            loss_tot.backward()
            optimizer.step()            
        
def compute_nb_errors_al(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output[0].data, 1)
        for k in range(mini_batch_size):
            if target.data[b + k]!= predicted_classes[k]:
                nb_errors = nb_errors + 1

    return nb_errors

##Training and testing
mini_batch_size=50
nb_epoch=25
nb_tests=1

print("Training on 1000 samples, during {:d} epochs using 3 convolutional layers with ~70'000 parameters:".format(nb_epoch) )
#test simple
for i in range(nb_tests):
    model=ConvNet3()
    model.to(device)
    train_model(model, train_input, train_target.to(device),mini_batch_size,nb_epoch)
    nb_test_errors=compute_nb_errors(model, test_input, test_target,mini_batch_size)
    print('Simple architecture: {:0.2f}% of errors '.format((100 * nb_test_errors) / test_input.size(0)))
#test weight sharing
for i in range(nb_tests):
    model=ConvNet3_ws()
    model.to(device)
    train_model(model, train_input, train_target.to(device),mini_batch_size,nb_epoch)
    nb_test_errors=compute_nb_errors(model, test_input, test_target,mini_batch_size)
    print('With weight sharing: {:0.2f}% of errors '.format((100 * nb_test_errors) / test_input.size(0)))
#test auxiliary losses
for i in range(nb_tests):
    model=ConvNet3_al()
    model.to(device)
    train_model_al(model, train_input, train_target.to(device),mini_batch_size,nb_epoch)
    nb_test_errors=compute_nb_errors_al(model, test_input, test_target,mini_batch_size)
    print('With Weight sharing and auxiliary losses: {:0.2f}% of errors '.format((100 * nb_test_errors) / test_input.size(0)))
       
