import proj2_framework as fw
import torch
import math

def generate_disc_set(nb):
    input = torch.Tensor(nb, 2).uniform_(0, 1)
    target = input.sub(0.5).pow(2).sum(1).sub(1/ (2*math.pi)).sign().sub(1).div(-2).long()
    return input, target

#generate train and test dataset and normalize them
train_input,train_target=generate_disc_set(1000)
test_input,test_target=generate_disc_set(1000)

mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

#choose the criterion (LossMSE or CrossEntroyLoss)
loss_method=fw.LossMSE()
#declare sequential network
net=fw.Sequential(loss_method,fw.Linear(2,25),fw.Tanh(),fw.Linear(25,1),fw.Sigmoid())

nb_epochs=100
#choose the step of the gradient descent
eta=1e-3
for i in range(nb_epochs):
    acc_loss = 0
    nb_train_errors = 0
    nb_test_errors = 0
    #Train the network
    for module in net.network:
        #reset the cumulative gradients of all modules
        module.reset()
        
    for j in range(train_target.shape[0]):  
        output=net.forward(train_input[j,:])
        nb_train_errors+=int(bool(output>0.5)!=bool(train_target[j]))
        acc_loss += loss_method.loss(output,train_target[j]) 
        net.backward(output,train_target[j])
    print('{:d} acc_train_loss {:.02f} '.format(i,acc_loss))    
    for module in net.network:
        #Make a step of the gradient descent
        module.update(eta)
    
    #Test the network
    for j in range(test_target.shape[0]):  
        output=net.forward(test_input[j,:])
        nb_test_errors+=int(bool(output>0.5)!=bool(test_target[j]))   

print('train_error {:.02f}% test_error {:.02f}%'.format((100 * nb_train_errors) / train_input.size(0),
                                                         (100 * nb_test_errors) / test_input.size(0)))