import torch
torch.set_grad_enabled (False);

class Tanh():  
    def forward(self, *input):
        self.s=input[0]
        return input[0].tanh()
    
    def backward(self, *gradwrtoutput):
        dl_dx=gradwrtoutput[0]
        dl_ds=(4 * (self.s.exp() + self.s.mul(-1).exp()).pow(-2))*dl_dx
        return dl_ds
    
    def param(self):
        return []  
    def reset(self):
        return []
    def update(self,eta):
        return []
    
class Relu():
    def forward(self, *input):
        self.s=input[0]
        return input[0].relu()
    
    def backward(self, *gradwrtoutput):
        dl_dx=gradwrtoutput[0]
        dl_ds=(self.s>0).float()*dl_dx
        return dl_ds
    
    def param(self):
        return []  
    def reset(self):
        return []
    def update(self,eta):
        return []
    
class Sigmoid():
    def forward(self, *input):
        self.s=input[0]
        return input[0].sigmoid()
    
    def backward(self, *gradwrtoutput):
        dl_dx=gradwrtoutput[0]
        dl_ds=(self.s.mul(-1).exp()/(1+self.s.mul(-1).exp()).pow(2))*dl_dx
        return dl_ds
    
    def param(self):
        return []  
    def reset(self):
        return []
    def update(self,eta):
        return []    
    
#Require the input and output dimensions as parameters
class Linear():
       
    def __init__(self,n_in,n_out):      
        self.w = torch.empty(n_out, n_in).normal_()
        self.b =torch.empty(n_out).normal_()
        
        self.dl_dw = torch.empty(self.w.size())
        self.dl_db = torch.empty(self.b.size())
        self.x=torch.empty(n_in)
        
    def reset(self):
        self.dl_dw.zero_()
        self.dl_db.zero_()
    
    def forward(self, *input):
        self.x=input[0]
        return self.w.mv(self.x)+self.b
    
    def backward(self, *gradwrtoutput):
        dl_ds=gradwrtoutput[0]
        dl_dx=self.w.t().mv(dl_ds)
        self.dl_dw.add_(dl_ds.view(-1,1).mm(self.x.view(1,-1)))
        self.dl_db.add_(dl_ds)
        return dl_dx
        
    def param(self):
        return [[self.w,self.dl_dw],[self.b, self.dl_db]]
    
    def update(self,eta):
        self.w -= eta * self.dl_dw
        self.b -= eta * self.dl_db

class LossMSE():
    def loss(self,x,target):
        return (x - target).pow(2).sum()

    def dloss(self,x,target):
        dl_dx=2 * (x - target)
        return dl_dx
    
class LossMAE():
    def loss(self,x,target):
        return abs(x - target).sum()

    def dloss(self,x,target):
        dl_dx=(x-target>0).float()*2-1
        return dl_dx
    
class CrossEntroyLoss():
    def loss(self,x,target):    
        return x.softmax(-1)[target].log().mul(-1)

    def dloss(self,x,target):
        dl_dx=x.softmax(-1)
        dl_dx[target].add_(-1)
        return dl_dx
    
    
class Sequential():
    def __init__(self,loss_method,*modules):
        self.network=modules
        self.loss_method=loss_method
        
    def forward(self,input):
        x=input
        for module in self.network:
            x=module.forward(x)
        return x
    
    def backward(self,output,target):
        x=self.loss_method.dloss(output,target)
        #go through the network backward
        for module in self.network[::-1]:
            x=module.backward(x)
        
    