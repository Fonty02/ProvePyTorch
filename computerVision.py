import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm


def print_train_time(start,end):
    total_time=end-start
    print(f"Training time: {total_time:.3f} seconds")
    return total_time


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    train_loss= 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    print(f"Train loss: {train_loss:.5f}\n")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module):
    test_loss= 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)

        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} \n")



def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#download del dataset
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=False, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=False,
    transform=ToTensor()
)



image, label = train_data[0]

# in genere però si usa NHWC (numero di immagini, altezza, larghezza, canali)

class_names=train_data.classes
print("NOMI DELLE CLASSI->" ,class_names)

print(image.shape)
print(image.squeeze().shape) # rimuove la dimensione del canale colore


fig=plt.figure(figsize=(9,9))
rows,cols=4,4
for i in range(1, rows*cols+1):
    random_idx=torch.randint(0, len(train_data), size=[1]).item()
    img,label=train_data[random_idx]
    fig.add_subplot(rows,cols,i)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(class_names[label])
    plt.axis(False)
plt.show()


BATCH_SIZE=32

train_data_loader=DataLoader(train_data,         #il dataset da cui caricare i dati 
                             batch_size=BATCH_SIZE, #la dimensione del batch
                             shuffle=True)         #mescola i dati ad ogni epoca
#dataLoader dunque prende il dataset e lo divide in batch di dimensione BATCH_SIZE e diventa un iterabile (iterabile=si può fare un ciclo for)

test_data_loader=DataLoader(test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False) #non serve mescolare i dati di test


#UN PO DI STATISTICHE

print("Per il training ci sono ",len(train_data_loader), " batch sono da ",BATCH_SIZE)
print("Per il test ci sono ",len(test_data_loader), " batch sono da ",BATCH_SIZE)


###COSTRUZIONE DEL MODELLO
train_features_batch, train_labels_batch=next(iter(train_data_loader))



class FashionMNISTModel(torch.nn.Module):
    def __init__(self,intput_shape,hiden_units,output_shape):
        super().__init__()
        self.layer_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(intput_shape,hiden_units),
            nn.Linear(hiden_units,output_shape)
        )

    def forward(self,x):
        return self.layer_stack(x)
    

model_0=FashionMNISTModel(intput_shape=784,hiden_units=10,output_shape=len(class_names))
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model_0.parameters(),lr=0.1)


class FashionMNISTModelV1(nn.Module):
    def __init__(self,input_shape,hidden_units,output_shape):
        super().__init__()
        self.layer_stack=nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape,hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units,output_shape),
            nn.ReLU()
        )
    
    def forward(self,x):
        return self.layer_stack(x)
    

model_1=FashionMNISTModelV1(input_shape=784,hidden_units=10,output_shape=len(class_names))
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model_1.parameters(),lr=0.1)

start_time=timer()

epochs=3

for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_step(model_1,train_data_loader,loss_fn,optimizer)
    test_step(test_data_loader,model_1,loss_fn)

end_time=timer()
print_train_time(start_time,end_time)



##CONVOULUTIONAL NEURAL NETWORK

class FashionMNISTModelV2(nn.Module):

    def __init__(self,input_shape,hidden_units,output_shape):
        super().__init__()
        self.block_1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.block_2=nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*7*7, #7*7 è la dimensione dell'immagine dopo averla passata attraverso i blocchi convoluzionali. Lo si capisce
                      output_shape),
        )

    def forward(self,x):
            x=self.block_1(x)
            x=self.block_2(x)
            return self.classifier(x)
        
model_2=FashionMNISTModelV2(input_shape=1,hidden_units=10,output_shape=len(class_names))

#input_shape=1 perchè le immagini sono in scala di grigi
loss_fn=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model_2.parameters(),lr=0.1)

start_time=timer()
epochs=3
for epoch in tqdm(range(epochs)):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_step(model_2,train_data_loader,loss_fn,optimizer)
    test_step(test_data_loader,model_2,loss_fn)

end_time=timer()
print_train_time(start_time,end_time)



           