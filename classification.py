import torch
from torch import nn #modulo che contiene le funzioni per la costruzione dei modelli di rete neurale
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles #per fare degli esempi
import pandas as pd
from sklearn.model_selection import train_test_split


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42) #per rendere i risultati riproducibili
torch.cuda.manual_seed(42) #per rendere i risultati riproducibili anche con la GPU
'''
n_samples=1000
X,y=make_circles(n_samples=n_samples,noise=0.03,random_state=42)
circle=pd.DataFrame({"X1":X[:,0],"X2":X[:,1],"label":y}) #X1 contiene la prima feature di tutte le righe, X2 la seconda e label il target
print(circle.head())

plt.figure(figsize=(10,6))
plt.scatter(x=X[:,0],y=X[:,1],c=y,
            cmap=plt.cm.RdYlBu) #cm è un modulo di matplotlib che contiene delle mappe di colori
plt.legend()
plt.show()

#CONTROLLIAMO LE SHAPE DEI DATI per capire come sono strutturati
print(X.shape) #1000 righe e 2 colonne
print(y.shape) #1000 righe e 1 colonna


#Convertiamo i dati in tensori per poterli utilizzare con PyTorch
X=torch.from_numpy(X).float().to(device)
y=torch.from_numpy(y).float().to(device)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# 2) Costruire il modello per la classificazione binaria
class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1=nn.Linear(in_features=2,out_features=64)
        self.activation_1=nn.ReLU() #funzione di attivazione
        self.layer_2=nn.Linear(in_features=64,out_features=1)
        self.activationLayer=nn.Sigmoid() #funzione di attivazione per la classificazione binaria
        #nn.Sequential è un contenitore di moduli che vengono eseguiti in sequenza. Questo è utile quando si vuole costruire un modello semplice dove non c'è bisogno di definire un metodo forward personalizzato
        #self.sequential=nn.Sequential(
         #   nn.Linear(in_features=2,out_features=5),
          #  nn.Linear(in_features=5,out_features=1),
           # )

    def forward(self, x):
        return self.activationLayer(self.layer_2(self.activation_1(self.layer_1(x))))
        #return self.sequential(x)
        
model_0=CircleModelV0().to(device)
print(model_0.state_dict())  #il peso è del tipo [z, l] perchè sono 2 feature e poi sta il bias che è solo scalare

loss_fn=nn.BCELoss()
#La differenza tra BCEWithLogitsLoss e BCELoss è che la prima è più stabile numericamente perchè combina la sigmoide e la loss binaria in un unico layer
#BCEWithLogitsLoss vuole in input i valori predetti prima della sigmoide, mentre BCELoss vuole i valori predetti dopo la sigmoide (quindi tra 0 e 1)

optimizer=torch.optim.SGD(model_0.parameters(),lr=0.1) #ottimizzatore Adam con learning rate 0.1

def accuracy(y_true, y_pred):
    correct=torch.eq(y_true,y_pred).sum().item() #eq confronta gli elementi di due tensori e restituisce un tensore di booleani, sum() somma i booleani e item() restituisce il valore
    return (correct/len(y_true))*100 #ovvero l'accuratezza in percentuale

# 3) Addestrare il modello
n_epochs = 1000
for epoch in range(n_epochs):
    # Forward pass
    model_0.train()  # mette il modello in modalità training
    y_pred = model_0(X_train).squeeze()  # applica squeeze per ridurre le dimensioni
    optimizer.zero_grad()  # azzera i gradienti
    train_loss = loss_fn(y_pred, y_train)
    
    # Backward pass
    train_loss.backward()
    optimizer.step()

    # TESTING
    model_0.eval()  # mette il modello in modalità valutazione
    with torch.inference_mode():
        test_pred = torch.round(model_0(X_test).squeeze())
        test_loss = loss_fn(test_pred, y_test)
        acc = accuracy(y_test, test_pred)
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Train Loss: {train_loss.item():.5f}, Train Acc: {acc:.2f}, Test Loss: {test_loss.item():.5f}, Test Accuracy: {acc:.2f}%")

model_0.eval()
with torch.inference_mode():
    y_pred = torch.round(model_0(X_test).squeeze())
    acc = accuracy(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}%")
plt.figure(figsize=(10,6))
X_test,y_test=X_test.cpu().detach().numpy(),y_test.cpu().detach().numpy()
plt.scatter(x=X_test[:,0],y=X_test[:,1],c=y_test,cmap=plt.cm.RdYlBu)
plt.legend()
plt.show()
'''
#Costruzione modello per la classificazione multiclasse

from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

n_samples=1000
X,y=make_blobs(n_samples=1000,n_features=2,centers=4,cluster_std=1.5,random_state=42)
plt.figure(figsize=(10,7))
plt.scatter(x=X[:,0],y=X[:,1],c=y,cmap=plt.cm.RdYlBu)
plt.legend()
plt.show()

X=torch.from_numpy(X).float().to(device)
y=torch.from_numpy(y).long().to(device) #in questo caso ci vuole LONG perchè è una classificazione multiclasse
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

class BlobModel(nn.Module):
    def __init__(self,input_features,output_features,hidden_units=8):
        super().__init__()
        self.layer_1=nn.Linear(in_features=input_features,out_features=hidden_units)
        self.activation_1=nn.ReLU()      #actually visualizzando i punti nello spazio non è necessario, è possibile dividere linearlmente
        self.layer_2=nn.Linear(in_features=hidden_units,out_features=hidden_units)
        self.activation_2=nn.ReLU() #actually visualizzando i punti nello spazio non è necessario, è possibile dividere linearlmente
        self.layer_3=nn.Linear(in_features=hidden_units,out_features=output_features)
    def forward(self, x):
        return self.layer_3(self.activation_2(self.layer_2(self.activation_1(self.layer_1(x)))))
    
model_4=BlobModel(input_features=2,output_features=4).to(device)   #2 feature di input, 4 classi di output (i centers)
loss_fn=nn.CrossEntropyLoss() #funzione di loss per la classificazione multiclasse
optimizer=torch.optim.Adam(model_4.parameters(),lr=0.01) #ottimizzatore Adam con learning rate 0.1

epochs=1000
for epoch in range(epochs):
    model_4.train()
    y_logits=model_4(X_train)
    y_pred= torch.softmax(y_logits,dim=1).argmax(dim=1) #softmax(dim=1) perchè le classi si trovano sulle colonne di ogni riga, arg_max(dim=1) perchè si prende la colonna piu grande di ogni riga
    loss_train=loss_fn(y_logits,y_train)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    accuracy_train=accuracy_score(y_train.cpu().detach().numpy(),y_pred.cpu().detach().numpy())


    #testing
    model_4.eval()
    with torch.inference_mode():
        y_logits=model_4(X_test)
        y_pred=torch.softmax(y_logits,dim=1).argmax(dim=1)
        loss_test=loss_fn(y_logits,y_test)
        accuracy_test=accuracy_score(y_test.cpu().detach().numpy(),y_pred.cpu().detach().numpy())
        if epoch%10==0:
            print(f"Epoch: {epoch}, Train Loss: {loss_train.item():.5f}, Train Acc: {accuracy_train:.2f}, Test Loss: {loss_test.item():.5f}, Test Accuracy: {accuracy_test:.2f}")

