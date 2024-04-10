import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
#INTRODUZIONE A TENSORI

"""
scalar=torch.tensor(7)
print("SCALARE->",scalar.item()) # .item() SOLO SE CONTINENE UN SOLO VALORE
print("Dim scalare->",scalar.ndim)
print("SHAPE scalare->",scalar.shape)

vettore=torch.tensor([7,7])
print("VETTORE->",vettore)
print("Dim vettore->",vettore.ndim) #la dimensione è il "nuemero di parentesi quadre"
print("SHAPE vettore->",vettore.shape)


matrice=torch.tensor([[7,8],[9,10],[11,12]])
print("MATRICE->",matrice)
print("Dim matrice->",matrice.ndim)
print("SHAPE matrice->",matrice.shape)



TENSOR = torch.tensor([[[1, 2, 3,5],
                        [3, 6, 9,8],
                        [2, 4, 5,10]]])
print("TENSOR->",TENSOR)
print("Dim tensor->",TENSOR.ndim)
print("SHAPE tensor->",TENSOR.shape)



random_tensor = torch.rand(size=(3, 4,1))
print("TENSOR RANDOM->",random_tensor)
print("Dim tensor random->",random_tensor.ndim)
print("SHAPE tensor random->",random_tensor.shape)


#Come numpy puoi creare zeros e ones

zero_to_ten=torch.arange(0,10,0.5)
print("ARANGE->",zero_to_ten)


print(TENSOR.device)




#OPERAZIONI CON TENSORI
tensor1=torch.tensor([[1,2,3]])
print(tensor1+10) # -> fa tutti gli elementi (idem addizione, moltiplicazione e divisione)


print(torch.add(tensor1,10)) #-> funzione che fa la stessa cosa di sopra

tensor2=torch.tensor([[1],[2],[3]])

print(tensor1*tensor2) #-> moltiplicazione elemento per elemento

print(tensor1.shape,tensor2.shape)
print(tensor1@tensor2)



print(tensor2)
print(torch.transpose(tensor2,1,0)) #-> trasposta (richiede le due dimensioni di cui fare la trasposta)
print(tensor2.T)

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [76, 6]], dtype=torch.float32)

torch.manual_seed(42) #-> per avere sempre lo stesso risultato
linear=torch.nn.Linear(in_features=2,out_features=6)
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}\n\n Output dimension: {output.ndim}\n\n")

print(x.shape)
print(x.ndim)
print(x.argmax())


#MANIPOLAZIONE

x=torch.arange(1.,8.)
print(x)
print(x.shape)

x_reshaped=x.reshape(1,7)
print(x_reshaped)
print(x_reshaped.shape)
z=x_reshaped
z[:,1:3]=5
print(z)

print(x)
x_stacked=torch.stack([x,x,x,x],dim=1)
print(x_stacked)
print(device)
x=x.to(device)
print(x.device)"""


from torch import nn #-> modulo per la creazione di reti neurali
import matplotlib.pyplot as plt

weight=0.7
bias=0.3

start=0
end=1
step=0.02
X=torch.arange(start,end,step).unsqueeze(dim=1).to(device) #Unsqueeze aggiunge una dimensione
#suppongiamo vettore partenza [1,2,3,4]
#con dim=0 "aggiungo riga", quindi dal vettore [1,2,3,4] ottengo la matrice [[1,2,3,4]]
#con dim=1 "aggiungo colonna", quindi dal vettore [1,2,3,4] ottengo la matrice [[1],[2],[3],[4]]

y=weight*X+bias #l'equazione della retta


train_split = int(0.8 * len(X)) # 80% per il training, 20% testing
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(len(X_train), len(y_train), len(X_test), len(y_test))


def plot_predictions(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=None):
    #move all tensors to CPU for this funciton
    train_data, train_labels, test_data, test_labels = train_data.cpu(), train_labels.cpu(), test_data.cpu(), test_labels.cpu()
    plt.figure(figsize=(10,7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        predictions=predictions.cpu()
    # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
    plt.legend(prop={"size": 14})
    plt.show()





class LinearRegressionModel(nn.Module): #-> nn.Module è la classe base per tutte le reti neurali
    #costruttore
    def __init__(self):
        super().__init__() #-> chiama il costruttore della classe padre
        self.weights=nn.Parameter(torch.randn(1, #1 peso solo scalare
                                              dtype=torch.float,
                                              requires_grad=True) #-> usa la discesa del gradiente
                                 )
        self.bias=nn.Parameter(torch.randn(1, #1 peso solo scalare
                                              dtype=torch.float,
                                              requires_grad=True) #-> usa la discesa del gradiente
                                 )
        
    #metodo che definisce come la rete deve comportarsi
    def forward(self,X:torch.Tensor) -> torch.Tensor:
        return self.weights*X+self.bias


model_0=LinearRegressionModel().to(device) #-> istanza della classe LinearRegressionModel

with torch.inference_mode(): #bisogna entrare nel contesto
    y_pred=model_0(X_test) #-> calcola la predizione usando il metodo forward


plot_predictions(predictions=y_pred)

loss_fn=nn.L1Loss().to(device) #-> funzione di loss (mean absolute error)
optimizer=torch.optim.SGD(params=model_0.parameters(),lr=0.01)#-> ottimizzatore (stocastic gradient descent)


epochs=1000

train_loss_values=[] #-> lista per salvare i valori della loss di training per ogni epoca
test_loss_values=[] #-> lista per salvare i valori della loss di test per ogni epoca
epoch_counter=[] #-> lista per salvare i valori delle epoche

for epoch in range(epochs):

    ###TRANING

    model_0.train() #-> mette il modello in modalità training
    #effettua il training
    y_pred=model_0(X_train)
    #calcola la loss
    loss=loss_fn(y_pred,y_train)
    #eseguo l'ottimizzazione
    optimizer.zero_grad() #-> azzera i gradienti
    #eseguo il backpropagation
    loss.backward()
    #aggiorno i pesi
    optimizer.step()


    ###TESTING

    model_0.eval() #-> mette il modello in modalità testing

    with torch.inference_mode():            # -> entro in inference per non poter usare optimizer e loss
        test_pred=model_0(X_test)
        test_loss=loss_fn(test_pred,y_test.type(torch.float))

        epoch_counter.append(epoch)
        train_loss_values.append(loss.detach().to("cpu").numpy())
        test_loss_values.append(test_loss.detach().to("cpu").numpy())
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


plt.plot(epoch_counter, train_loss_values, label="Train loss")
plt.plot(epoch_counter, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

# Find our model's learned parameters
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")



model_0.eval()
with torch.inference_mode():
    y_pred=model_0(X_test)

plot_predictions(predictions=y_pred)

