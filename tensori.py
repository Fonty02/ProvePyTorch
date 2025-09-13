import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device_mine = "cuda" if torch.cuda.is_available() else "cpu"



#TENSORE -> ARRAY MULTIDIMENSIONAL

'''
Un tensore è caratterizzato da tre proprietà fondamentali:
- Numero di dimensioni (rank) (scalari 0 dimensioni, vettori 1, matrici 2, etc...) -> numero delle parentesi quadre
- Forma (shape) -> numero di elementi per dimensione (esempio [5,3] mi dice che ho 5 righe e 3 colonne)
- Tipo di dati (dtype)


Per ottenere il contenuto ssi utilizza il metodo .item()  se scalare, .tolist() se non scalare
'''

#Scalare (0D tensor)

x = torch.tensor(10)
print(x.ndim)
print(x.shape)
print(x.item(), x.tolist(),type(x.item()), type(x.tolist()))

#Vettore (1D tensor)
x= torch.tensor([1,2,3,4,5])
print(x.ndim)
print(x.shape)
print(x.tolist(),type(x.tolist()))


#Matrice (2D tensor)
x= torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
print(x[1])
print(x.ndim)
print(x.shape)
print(x.tolist(),type(x.tolist()))

#TENSORE (3D tensor)
x= torch.tensor([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
print(x[1])
print(x.ndim)
print(x.shape) # 2 matrici 3x3
print(x.tolist(),type(x.tolist()))



#RANDOM TENSOR -> importanti per inizializzare i pesi di una rete neurale

random_tensor = torch.rand(3,3) #numero di dimensioni e elementi per dimensione (in questo caso 2 dimensioni da 3 elementi ciascuno, quindi matrice 3x3)
print(random_tensor)

random_image_tensor=torch.rand(224,224,3) #height, width, channels [R,G,B]   (a volte si usa channels, height, width)


#zeros and ones -> di default il tipo di dato è float32
zeros=torch.zeros(3,3)
ones=torch.ones(3,3)

#Create a range of tensor
range_tensor = torch.arange(0,10,2) #start(incluso), end(escluso), step
print(range_tensor)

#Tensor-like -> se voglio creare un tensore con le stesse dimensioni di un altro tensore
like_tensor = torch.zeros_like(range_tensor)


#TENSOR DATA TYPE

float_32_tensor = torch.tensor([1,2,3,4,5], dtype=torch.float32,
                               device=None,
                               requires_grad=False) # requires_grad=True -> per calcolare il gradiente
print(float_32_tensor)

#Matrix multiplication
matrix1 = torch.tensor([[1,2],[3,4]])
matrix2 = torch.tensor([[5,6],[7,8]])
result = torch.matmul(matrix1,matrix2) # oppure matrix1@matrix2
print(result)

tensor_A = torch.rand(2,3)
tensor_B = torch.rand(2,3)
matMul=torch.matmul(tensor_A,tensor_B.T) # oppure tensor_A@tensor_B.T (trasposto per non dare errori qui)

#operazioni su tensori
print("Tensor A: ", tensor_A)
print("Sum: ", torch.sum(tensor_A))
print("Mean: ", torch.mean(tensor_A)) #richiede un tensore di tipo float
print("Max: ", torch.max(tensor_A).item()) #item() per ottenere il valore
print("Min: ", torch.min(tensor_A).item()) #item() per ottenere il valore
print("Argmax: ", torch.argmax(tensor_A).item()) #indice del valore massimo
print("Argmin: ", torch.argmin(tensor_A).item()) #indice del valore minimo

tensor_A=torch.arange(1.,10.)
print("Tensor A: ", tensor_A)
print("Shape di A: ", tensor_A.shape)   

#RESHAPING -> crea una copia del tensore con una nuova forma
x_reshaped=tensor_A.reshape(3,3) #ovviamente le nuove dimensioni devono avere lo stesso numero di elementi delle dimensioni originali
print("Tensor A reshaped: ", x_reshaped)
print("Shape di A reshaped: ", x_reshaped.shape)


#View -> cambia la forma del tensore ma usa la stessa memoria

x_view=tensor_A.view(3,3)
x_view[0,0]=1000
print("Tensor A view: ", x_view)
print("Shape di A view: ", x_view.shape)
print("Tensor A: ", tensor_A) #anche tensor_A è cambiato


#Stacking -> unisce due tensori lungo una nuova dimensione
print("ORIGINALE: ", x_view)
x_stack=torch.stack([x_view,x_view,x_view,x_view])
print("Stack: ", x_stack)
print("Shape stack: ", x_stack.shape) #default dim=0, aggiunge nella prima dimensione (in questo caso creto 4 volte la stessa matrice)
x_stack=torch.stack([x_view,x_view,x_view,x_view],dim=1)
print("Stack: ", x_stack)
print("Shape stack: ", x_stack.shape) #dim=1, aggiunge nella seconda dimensione (in questo caso creo 3 matrici che contengono 4 volte la riga i-esima
x_stack=torch.stack([x_view,x_view,x_view,x_view],dim=2)
print("Stack: ", x_stack)
print("Shape stack: ", x_stack.shape) #dim=2, aggiunge nella terza dimensione (creo 3 matrici dove ogni riga contiene 4 volte l'elemento i-esimo)


#Squeeze -> rimuove le dimensioni con un solo elemento
print("ORIGINAL: ",tensor_A)
tensor_B = tensor_A.squeeze()
print("Squeeze: ", tensor_B) #non cambia nulla perchè non ci sono dimensioni con un solo elemento



#Unsqueeze -> aggiunge una dimensione con un solo elemento

print("ORIGINAL: ",tensor_A)
tensor_B = tensor_A.unsqueeze(dim=0)
print("Unsqueeze su dim 0: ", tensor_B)
tensor_B = tensor_A.unsqueeze(dim=1)
print("Unsqueeze su dim 1: ", tensor_B)

#tensor_B = tensor_A.unsqueeze(dim=2) 
#print("Unsqueeze su dim 2: ", tensor_B)
#IN QUESTO CASO NON FUNZIONA PERCHE' TENSOR_A HA SOLO 1 DIMENSIONE, QUINDI NON POSSO AGGIUNGERE UNA TERZA DIRETTAMENTE

#Permute -> ritorna una view del tensore con le dimensioni permutate

tensor_A = torch.rand(2,3,5)
print("ORIGINAL: ",tensor_A)
print("Shape: ", tensor_A.shape)    
tensor_B = tensor_A.permute(2,0,1) #nuova asse 0 diventa vecchia asse 2, nuova asse 1 diventa vecchia asse 0, nuova asse 2 diventa vecchia asse 1
print("Permute: ", tensor_B)
print("Shape: ", tensor_B.shape)


#INTERAZIONE CON NUMPY
x=torch.tensor([1,2,3,4,5])
print(x)
y=x.numpy()
print(y)
z=torch.from_numpy(y) #di default il tipo di dato è float64, si può cambiare con il parametro dtype
print(z)

x=x.to(device_mine) #trasferisco il tensore sulla GPU
print(x.device)
x=x.to("cpu") #trasferisco il tensore sulla CPU, utile se devo lavorare con librerie che non supportano la GPU (es. matplotlib,numpy,...)
print(x.device)
#POSSONO INTERAGIRE TRA LORO SE SONO SULLO STESSO DISPOSITIVO