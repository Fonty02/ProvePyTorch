import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics import Accuracy


torch.manual_seed(42)
device="cuda" if torch.cuda.is_available() else "cpu"

dataset=pd.read_csv('playlist_tracks.csv')
dataset=dataset.drop(columns=['author','name'])
y=pd.DataFrame(dataset['playlistName'])
X=dataset.drop(columns=['playlistName'])

NUM_CLASSES=y.nunique().values[0]
NUM_FEATURES=X.shape[1]


mymap = sorted(list(set(y['playlistName'])))
for _, v in enumerate(mymap):
	y.loc[y['playlistName'] == v] = mymap.index(v)

y = y.astype(float)
y=y.to_numpy()
X=X.to_numpy()

X=torch.from_numpy(X).type(torch.float)
y=torch.from_numpy(y).type(torch.LongTensor).squeeze()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=True)

X_test=X_test.to(device)
X_train=X_train.to(device)
y_test=y_test.to(device)
y_train=y_train.to(device)




model_0 = torch.nn.Sequential(
    torch.nn.Linear(in_features=NUM_FEATURES,out_features=1024),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=1024,out_features=2048),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=2048,out_features=NUM_CLASSES)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model_0.parameters(), lr=0.01)
torchmetrics_accuracy = Accuracy(task='multiclass', num_classes=int(NUM_CLASSES)).to(device)

epochs=1300
epochs_list=[]
train_error_list=[]
test_error_list=[]
accuracy_list=[]


for epoch in range(epochs):
	
    model_0.train()

    y_logits=model_0(X_train)

    y_pred_train=torch.softmax(y_logits, dim=1).argmax(dim=1)


    loss_calculated_train=loss_fn(y_logits,y_train)

    optimizer.zero_grad()
    
    loss_calculated_train.backward()

    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
          
          test_logits=model_0(X_test)
          y_pred_test=torch.softmax(test_logits, dim=1).argmax(dim=1)
          loss_calculated_test=loss_fn(test_logits,y_test)
          test_accuracy=torchmetrics_accuracy(y_pred_test,y_test)
          epochs_list.append(epoch)
          train_error_list.append(loss_calculated_train.detach().to("cpu").numpy())
          test_error_list.append(loss_calculated_test.detach().to("cpu").numpy())
          accuracy_list.append(torchmetrics_accuracy(y_pred_test,y_test).detach().to("cpu").numpy())
          if epoch % 100 == 0:
                print(f"Epoch: {epoch} | MAE Train Loss: {loss_calculated_train} | MAE Test Loss: {loss_calculated_test}  | Accuracy: {test_accuracy}")



plt.plot(epochs_list, train_error_list, label="Train loss")
plt.plot(epochs_list, test_error_list, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()
