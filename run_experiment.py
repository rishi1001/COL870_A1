
import torch
from data_utils import load_data
import argparse
from model_experiment import Lin,GNN
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="name of the dataset", type=str)
parser.add_argument("--k", help="number of GNN layers",type=int)
parser.add_argument("--model_type", help="type of model",type=str)        # linear or with GNNs

args = parser.parse_args()
dataset =load_data(args.dataset)
print()
print(f'Dataset: {dataset}:')
print(' Number of GNN layers ', args.k)
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
print()
print(data)
print('===========================================================================================================')

# Stats about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')

model_folder = './models/'+args.dataset        # TODO maybe remove this while submitting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.model_type == 'linear':
    model = Lin(hidden_channels=32,num_features=dataset.num_features,num_layers=args.k,num_classes=dataset.num_classes).to(device)   # Note change num_features when needed
else:
    model = GNN(hidden_channels=32,num_features=dataset.num_features,num_layers=args.k,num_classes=dataset.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4,weight_decay=5e-6)
criterion = torch.nn.CrossEntropyLoss()

def getMacroF1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='macro')


f1_train_list = []
f1_val_list = []
loss_train_list = []
loss_val_list = []

def train(epoch):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    if epoch%20 == 0:
        f1_train = getMacroF1(data.y[data.train_mask].cpu().numpy(), out[data.train_mask].argmax(dim=1).cpu().numpy())
        f1_train_list.append(f1_train)
        loss_train_list.append(loss.item())
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

best_val_f1 = -1
def val(epoch):
    model.eval()
    out = model(data.x, data.edge_index)
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    f1_val = getMacroF1(data.y[data.val_mask].cpu().numpy(), out[data.val_mask].argmax(dim=1).cpu().numpy())
    global best_val_f1
    if f1_val > best_val_f1:
        best_val_f1 = f1_val
    if epoch%20 == 0:
        f1_val_list.append(f1_val)
        loss_val_list.append(val_loss.item())
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    val_correct = pred[data.val_mask] == data.y[data.val_mask]  # Check against ground-truth labels.
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())  # Derive ratio of correct predictions.
    return val_loss, val_acc

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    f1_test = getMacroF1(data.y[data.test_mask].cpu().numpy(), out[data.test_mask].argmax(dim=1).cpu().numpy())
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    return f1_test,test_acc

def plot():
    # plot f1 score for train and val every 20 epochs
    plt.clf()
    plt.plot([20*i for i in range(len(f1_train_list))],f1_train_list, label='train')    # TODO maybe just set ticks at 20
    plt.plot([20*i for i in range(len(f1_val_list))],f1_val_list, label='val')
    plt.legend()
    plt.title("Macro F1 vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.draw()
    plt.savefig(f"./topology_generated/F1_{args.model_type}_{args.k}.png")
    plt.close()

    # plot loss for train and val every 20 epochs
    plt.clf()
    plt.plot([20*i for i in range(len(loss_train_list))],loss_train_list, label='train')
    plt.plot([20*i for i in range(len(loss_val_list))],loss_val_list, label='val')
    plt.legend()
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.draw()
    plt.savefig(f"./topology_generated/Loss_{args.model_type}_{args.k}.png")
    plt.close()

num_epochs = 5
best_val_loss = -1
bestModel = None
for epoch in range(num_epochs):
    loss = train(epoch)
    val_loss, val_acc = val(epoch)
    if best_val_loss == -1 or val_loss < best_val_loss:
        best_val_loss = val_loss
        bestModel = model
        torch.save(model.state_dict(), model_folder+f'/{args.model_type}_{args.k}_bestval.pth')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

best_test_f1 = -1
with torch.no_grad():
    model.load_state_dict(torch.load(model_folder+f'/{args.model_type}_{args.k}_bestval.pth'))
    best_test_f1, test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

plot()



