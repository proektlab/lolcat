import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from lolcat.nn import LOLCAT

def train_lolcat(model: LOLCAT, train_set: Dataset, test_set: Dataset, cell_type_names: list[str],
                 batch_size=32, weighted=True, lr=1e-2, weight_decay=1e-5, epochs=300, use_early_stopping=True,
                 early_stopping_patience=25, early_stopping_patience_inc=1e-5, set_classifier_bias_from_weights=True):

    model = model.to(device='cuda')
    n_classes = len(cell_type_names)

    # compute weights
    if weighted:
        counts = torch.zeros(n_classes, device='cuda')
        for item in train_set:
            counts[item.y.item()] += 1
        for item in test_set:
            counts[item.y.item()] += 1
        
        class_proportions = (counts / counts.sum()).to(dtype=torch.float32)
        inv_proportions = 1 / class_proportions
        class_weights = inv_proportions / torch.sum(inv_proportions)

        if set_classifier_bias_from_weights:
            with torch.no_grad():
                model.classifier.layers.linear2.bias[:] = torch.log(class_weights)
    else:
        class_weights = torch.full((n_classes,), 1 / len(cell_type_names), dtype=torch.float32)
    print('class weights: ' + ', '.join([f'{name}={wt}' for name, wt in zip(cell_type_names, class_weights)]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    softmax = torch.nn.LogSoftmax(dim=1)
    crit = torch.nn.NLLLoss(weight=class_weights)

    best_loss = np.inf
    patience_timer = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)  # type: ignore
    # try loading whole test set - if needed, this can be broken up into batches as well
    test_loader = DataLoader(test_set, batch_size=len(test_set))  # type: ignore
    test_batch = next(iter(test_loader))

    def compute_loss_and_n_correct(batch):
        logits, _ = model(batch.x, batch.batch)
        log_probs = softmax(logits)
        loss = crit(log_probs, batch.y)

        with torch.no_grad():
            # compute additional things for reporting/early stopping
            pred_classes = torch.argmax(log_probs, 1)
            n_per_class = torch.tensor([sum(batch.y == k) for k in range(n_classes)], device='cuda')
            n_correct_per_class = torch.tensor([sum(pred_classes[batch.y == k] == k) for k in range(n_classes)], device='cuda')

        return loss, n_per_class, n_correct_per_class


    loss_history = {'train_epoch': [], 'train_loss': [], 'train_acc': [], 'train_acc_per_class': [],
                    'test_epoch': [], 'test_loss': [], 'test_acc': [], 'test_acc_per_class': []}

    for i in range(epochs):
        epoch_loss_sum = 0
        n_per_class = torch.zeros(n_classes, dtype=torch.int)
        n_correct_per_class = torch.zeros(n_classes, dtype=torch.int)
        
        for batch in train_loader:
            optimizer.zero_grad()

            loss, this_n, this_n_correct = compute_loss_and_n_correct(batch)
            with torch.no_grad():
                epoch_loss_sum += loss.cpu().item()
                n_per_class += this_n.cpu()
                n_correct_per_class += this_n_correct.cpu()
            
            loss.backward()
            optimizer.step()

        # Report loss/accuracy
        with torch.no_grad():
            epoch_loss = epoch_loss_sum / sum(n_per_class)
            acc_per_class = n_correct_per_class / n_per_class
            acc_weighted = torch.dot(class_weights.cpu(), acc_per_class)

            print(f'Epoch {i:03d}, training: loss={epoch_loss:.6f}, {"weighted " if weighted else ""} acc={acc_weighted:.6f}, ' + 
                  ', '.join([f'{cell_type} acc={acc:.6f}' for cell_type, acc in zip(cell_type_names, acc_per_class)]))

            loss_history['train_epoch'].append(i)
            loss_history['train_loss'].append(epoch_loss)
            loss_history['train_acc'].append(acc_weighted)
            loss_history['train_acc_per_class'].append(acc_per_class)
                
        # early stopping
        stopping = False
        if use_early_stopping:
            if epoch_loss / epochs < (best_loss - early_stopping_patience_inc):
                best_loss = epoch_loss / epochs
                patience_timer = 0
            else:
                patience_timer += 1
                if patience_timer > early_stopping_patience:
                    stopping = True

        if i % 10 == 9 or stopping:
            # do test data validation
            with torch.no_grad():
                test_loss, test_n_per_class, test_n_correct = compute_loss_and_n_correct(test_batch)
                test_acc_per_class = test_n_correct / test_n_per_class
                test_acc_weighted = torch.dot(class_weights, test_acc_per_class)

                print(f'Epoch {i:03d}, TEST: loss={test_loss.cpu():.6f}, {"weighted " if weighted else ""} acc={test_acc_weighted.cpu():.6f}, ' +
                    ', '.join([f'{cell_type} acc={acc:.6f}' for cell_type, acc in zip(cell_type_names, test_acc_per_class.cpu())]))

                loss_history['test_epoch'].append(i)
                loss_history['test_loss'].append(test_loss.cpu())
                loss_history['test_acc'].append(test_acc_weighted.cpu())
                loss_history['test_acc_per_class'].append(test_acc_per_class.cpu())

        if stopping:
            print(f'Stopped at epoch {i}')
            break
    
    return loss_history