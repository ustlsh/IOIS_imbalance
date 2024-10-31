import numpy as np

data = np.load('/home/slidm/OCTA/Awesome-Backbones/logs/ResNet/2024-02-22-19-16-17/preds/Epoch119.npz')

y_pred = data['y_pred'] # [1504, 7]
y_true = data['y_true'] # [1504]

y_pred_argmax = np.argmax(y_pred, axis=1)

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

acc = np.sum(y_true == y_pred_argmax)/y_true.shape[0]

mcc = matthews_corrcoef(y_true, y_pred_argmax)
bacc = balanced_accuracy_score(y_true, y_pred_argmax)
precision = precision_score(y_true, y_pred_argmax, average='macro')
macro_f1 = f1_score(y_true, y_pred_argmax, average='macro')
ovr_auc = roc_auc_score(y_true, y_pred, multi_class='ovr')
ovo_auc = roc_auc_score(y_true, y_pred, multi_class='ovo')
cm = confusion_matrix(y_true, y_pred_argmax)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print("ACC:", acc)
print("MCC:", mcc)
print("BACC:", bacc)
print("precision:", precision)
print("macro_f1:", macro_f1)
print("ovo_auc:", ovo_auc)
print("ovr_auc:", ovr_auc)

#print(cm, cm_norm)
