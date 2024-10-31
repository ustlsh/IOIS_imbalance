import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from numpy import mean
import os
import csv
#from torch.utils.tensorboard import SummaryWriter

class History():
    def __init__(self, dir):
        self.csv_dir = os.path.join(dir,'metrics_outputs.csv')
        self.pic_dir = os.path.join(dir,'loss-acc.png')
        self.train_acc_pic_dir = os.path.join(dir,'train-perclass-acc.png')
        self.val_acc_pic_dir = os.path.join(dir,'val-perclass-acc.png')
        self.losses_epoch = []
        self.acc_epoch = []
        self.epoch_outputs = [['Epoch', 'Train Loss', 'Train Acc', 'Val Acc', 'Precision', 'Recall', 'F1 Score', 'Confusion','train_0','train_1', 'train_2', 'train_3', 'train_4', 'train_5', 'train_6', 'val_0','val_1', 'val_2', 'val_3', 'val_4', 'val_5', 'val_6']]
        self.temp_data = []
        
    def update(self,data,mode):
        if mode == 'train':
            self.temp_data.append(data)
            self.losses_epoch.append(data)
        elif mode == 'test':
            self.temp_data.extend([data.get('accuracy_top-1'),mean(data.get('precision',0.0)),mean(data.get('recall',0.0)),mean(data.get('f1_score',0.0))])
            self.acc_epoch.append(data.get('accuracy_top-1'))
        
    
    def after_epoch(self,meta):
        '''
        保存每周期的 'Train Loss', 'Val Acc', 'Precision', 'Recall', 'F1 Score'
        '''
        acc_epoch = []
        train_acc_epoch = []
        epoch_outputs = []
        with open(self.csv_dir, 'w', newline='') as f:
            writer          = csv.writer(f)
            for i in range(len(meta['train_info']['train_loss'])):
                temp_data = [i+1, meta['train_info']['train_loss'][i], meta['train_info']['val_acc'][i].get('accuracy_top-1'),mean(meta['train_info']['val_acc'][i].get('precision',0.0)),mean(meta['train_info']['val_acc'][i].get('recall',0.0)),mean(meta['train_info']['val_acc'][i].get('f1_score',0.0)), meta['train_info']['val_acc'][i].get('confusion',0.0), meta['train_perclass_acc']['class0'][i],meta['train_perclass_acc']['class1'][i], meta['train_perclass_acc']['class2'][i], meta['train_perclass_acc']['class3'][i], meta['train_perclass_acc']['class4'][i], meta['train_perclass_acc']['class5'][i], meta['train_perclass_acc']['class6'][i], meta['val_perclass_acc']['class0'][i], meta['val_perclass_acc']['class1'][i], meta['val_perclass_acc']['class2'][i], meta['val_perclass_acc']['class3'][i], meta['val_perclass_acc']['class4'][i], meta['val_perclass_acc']['class5'][i], meta['val_perclass_acc']['class6'][i]]

                acc_epoch.append(meta['train_info']['val_acc'][i].get('accuracy_top-1'))
                train_acc_epoch.append(meta['train_info']['train_acc'][i].get('accuracy_top-1'))
                epoch_outputs.append(temp_data)
            writer.writerows(epoch_outputs)

        '''
        绘制每周期Train Loss以及Validation Accuracy
        '''
        total_epoch = range(1,len(meta['train_info']['train_loss'])+1)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.plot(total_epoch, meta['train_info']['train_loss'], 'red', linewidth = 2, label='Train loss')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Acc')
        ax2.plot(total_epoch, acc_epoch, 'blue', linewidth = 2, label='Val acc')
        
        ax2.plot(total_epoch, train_acc_epoch, 'green', linewidth = 2, label='Train acc')
        
        fig.legend()
        fig.tight_layout()
        plt.savefig(self.pic_dir)
        plt.close("all")

        #绘制每周期Train Accuracy and val Accuracy for each class
        # MEL, NV, BCC, AKIEC, BKL, DF, VASC
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Per Class Acc')
        ax1.plot(total_epoch, meta['train_perclass_acc']['class0'], 'red', linewidth = 2, label='Class MEL')
        ax1.grid(True)
        
        ax1.plot(total_epoch, meta['train_perclass_acc']['class1'], 'blue', linewidth = 2, label='Class NV')
        
        ax1.plot(total_epoch, meta['train_perclass_acc']['class2'], 'green', linewidth = 2, label='Class BCC')

        ax1.plot(total_epoch, meta['train_perclass_acc']['class3'], 'yellow', linewidth = 2, label='Class AKIEC')

        ax1.plot(total_epoch, meta['train_perclass_acc']['class4'], 'black', linewidth = 2, label='Class BKL')

        ax1.plot(total_epoch, meta['train_perclass_acc']['class5'], 'c', linewidth = 2, label='Class DF')

        ax1.plot(total_epoch, meta['train_perclass_acc']['class6'], 'm', linewidth = 2, label='Class VASC')
        
        fig.legend()
        fig.tight_layout()
        plt.savefig(self.train_acc_pic_dir)
        plt.close("all")

        #绘制每周期Train Accuracy and val Accuracy for each class
        # MEL, NV, BCC, AKIEC, BKL, DF, VASC
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Per Class Acc')
        ax1.plot(total_epoch, meta['val_perclass_acc']['class0'], 'red', linewidth = 2, label='Class MEL')
        ax1.grid(True)
        
        ax1.plot(total_epoch, meta['val_perclass_acc']['class1'], 'blue', linewidth = 2, label='Class NV')
        
        ax1.plot(total_epoch, meta['val_perclass_acc']['class2'], 'green', linewidth = 2, label='Class BCC')

        ax1.plot(total_epoch, meta['val_perclass_acc']['class3'], 'yellow', linewidth = 2, label='Class AKIEC')

        ax1.plot(total_epoch, meta['val_perclass_acc']['class4'], 'black', linewidth = 2, label='Class BKL')

        ax1.plot(total_epoch, meta['val_perclass_acc']['class5'], 'c', linewidth = 2, label='Class DF')

        ax1.plot(total_epoch, meta['val_perclass_acc']['class6'], 'm', linewidth = 2, label='Class VASC')
        
        fig.legend()
        fig.tight_layout()
        plt.savefig(self.val_acc_pic_dir)
        plt.close("all")
