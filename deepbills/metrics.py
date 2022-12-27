from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import sklearn.metrics
import sklearn.exceptions
from sklearn.utils.multiclass import unique_labels
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Score_Calculation(object):

    def __init__(self, problem_type, file_name):

        self.problem_name = problem_type
        self.file_name = file_name


    def c_matrix(self, labels, preds,  normalize=False,
                                        cmap=plt.cm.Blues,
                                        show = True):

        np.set_printoptions(precision=2)
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

        cm = confusion_matrix(labels, preds)
        classes = unique_labels(labels, preds)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig(self.file_name)
        if(show):
          return plt.show()

        

    def c_reports(self, labels, preds):
        return classification_report(labels, preds)


    def p_score(self, y_true, y_pred):
    
        pmi = precision_score(y_true, y_pred, average = 'micro')
        pma = precision_score(y_true, y_pred, average = 'macro')

        return {'micro_precision': pmi, 
                'macro_precision': pma}


    def r_score(self,y_true, y_pred):
        
        rmi = recall_score(y_true, y_pred, average = 'micro')
        rma = recall_score(y_true, y_pred, average = 'macro')

        return {'micro_recall': rmi, 
                'macro_recall': rma}


    def fsc(self, y_true, y_pred):
        
        fmi = f1_score(y_true, y_pred, average = 'micro')
        fma = f1_score(y_true, y_pred, average = 'macro')

        return {'micro_f1': fmi, 
                'macro_f1': fma}


    def acc_score(self, y_true, y_pred):
        return {'acc': accuracy_score(y_true, y_pred)}


    def compute_metrics(self, pred):

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        acc_dict = {}

        y_true = labels
        y_pred = preds

        acc_dict.update(self.p_score(y_true, y_pred))
        acc_dict.update(self.r_score(y_true, y_pred))
        acc_dict.update(self.fsc(y_true, y_pred))
        acc_dict.update(self.acc_score(y_true, y_pred))

        if self.problem_name  == 'multiclass':
                creport = self.c_reports(y_true, y_pred)
                cmatrix = self.c_matrix(y_true, y_pred)
                acc_dict.update({'c_report':creport, 'c_matrix': cmatrix})
        
        self.save_acc(acc_dict)
        return acc_dict


    def save_acc(self, acc_dict):

      try:
        custom_df = pd.read_csv(f'{self.file_name}/eval_result.csv')
        acc_dict  = {key: [value] for key, value in acc_dict.items()}
        custom_df = pd.concat([pd.DataFrame(acc_dict), custom_df])
        custom_df.to_csv(f'{self.file_name}/eval_result.csv', index=False)
      except Exception as e:

        acc_dict  = {key: [value] for key, value in acc_dict.items()}
        custom_df = pd.DataFrame(acc_dict)
        custom_df.to_csv(f'{self.file_name}/eval_result.csv', index=False)
        print(e)
      
      return 0