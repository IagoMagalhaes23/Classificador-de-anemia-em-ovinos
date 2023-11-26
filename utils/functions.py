'''
    Autores: 
    Data: 09/11/2023
    Descrição:
        - Implementa funções para leitura das imagens
        - Composição do dataset para treino, teste e validação
        - Plotagem dos gráficos de treinamento
        - Plotagem da matriz de confusão
'''

import os
import cv2
import psutil
import itertools
import tracemalloc
import numpy as np
import pandas as pd
from time import time_ns
import matplotlib.pyplot as plt
plt.style.use('default')

from keras.preprocessing import image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc

def readFiles(caminhos):
    '''
        Função para ler todos os arquivos de imagem em uma pasta
        :param caminhos: caminho dos arquivos de imagem
        :return: retorna uma lista com o endereço e nome da imagem e a respectiva classe
    '''
    cont = 0
    data_list = []

    for caminho, _, arquivo in os.walk(caminhos):
        cam = str(caminho.replace("\\", "/"))+"/"
        for file in arquivo:
            # print(file[-5])
            data_list.append([os.path.join(cam, file), file[-5]])

    return data_list

def compose_dataset(df, size, filtro):
    '''
        Função para compor o dataset de treino, teste e validação
        :param df: recebe um dataframe com o endereço da imagem e seu label
        :param size: tamanho da imagem final
        :param filtro: defini qual filtro será aplicado na imagem
        :return: retorna dois np.arrays com a imagem e o label
    '''
    data = []
    labels = []

    for img_path, label in df.values:
        data.append(filtros(img_path, size, filtro))
        if label == 'A':
            labels.append(0)
        elif label == 'B':
            labels.append(1)
        elif label == 'C':
            labels.append(2)
        elif label == 'D':
            labels.append(3)
        elif label == 'E':
            labels.append(4)

    return np.array(data), np.array(labels)

def plot_hist(history):
    '''
        Função para plotar o gráfico de treinamento
        :param hist: recebe o histórico de treinamento da rede com os dados de acurácia e loss
    '''
    plt.figure(figsize=(12,10))
    plt.subplot(2,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylim(0.8,1)
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','valid'], loc='lower right')
    plt.subplot(2,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss Function')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','valid'], loc='upper right')
    plt.ylim([0,1])

def metrics(y_actual, y_pred):
    '''
        Função para plotagem das métricas de avaliação do modelo
        :param y_actual: valor original da classe
        :param y_pred: valor predito pelo modelo
        :return: retorna o valor de acurácia, precisão, sensibilidade, fpr, tpr, roc_auc
    '''
    accuracy = accuracy_score(y_actual, y_pred)
    precisao = precision_score(y_actual, y_pred, average='macro')
    sensibilidade = recall_score(y_actual, y_pred, average='macro')
    fpr, tpr, _ = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    print('------------------------------------')
    print('Acurácia:%.3f' %accuracy)
    print('Precisão:%.3f' %precisao)
    print('Sensibilidade:%.3f' %sensibilidade)
    print('------------------------------------')
    return accuracy, precisao, sensibilidade, fpr, tpr, roc_auc

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    '''
        Função para plotagem da matriz de confusão
        :param :
        :return: 
    '''
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=15)
        plt.yticks(tick_marks, target_names, fontsize=15)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=30)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=30)


    plt.tight_layout()
    plt.ylabel('Classificação correta', fontsize=20)
    plt.xlabel('Predição', fontsize=20)
    plt.show()

def roc_curve():
    '''
        Função para plotagem da curva roc
    '''
    # plt.figure()
    # lw = 2
    # plt.plot(acfpr, actpr, color='darkred',
    #         lw=lw, label='CNN-IF + dropout - ROC curve (area = %0.2f)' % acroc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC curve of test data without diaphragm')
    # plt.legend(loc="lower right")
    # plt.show()
    print('Tu ainda precisa fazer essa função')