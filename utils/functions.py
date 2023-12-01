'''
    Autores: Iago, Francilândio, Vanessa, Raniery, Sávio, Iális e Fischer
    Data: 09/11/2023
    Descrição:
        - Implementa funções para leitura das imagens
        - Calcula métricas de avalição
        - Plotagem dos gráficos de treinamento
        - Plotagem da matriz de confusão
'''

import os
import cv2
import itertools
import numpy as np
import pandas as pd
from time import time_ns
import matplotlib.pyplot as plt
plt.style.use('default')

from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc

def readFiles(caminhos):
    '''
        Função para ler todos os arquivos de imagem em uma pasta
        :param caminhos: caminho dos arquivos de imagem
        :return: retorna uma lista com o endereço e nome da imagem e a respectiva classe
    '''
    data_list = []

    for caminho, _, arquivo in os.walk(caminhos):
        cam = str(caminho.replace("\\", "/"))+"/"
        for file in arquivo:
            data_list.append([os.path.join(cam, file), file[0]])

    return data_list

def process_data(img_path, size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (size, size))
    return img

def compose_dataset(df, size):
    '''
        Função para compor o dataset de treino, teste e validação
        :param df: recebe um dataframe com o endereço da imagem
        :return: retorna um np.array com a imagem
    '''
    data = []

    for img_path in df.values:
        data.append(process_data(img_path, size))

    return np.array(data)

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
    plt.title('Acurácia do modelo')
    plt.ylabel('acurácia')
    plt.xlabel('épocas')
    plt.legend(['treino','validação'], loc='lower right')
    plt.subplot(2,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Função de perda')
    plt.ylabel('perda')
    plt.xlabel('épocas')
    plt.legend(['treino','validação'], loc='upper right')
    plt.ylim([0,1])

def metrics(y_actual, y_pred):
    '''
        Função para plotagem das métricas de avaliação do modelo
        :param y_actual: valor original da classe
        :param y_pred: valor predito pelo modelo
        :return: retorna o valor de acurácia, precisão, sensibilidade, fpr, tpr, roc_auc
    '''
    acuracia = accuracy_score(y_actual, y_pred)
    precisao = precision_score(y_actual, y_pred, average='macro')
    sensibilidade = recall_score(y_actual, y_pred, average='macro')
    fpr, tpr, _ = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    print('------------------------------------')
    print('Acurácia:%.3f' %acuracia)
    print('Precisão:%.3f' %precisao)
    print('Sensibilidade:%.3f' %sensibilidade)
    print('------------------------------------')
    return acuracia, precisao, sensibilidade, fpr, tpr, roc_auc

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

def iou(caixa1, caixa2):
    '''
        Função para calculo da métrica IoU
        :param caixa1: variavel com o valor de bound box real
        :param caixa2: variavel com o valor de bound box previsto pelo modelo
        :return: retorna o valor de IoU
    '''
    x1_intersecao = max(caixa1[0], caixa2[0])
    y1_intersecao = max(caixa1[1], caixa2[1])
    x2_intersecao = min(caixa1[2], caixa2[2])
    y2_intersecao = min(caixa1[3], caixa2[3])

    area_intersecao = max(0, x2_intersecao - x1_intersecao + 1) * max(0, y2_intersecao - y1_intersecao + 1)

    area_caixa1 = (caixa1[2] - caixa1[0] + 1) * (caixa1[3] - caixa1[1] + 1)
    area_caixa2 = (caixa2[2] - caixa2[0] + 1) * (caixa2[3] - caixa2[1] + 1)

    area_uniao = area_caixa1 + area_caixa2 - area_intersecao

    iou = area_intersecao / area_uniao

    return iou

def roc_curve(fprenb0, tprenb0, roc_aucENB0, fprvgg, tprvgg, roc_aucVGG, fpru, tpru, roc_aucU, fprenm, tprenm, roc_aucENM):
    '''
        Função para plotagem da curva roc
        :param fprenb0:
        :param tprenb0:
        :param roc_aucENB0:
        :param fprvgg:
        :param tprvgg:
        :param roc_aucVGG:
        :param fpru:
        :param tpru:
        :param roc_aucU:
        :param fprenm:
        :param tprenm:
        :param roc_aucENM:
        :return: retorna o gráfico da curva ROC
    '''
    plt.figure()
    lw = 2
    plt.plot(fprenb0, tprenb0, color='darkred', lw=lw, label='EfficientNetB0 - curva ROC (área = %0.2f)' % roc_aucENB0)
    plt.plot(fprvgg, tprvgg, color='darkgreen', lw=lw, label='VGGNet16 - curva ROC (área = %0.2f)' % roc_aucVGG)
    plt.plot(fpru, tpru, color='darkblue', lw=lw, label='UNet++ - curva ROC (área = %0.2f)' % roc_aucU)
    plt.plot(fprenm, tprenm, color='darkorange', lw=lw, label='EfficientNetModificada - curva ROC (área = %0.2f)' % roc_aucENM)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de falso positivo')
    plt.ylabel('Taxa de verdadeiro positivo')
    plt.title('Curva ROC para dados de validação')
    plt.legend(loc="lower right")
    plt.show()

def show_image(image_path):
    '''
        Função para plotar imagem
        :param image_path: recebe o enderenço de uma imagem
    '''
    image = cv2.imread(image_path)
    plt.imshow(image)

def make_predictions(image_path, model):
    '''
        Função para realizar predições com o modelo treinado
        :param image_path: recebe o enderenço de uma imagem
        :param model: recebe o modelo treinado
        :return: retorna as predições realizada pelo modelo
    '''
    show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds