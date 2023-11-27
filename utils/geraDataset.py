'''
    Autores: Iago, Francilândio, Vanessa, Raniery, Sávio, Iális e Fischer
    Data: 09/11/2023
    Descrição:
        - Realiza a leitura do dataset original
        - Recorta e extrai a ROI da imagem
        - Redimensiona as imagens
        - Rotula as imagens
'''

import os
import cv2

import dataset

caminhos = 'C:/Users/iagom/OneDrive/Desktop/Mestrado/Projeto ovelhas/Doenca Parasitaria.v1i.yolov5pytorch/Doenca Parasitaria.v1i.voc'

for caminho, _, arquivo in os.walk(caminhos):
    caminho = str(caminho.replace("\\", "/"))
    positions = []
    cont = 0
    if(caminho == 'C:/Users/iagom/OneDrive/Desktop/Mestrado/Projeto ovelhas/Doenca Parasitaria.v1i.yolov5pytorch/Doenca Parasitaria.v1i.voc/valid'):
        for file in arquivo:
            if(file[-3:-1] == 'xm'):
                positions = dataset.readXML(caminho + '/' + file)
                image = cv2.imread(os.path.join(caminho + '/' + file[:-3] + 'jpg'))
                dataset.cropImage(image, positions, 'dataset/test/Mucosa', cont)
                cont += 1
    if(caminho == 'C:/Users/iagom/OneDrive/Desktop/Mestrado/Projeto ovelhas/Doenca Parasitaria.v1i.yolov5pytorch/Doenca Parasitaria.v1i.voc/train'):
        for file in arquivo:
            if(file[-3:-1] == 'xm'):
                positions = dataset.readXML(caminho + '/' + file)
                image = cv2.imread(os.path.join(caminho + '/' + file[:-3] + 'jpg'))
                dataset.cropImage(image, positions, 'dataset/train/Mucosa', cont)
                cont += 1