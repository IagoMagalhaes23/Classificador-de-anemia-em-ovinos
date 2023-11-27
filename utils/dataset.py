'''
    Autores: Iago, Francilândio, Vanessa, Raniery, Sávio, Iális e Fischer
    Data: 09/11/2023
    Descrição:
        - Ler arquivos XML e extrai as posições da ROI
        - Gerar novas imagens com as regiões recortadas
        - Organiza os arquivos em pastas para treino, teste e validação
'''

from bs4 import BeautifulSoup

import cv2
import os

def readXML(file):
    '''
        Função para ler arquivos XML e pegar os valores das posições da mucosa em x, y, w e h.
        :param file: recebe o endereço do arquivo
        :return: retorna uma lista com os de x, y, w e h
    '''
    with open(file, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")
    box = Bs_data.find_all('bndbox')

    results = []

    for name in box:
        results.append(name.text)
    
    return results

def getClass(file):
    '''
        Função para ler arquivos XML e pegar o valor da classe
        :param file: recebe o endereço do arquivo
        :return: retorna uma lista com o valor da classe
    '''
    with open(file, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")
    box = Bs_data.find_all('object')

    results = []

    for name in box:
        results.append(name.text)
    
    string = results[0].split("\n")
    # print(string[1])

    return string[1]

def cropImage(image, positions, caminho, name):
    '''
        Função para recortar a mucosa da imagem original
        :param image: recebe a imagem original do dataset
        :param positions: recebe uma lista com as posições x, y, w e h
        :param caminho: recebe o endereço onde a imagem deve ser salva
        :param name: nome da imagem a ser salva
    '''
    string = positions[0].split("\n")
    x = int(string[1])
    y = int(string[3])
    w = int(string[2])
    h = int(string[4])

    # print(string)
    # print(x, y, w, h)

    roi = image[y:h, x:w]

    try:
        cv2.imwrite(os.path.join(caminho, '{}_{}_{}_{}_{}.png'.format(name, x, y, w, h)), roi)
    except:
        print('Erro ao salvar imagem: {}_{}.png'.format(name, x, y, w, h))