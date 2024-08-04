# Classificador de anemia em ovinos

## Autores
Iago M. de Mesquita, Francilandio L. Serafim, Vanessa C. do Nascimento, Raniery A. Vasconcelos, Savio A. Gomes, Ialis Cavalcante de P. Júnior, Fischer J. Ferreira e Selmo.

## Introdução
O nordeste brasileiro contem cerca de 60% do rebanho de ovinos do país. Em um setor tão produtivo na região, conseguir identificar e tratar patologias e de fundamental
importância. Neste trabalho e proposto a utilização de redes neurais convolucionais para identificação de anemia em ovinos. Para isso sao analisadas quatro arquiteturas
para a classificação, sendo uma delas desenvolvida especialmente para este problema, a CarNet. Dentre tais modelos, a rede Inception V3, obteve resultados 
de acuracia de 71.4%, precisão de 69.7% e recall de 69.7%.

## Bibliotecas
- OpenCV
- Keras
- TensorFlow
- Matplotlib
- Pandas

## Funcionalidades
- Leitura de imagens
- Classificação de anemia em ovinos através de imagens da mucosa ocular

## Redes analisadas
- CarNet
- EfficientNet B0
- Iception V3
- VGGNet 16

## Arquitetura CarNet
![image](https://github.com/IagoMagalhaes23/Classificador-de-anemia-em-ovinos/assets/65053026/123689b7-c311-42ad-8b54-62539f272470)

## Resultados
Para a analise dos resultados obtidos por cada rede CNN é utilizado valor das métricas de
avaliação. Um conjunto diferente das imagens de treinamento foi utilizado para validar os modelos. Para fins de comparação, cada resultado e apresentado na tabela 1.

![image](https://github.com/IagoMagalhaes23/Classificador-de-anemia-em-ovinos/assets/65053026/13be536e-aefb-4084-a46a-5a5ba7ec2819)

Ao se comparar os resultados obtidos no conjunto de teste, percebe-se que a rede Inception V3 e CarNet performaram melhor no conjunto de dados. Analisando a metrica de acuracia,
é possível notar que ambas chegaram ao valor de 71.4%, porem, a rede Inception V3 tem melhores resultados com as metricas de sensibilidade e precisão, estas com 69.7%.

A CarNet obteve resultados bastante promissores em relação as demais arquiteturas testadas, sendo superior aos modelos da EfficientNetB0 e VGGNet, tanto em valores de acuracia, como os de sensibilidade e precisão. Sua assertividade no conjunto de dados para teste foi de 71.4%. A Figura 6 apresenta a matriz de confusao resultante do modelo ao conjunto de teste.

![image](https://github.com/IagoMagalhaes23/Classificador-de-anemia-em-ovinos/assets/65053026/bb964f8d-7fa2-42b8-ac33-6d1f3d483520)

Para finalizar a analise, os resultados obtidos pelas quatro redes são utilizados para plotar a curva ROC, representada na Figura 7. A partir do grafico resultante é
possível notar que a rede Inception V3 possui um bom comportamento ao conjunto de dados utilizado, apresentando uma area sob a curva de 100%. Já a CarNet possui 75%.

![image](https://github.com/IagoMagalhaes23/Classificador-de-anemia-em-ovinos/assets/65053026/54f69aad-6028-46eb-baa1-988d2c02df39)

## Conclusão
A partir dos resultados apresentados neste trabalho, pode-se concluir que a pesquisa cumpre seu papel, desenvolvendo uma metodologia para a aplicação de CNN’s visando 
classificação de ovinos com anemia. A rede que obteve os melhores resultados foi a Inception V3 com acuracia de 71.4%, sensibilidade de 69.7% e precisão de 69.7%. Além 
disso, mostrou que a rede proposta, CarNet, tambem obteve resultados expressivos para tal tarefa, com resultados de acuracia de 71.4%, sensibilidade de 67.5% e precisão de 
64.2%.

Diante dos resultados obtidos, a pesquisa se mostrou promissora para o auxílio de identificação de anemia em ovinos, classificando em animais com ou sem anemia. Como trabalhos futuros pretende-se desenvolver uma ferramenta acessível em dispositivos móveis que empregue esta metologia de classificação, visando facilitar o uso do modelo por criadores de ovinos.
