# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:25:56 2020

@author: junio
"""

# função desenvolvida para entender como usar o Crop.
soma = 0
divisao = 0
multiplicacao = 0
subtracao = 0

# função que recebe um numero como parametro , calcula as operações por padrão
# Armazena nas respectivas variaveis de op, retorna todas ao mesmo tempo.
def calculadora_ambulante(numero):
    # comandos
    soma = numero+numero
    divisao= numero/numero
    multiplicacao = numero*numero
    subtracao = numero-numero
    return soma, divisao, multiplicacao, subtracao

# declarar um numero 
numero = 5

# capturar as variaveis uma a uma.
# chamando função e passando o parametro
soma, divisao, multiplicacao, subtracao = calculadora_ambulante(numero)

#mostrando variaveis
print(soma)
print(divisao)
print(multiplicacao)
print(subtracao)