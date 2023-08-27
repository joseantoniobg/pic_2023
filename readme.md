<div align="center">

![Logo do Unifeg](unifeg.png)

</div>

# PIC - Programa de Iniciação Científica - Unifeg - 2023

## Regressão Logística com Python

Esse projeto foi desenvolvido durante o desenvolvimento do PIC (Programa de Iniciação Científica) do Unifeg Guaxupé.
O presente repositório contém o código utilizado para o processamento dos dados e o documento elaborado durante o programa, ocorrido entre fevereiro e dezembro de 2023.

## Para poder executar:

- Tenha do Docker instalado e configurado
- Execute docker-compose up -d no seu ambiente
- entre no container usando docker-exec -it rl bash
- no terminal, execute python3 logistic-regression.py
- Você pode testar com outros arquivos CSV, contanto que a última coluna seja seu resultado, e as colunas binárias, como sexo, estejam no fim do arquivo.
Será necessário ajustar a leitura dessas colunasn no momento da normalizaçào dos dados.

Nota: exemplo.csv é um set de dados fictícios com 20 entradas apenas para demonstração.

Se tudo der certo, você deverá ver os seguintes resultados no console:
[[3 1]
 [0 1]]
0.8

2023 - Guaxupé/MG
