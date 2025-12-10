# Executando o projeto

Passos para execução:

1. Criar o ambiente virtual python;
2. Instalar as dependências usando pip
3. Executar o programa passando os parâmetros corretos

## 1. Criando um ambiente virtual

O primeiro passo é criar um ambiente virtual (venv) com python, para isso siga o passo a passo contido [neste link](https://www.alura.com.br/artigos/ambientes-virtuais-em-python?utm_term=&utm_campaign=topo-aon-search-gg-dsa-artigos_conteudos&utm_source=google&utm_medium=cpc&campaign_id=11384329873_164240702375_703853654617&utm_id=11384329873_164240702375_703853654617&hsa_acc=7964138385&hsa_cam=topo-aon-search-gg-dsa-artigos_conteudos&hsa_grp=164240702375&hsa_ad=703853654617&hsa_src=g&hsa_tgt=aud-396128415587:dsa-2276348409543&hsa_kw=&hsa_mt=&hsa_net=google&hsa_ver=3&gad_source=1&gad_campaignid=11384329873&gbraid=0AAAAADpqZICY_FCQTPCmOc816KvlEzYOV&gclid=CjwKCAiA55rJBhByEiwAFkY1QO3s2RgUOnoGBBkn-aDpTj3SNLWqbIEprCrdP9ZgvdW8H_ObWQtD8hoCpwkQAvD_BwE).
Comandos que serão executados:

```shell
python3 -m venv nome_do_ambiente_virtual
source $nome_do_ambiente_virtual/bin/activate
# Caso use windows o comando está abaixo
# $nome_do_ambiente_virtual\Scripts\Activate
```

## 2. Instalando as dependências com pip

Para executar o projeto é necessário ter o python3 e pip instalados na máquina, você pode fazer a instalação do python3 [aqui](https://www.python.org/downloads/). Para instalar o pip (gerenciador de pacotes do python) siga as instruções [deste site](https://pip.pypa.io/en/stable/installation/).

```shell
pip install -r requirements.txt # instala as dependências
```

## 3. Rodando o programa

O arquivo de entrada é o "RL-game.py", esse script **recebe três parâmetros**:

- train: True ou False, especifica se será executado o script em modo de treino ou não
- ia-algorithm: Obrigatório! Aceita ppo ou qlearning, especifica qual algoritmo será executado
- model-path: Obrigatório! Especifica a path onde está o arquivo do modelo que será carregado (caso não seja treino)
  Abaixo está um exemplo de execução considerando que o ambiente virtual (venv) **já foi criado.**

```shell
# Inicia o script em modo de treino
python3 RL-game.py --train=True --ia-algorithm=ppo --model-path=./modelos/ppo
# Roda o modelo treinado
python3 RL-game.py --train=False --ia-algorithm=ppo --model-path=./modelos/ppo
```

python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./modelos/ppo --step-reward=-0.1 --wall-reward=-5 --role-reward=-100 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000
