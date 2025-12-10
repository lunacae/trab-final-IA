#Exemplo gerando modelo qlearning
python3 RL-game.py --train=True --render=False --ia-algorithm=qlearning --model-path=./modelos/qlearning --step-reward=-0.1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

#Exemplo gerando modelo ppo
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./modelos/ppo --step-reward=-0.1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000