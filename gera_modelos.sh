echo "----------------------------------------------------------------"
echo "Gerando Modelo 1 (PPO) - Baseline"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/1-ppo_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Gerando Modelo 2 (PPO) - Apressado & Ganancioso"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/2-ppo_model --step-reward=-5 --wall-reward=-10 --hole-reward=-200 --goal-reward=500 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Gerando Modelo 5 (PPO) - Parede de Espinhos"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/5-ppo_model --step-reward=-2 --wall-reward=-20 --hole-reward=-500 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Gerando Modelo 7 (PPO) - Hiperativo"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/7-ppo_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Gerando Modelo 9 (PPO) - Explorador Nato"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/9-ppo_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Gerando Modelo 12 (PPO) - Passeio no Parque"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/12-ppo_model --step-reward=0 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Gerando Modelo 13 (PPO) - Labirinto de Vidro"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/13-ppo_model --step-reward=-1 --wall-reward=-50 --hole-reward=-100 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Gerando Modelo 14 (PPO) - Morte Súbita"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/14-ppo_model --step-reward=-1 --wall-reward=-5 --hole-reward=-1000 --goal-reward=100 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Gerando Modelo 15 (PPO) - Desmotivado"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/15-ppo_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=10 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Gerando Modelo 20 (PPO) - Equilibrado Conservador"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=True --render=False --ia-algorithm=ppo --model-path=./ModelsPPO/20-ppo_model --step-reward=-1 --wall-reward=-5 --hole-reward=-300 --goal-reward=300 --alpha=0.0003 --gamma=0.99 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.005 --decay-step=1000