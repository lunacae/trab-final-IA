# ...existing code...
echo "----------------------------------------------------------------"
echo "Teste 1: Baseline (Padrão)"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/1-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.1 --gamma=0.9 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.995 --decay-step=100

echo "----------------------------------------------------------------"
echo "Teste 2: Apressado & Ganancioso"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/2-q_learning_model --step-reward=-5 --wall-reward=-10 --hole-reward=-200 --goal-reward=500 --alpha=0.5 --gamma=0.8 --epsilon=1 --alpha-decay=0.995 --epsilon-decay=0.99 --decay-step=50

echo "----------------------------------------------------------------"
echo "Teste 3: Lento & Seguro"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/3-q_learning_model --step-reward=0 --wall-reward=-2 --hole-reward=-50 --goal-reward=50 --alpha=0.05 --gamma=0.99 --epsilon=0.5 --alpha-decay=1 --epsilon-decay=0.999 --decay-step=200

echo "----------------------------------------------------------------"
echo "Teste 4: High Stakes (Apostador)"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/4-q_learning_model --step-reward=-1 --wall-reward=-1 --hole-reward=-20 --goal-reward=1000 --alpha=0.2 --gamma=0.95 --epsilon=1 --alpha-decay=0.995 --epsilon-decay=0.99 --decay-step=100

echo "----------------------------------------------------------------"
echo "Teste 5: Parede de Espinhos"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/5-q_learning_model --step-reward=-2 --wall-reward=-20 --hole-reward=-500 --goal-reward=100 --alpha=0.1 --gamma=0.9 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.995 --decay-step=100

echo "----------------------------------------------------------------"
echo "Teste 6: Míope"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/6-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.3 --gamma=0.5 --epsilon=0.8 --alpha-decay=0.99 --epsilon-decay=0.99 --decay-step=20

echo "----------------------------------------------------------------"
echo "Teste 7: Hiperativo"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/7-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.8 --gamma=0.9 --epsilon=0.2 --alpha-decay=0.95 --epsilon-decay=0.995 --decay-step=1000

echo "----------------------------------------------------------------"
echo "Teste 8: Visionário"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/8-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=200 --alpha=0.1 --gamma=0.999 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.995 --decay-step=50

echo "----------------------------------------------------------------"
echo "Teste 9: Explorador Nato"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/9-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.1 --gamma=0.9 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.9995 --decay-step=500

echo "----------------------------------------------------------------"
echo "Teste 10: Explorador Preguiçoso"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/10-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.1 --gamma=0.9 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.9 --decay-step=10

echo "----------------------------------------------------------------"
echo "Teste 11: Maratona Pesada"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/11-q_learning_model --step-reward=-10 --wall-reward=-10 --hole-reward=-100 --goal-reward=500 --alpha=0.2 --gamma=0.9 --epsilon=0.9 --alpha-decay=0.999 --epsilon-decay=0.995 --decay-step=150

echo "----------------------------------------------------------------"
echo "Teste 12: Passeio no Parque"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/12-q_learning_model --step-reward=0 --wall-reward=-5 --hole-reward=-100 --goal-reward=100 --alpha=0.1 --gamma=0.95 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.995 --decay-step=100

echo "----------------------------------------------------------------"
echo "Teste 13: Labirinto de Vidro"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/13-q_learning_model --step-reward=-1 --wall-reward=-50 --hole-reward=-100 --goal-reward=100 --alpha=0.1 --gamma=0.9 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.995 --decay-step=100

echo "----------------------------------------------------------------"
echo "Teste 14: Morte Súbita"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/14-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-1000 --goal-reward=100 --alpha=0.1 --gamma=0.9 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.995 --decay-step=50

echo "----------------------------------------------------------------"
echo "Teste 15: Desmotivado"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/15-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=10 --alpha=0.1 --gamma=0.9 --epsilon=0.5 --alpha-decay=0.999 --epsilon-decay=0.995 --decay-step=100

echo "----------------------------------------------------------------"
echo "Teste 16: Super Motivado"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/16-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-100 --goal-reward=1000 --alpha=0.1 --gamma=0.9 --epsilon=1 --alpha-decay=0.999 --epsilon-decay=0.995 --decay-step=500

echo "----------------------------------------------------------------"
echo "Teste 17: Caótico"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/17-q_learning_model --step-reward=-5 --wall-reward=-5 --hole-reward=-50 --goal-reward=500 --alpha=0.9 --gamma=0.1 --epsilon=0.7 --alpha-decay=0.9 --epsilon-decay=0.95 --decay-step=75

echo "----------------------------------------------------------------"
echo "Teste 18: Cirurgião"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/18-q_learning_model --step-reward=-2 --wall-reward=-10 --hole-reward=-200 --goal-reward=200 --alpha=0.01 --gamma=0.99 --epsilon=1 --alpha-decay=1 --epsilon-decay=0.999 --decay-step=250

echo "----------------------------------------------------------------"
echo "Teste 19: Equilibrado Agressivo"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/19-q_learning_model --step-reward=-2 --wall-reward=-10 --hole-reward=-150 --goal-reward=150 --alpha=0.2 --gamma=0.85 --epsilon=1 --alpha-decay=0.995 --epsilon-decay=0.99 --decay-step=25

echo "----------------------------------------------------------------"
echo "Teste 20: Equilibrado Conservador"
echo "----------------------------------------------------------------"
python3 RL-game.py --train=False --render=False --ia-algorithm=qlearning --model-path=./Models/20-q_learning_model --step-reward=-1 --wall-reward=-5 --hole-reward=-300 --goal-reward=300 --alpha=0.05 --gamma=0.95 --epsilon=0.9 --alpha-decay=0.999 --epsilon-decay=0.998 --decay-step=125