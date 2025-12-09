import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from config import STEP_REWARD, WALL_REWARD, HOLE_REWARD, GOAL_REWARD, set_train, ALPHA, GAMMA, EPSILON, SIMMULATION_NUMBER, ALPHA_DECAY, EPSILON_DECAY, DECAY_STEP, TRAIN, RENDERS
from MazeEnv import MazeEnv
from QLearning import QLearningAgent

global model_path
global train
train = True

def run_ppo(model_path):
    env = MazeEnv()
    try:
        if TRAIN():
            model = PPO(
                "MlpPolicy",       # MlpPolicy para observações vetoriais
                env,
                learning_rate=ALPHA(),      # Taxa de aprendizado tipica para PPO
                n_steps=2048,              # Número de passos de ambiente coletados antes de cada atualização
                batch_size=64,             # Tamanho do mini-lote (pode ser maior que no DQN)
                n_epochs=10,               # Número de vezes que o mini-lote é percorrido
                gamma=GAMMA(),
                gae_lambda=0.95,           # Parâmetro para Advantage Estimation
                clip_range=0.2,            # Parâmetro central do PPO (corte de gradiente)
                verbose=1,
                # NOTA: Parâmetros de exploração (epsilon) não são necessários aqui,
                # pois o PPO trata a exploração pela sua política estocástica e o clip_range
            )

            # 3. Treinamento
            TOTAL_TIMESTEPS = 200000
            print(f"\nIniciando o treinamento por {TOTAL_TIMESTEPS} passos...")
            model.learn(total_timesteps=TOTAL_TIMESTEPS)
            print("Treinamento concluído.")
            model.save(model_path)
            # 4. Avaliação do Agente Treinado
            print("\nAvaliando a política treinada...")
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)

            print(f"\nRecompensa Média em 50 episódios: {mean_reward:.2f}")
            print(f"Desvio Padrão da Recompensa: {std_reward:.2f}")
        else:
            model = PPO.load(model_path)
            state, _ = env.reset(isnumpy = True)
            done = False
            if RENDERS():
                env.render()
            while not done:
                action, _states = model.predict(state, deterministic=False)
                state, reward, done, _, _ = env.step(action, isnumpy = True)
                print(f"Action: {action}, Reward: {reward}")
                if RENDERS():
                    env.render()
    except Exception as ex:
        print(ex)
        raise ex
    finally:
        env.close()

def run_qlearning(model_path):
    env = MazeEnv()
    try:
        if TRAIN():
            agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), EPSILON())
            episodes = SIMMULATION_NUMBER()
            total_reward = 0
            sucess = 0
            for episode in range(1, episodes+1):
                state, _ = env.reset(isnumpy = False)
                done = False
                if RENDERS():
                    env.render()

                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done, _, _ = env.step(action, isnumpy = False)

                    agent.update(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                    if reward == GOAL_REWARD():
                        sucess += 1
                    if RENDERS():
                        env.render()

                # Parameter Decay
                if episode % DECAY_STEP() == 0:
                    agent.epsilon *= EPSILON_DECAY()
                    agent.alpha *= ALPHA_DECAY()
                    print(f"Episode {episode}, Mean Reward: {(total_reward/DECAY_STEP()):.2f}, Success Rate: {(sucess/DECAY_STEP()):.2f}")
                    print("Explore Chance (epsilon): ", agent.epsilon)
                    print("Exploit Chance (1-epsilon): ", 1-agent.epsilon)
                    print("Learning Rate (alpha): ", agent.alpha)
                    total_reward = 0
                    sucess = 0

            # Save the trained agent
            print("Saving agent model")
            agent.save_model(model_path + ".pkl")
            # Test the trained agent
        else:
            state, _ = env.reset(isnumpy = False)
            done = False
            if RENDERS():
                env.render()
            # Load the trained agent
            agent = QLearningAgent(env.action_space, ALPHA(), GAMMA(), 0)
            print("Opening agent model")
            agent.load_model(model_path + '.pkl')

            while not done:
                action = agent.get_action(state)
                state, reward, done, _, _ = env.step(action, isnumpy = False)
                print(f"Action: {action}, Reward: {reward}")
                if RENDERS():
                    env.render()
    except Exception as ex:
        print(ex)
        raise ex
    finally:
        env.close()

def main(args):
    print("Iniciando programa...")
    # Valida argumentos obrigatórios
    if args.model_path is None:
        raise Exception("model-path cannot be none!")
    if args.ia_algorithm is None:
        raise Exception("ia-algorithm cannot be none! Accepted values are: ppo or qlearning")
    
    # Ajusta o parâmetro "TRAIN" se for passado como argumento
    if args.train is not None:
        set_train(args.train.lower() == "true")
        print(TRAIN())

    # Chama a função do algoritmo de IA ou lança exceção
    if args.ia_algorithm == "ppo":
        run_ppo(args.model_path)
    elif args.ia_algorithm == "qlearning":
        run_qlearning(args.model_path)
    else:
        raise Exception("Invalid value for ia-algorithm parameter, accepted values are: ppo or qlearning")

parser=argparse.ArgumentParser()
parser.add_argument("--train", help="Optional, set the TRAIN config value")
parser.add_argument("--ia-algorithm", help="ppo or qlearning, mandatory")
parser.add_argument("--model-path", help="Path to the agent model file, mandatory")

args=parser.parse_args()

main(args)