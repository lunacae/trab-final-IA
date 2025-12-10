import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from MazeEnv import MazeEnv
from QLearning import QLearningAgent
from config import Config
global model_path
global train
train = True

def run_ppo(model_path):
    env = MazeEnv(input_config)
    try:
        if input_config.TRAIN:
            model = PPO(
                "MlpPolicy",       # MlpPolicy para observações vetoriais
                env,
                learning_rate=input_config.ALPHA,      # Taxa de aprendizado tipica para PPO
                n_steps=2048,              # Número de passos de ambiente coletados antes de cada atualização
                batch_size=64,             # Tamanho do mini-lote (pode ser maior que no DQN)
                n_epochs=10,               # Número de vezes que o mini-lote é percorrido
                gamma=input_config.GAMMA,
                gae_lambda=0.95,           # Parâmetro para Advantage Estimation
                clip_range=0.2,            # Parâmetro central do PPO (corte de gradiente)
                verbose=1,
                # NOTA: Parâmetros de exploração (epsilon) não são necessários aqui,
                # pois o PPO trata a exploração pela sua política estocástica e o clip_range
            )

            # 3. Treinamento
            TOTAL_TIMESTEPS = input_config.SIMMULATION_NUMBER
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
            total_test_reward = 0
            test_success_count = 0
            n_test_episodes = 1000

            for i in range(n_test_episodes):
                # print(f"Testing Episode: {i+1}")
                state, _ = env.reset(isnumpy = True)
                done = False
                episode_reward = 0
                if input_config.RENDER:
                    env.render()
                while not done:
                    action, _states = model.predict(state, deterministic=False)
                    state, reward, done, _, _ = env.step(action, isnumpy = True)
                    episode_reward += reward
                    # print(f"Action: {action}, Reward: {reward}")
                    if input_config.RENDER:
                        env.render()
                
                total_test_reward += episode_reward
                if reward == input_config.GOAL_REWARD:
                    test_success_count += 1
            
            print(f"\nResults after {n_test_episodes} episodes:")
            print(f"Mean Reward: {total_test_reward / n_test_episodes:.2f}")
            print(f"Success Rate: {test_success_count / n_test_episodes:.2f}")
    except Exception as ex:
        print(ex)
        raise ex
    finally:
        env.close()

def run_qlearning(model_path):
    env = MazeEnv(input_config)
    try:
        if input_config.TRAIN:
            agent = QLearningAgent(env.action_space, input_config.ALPHA, input_config.GAMMA, input_config.EPSILON)
            episodes = input_config.SIMMULATION_NUMBER
            total_reward = 0
            sucess = 0
            for episode in range(1, episodes+1):
                state, _ = env.reset(isnumpy = False)
                done = False
                if input_config.RENDER:
                    env.render()

                while not done:
                    action = agent.get_action(state)
                    next_state, reward, done, _, _ = env.step(action, isnumpy = False)

                    agent.update(state, action, reward, next_state)
                    state = next_state
                    total_reward += reward
                    if reward == input_config.GOAL_REWARD:
                        sucess += 1
                    if input_config.RENDER:
                        env.render()

                # Parameter Decay
                if episode % input_config.DECAY_STEP == 0:
                    agent.epsilon *= input_config.EPSILON_DECAY
                    agent.alpha *= input_config.ALPHA_DECAY
                    print(f"Episode {episode}, Mean Reward: {(total_reward/input_config.DECAY_STEP):.2f}, Success Rate: {(sucess/input_config.DECAY_STEP):.2f}")
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
            # Load the trained agent
            agent = QLearningAgent(env.action_space, input_config.ALPHA, input_config.GAMMA, 0)
            print("Opening agent model")
            agent.load_model(model_path + '.pkl')

            total_test_reward = 0
            test_success_count = 0
            n_test_episodes = 1000

            for i in range(n_test_episodes):
                #print(f"Testing Episode: {i+1}")
                state, _ = env.reset(isnumpy = False)
                done = False
                episode_reward = 0
                if input_config.RENDER:
                    env.render()

                while not done:
                    action = agent.get_action(state)
                    state, reward, done, _, _ = env.step(action, isnumpy = False)
                    episode_reward += reward
                    print(f"Action: {action}, Reward: {reward}")
                    if input_config.RENDER:
                        env.render()
                
                total_test_reward += episode_reward
                if reward == input_config.GOAL_REWARD:
                    test_success_count += 1

            print(f"\nResults after {n_test_episodes} episodes:")
            print(f"Mean Reward: {total_test_reward / n_test_episodes:.2f}")
            print(f"Success Rate: {test_success_count / n_test_episodes:.2f}")
    except Exception as ex:
        print(ex)
        raise ex
    finally:
        env.close()

def initialize_config(args) -> Config:
    config = Config(args)
    return config

def validate_parameters(args):
    # Valida argumentos obrigatórios
    if args.model_path is None:
        raise Exception("model-path cannot be none!")
    if args.ia_algorithm is None:
        raise Exception("ia-algorithm cannot be none! Accepted values are: ppo or qlearning")

def main(args):
    print("Iniciando programa...")
    global input_config
    input_config = initialize_config(args)
    # Chama a função do algoritmo de IA ou lança exceção
    if args.ia_algorithm == "ppo":
        run_ppo(args.model_path)
    elif args.ia_algorithm == "qlearning":
        run_qlearning(args.model_path)
    else:
        raise Exception("Invalid value for ia-algorithm parameter, accepted values are: ppo or qlearning")

parser=argparse.ArgumentParser()
parser.add_argument("--train", help="Optional, set the TRAIN config value")
parser.add_argument("--render", help="Optional, set the RENDER config value")
parser.add_argument("--ia-algorithm", help="ppo or qlearning, mandatory")
parser.add_argument("--model-path", help="Path to the agent model file, mandatory")

parser.add_argument("--step-reward", help="set step-reward")
parser.add_argument("--wall-reward", help="set wall-reward")
parser.add_argument("--hole-reward", help="set role-reward")
parser.add_argument("--goal-reward", help="set goal-reward")

parser.add_argument("--alpha", help="Set alpha")
parser.add_argument("--gamma", help="Set gamma")
parser.add_argument("--epsilon", help="Set epsilon")
parser.add_argument("--alpha-decay", help="Set alpha-decay")
parser.add_argument("--epsilon-decay", help="Set epsilon-decay")
parser.add_argument("--decay-step", help="Set decay-step")

args=parser.parse_args()

main(args)