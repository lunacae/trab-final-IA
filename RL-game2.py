from MazeEnv import MazeEnv
from config import ALPHA, GAMMA, EPSILON, SIMMULATION_NUMBER, ALPHA_DECAY, EPSILON_DECAY, DECAY_STEP, TRAIN, RENDERS
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

def test_DQN_2():
    env = MazeEnv()
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=ALPHA(),
        buffer_size=50000, 
        learning_starts=DECAY_STEP(),
        batch_size=32,
        gamma=GAMMA(),
        exploration_fraction=EPSILON(), # 20% do treinamento para decaimento de epsilon
        exploration_final_eps=0.05,
        tensorboard_log="./dqn_frozenlake_tensorboard/"
    )

    # --- 3. Treinamento ---
    TIMESTEPS = 20000 # Número de passos (interações com o ambiente)
    print(f"Iniciando treinamento por {TIMESTEPS} passos...")

    # O método .learn() faz todo o trabalho: exploração, buffer, treinamento,
    # atualização da target network.
    model.learn(
        total_timesteps=TIMESTEPS, 
        log_interval=4 # Log a cada 4 chamadas de learn
    )

    print("Treinamento concluído.")

    # Salvar o modelo treinado
    model.save("dqn_frozenlake_model")

    # --- 4. Avaliação (Teste) ---

    # Carregar o modelo (opcional, se você salvou e deseja recarregar)
    # model = DQN.load("dqn_frozenlake_model")

    # Avaliar a política treinado em 100 episódios
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=20
    )

    print("\n--- Resultados da Avaliação ---")
    print(f"Média de Recompensa: {mean_reward:.4f}")
    print(f"Desvio Padrão da Recompensa: {std_reward:.4f}")

    # --- 5. Demonstração de Uso (Visualização) ---

    print("\nDemonstração do agente treinado...")
    for _ in range(5): # Rodar 5 episódios de demonstração
        terminated = False
        truncated = False
        obs, info = env.reset()
        while not terminated and not truncated:
            # Ação baseada na política do modelo
            action, _ = model.predict(obs, deterministic=True)
            
            # Executar a ação
            obs, reward, terminated, truncated, info = env.step(action)
            
            # O .render() renderiza o ambiente para visualização (janela pop-up)
            env.render()
            
            # Pequeno delay para a visualização
            import time; time.sleep(0.1)     
    env.close()

# --- Versão Corrigida da Função test_DQN ---
def test_DQN():
    env = MazeEnv()
    if TRAIN():
      agent = model = DQN(
          "MlpPolicy", 
          env, 
          verbose=1, 
          learning_rate=ALPHA(),
          buffer_size=1000000, 
          learning_starts=DECAY_STEP(),
          batch_size=32,
          gamma=GAMMA(),
          exploration_fraction=0.5, # 20% do treinamento para decaimento de epsilon
          exploration_initial_eps=EPSILON(),
          exploration_final_eps=0.05,
          target_update_interval=1000,
          tensorboard_log="./dqn_maze_tensorboard/"
        )
      model.learn(total_timesteps=10000, log_interval=4)
      model.save("dqn_maze")

      del model # remove to demonstrate saving and loading
      model = DQN.load("dqn_maze")
      obs, info = env.reset()
      if RENDERS():
            env.render()
      while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if RENDERS():
          env.render()  
        if terminated or truncated:
            obs, info = env.reset()
            print(info)

test_DQN()