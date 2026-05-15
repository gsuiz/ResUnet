import os
import numpy as np
from PIL import Image
import tensorflow as tf

def realizar_predicoes_locais(pasta_origem, pasta_destino, modelo_path):
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
        print(f"Pasta criada: {pasta_destino}")

    print("Carregando o modelo...")
    # Usamos compile=False pois precisamos apenas para inferência, evitando o erro da loss customizada
    modelo = tf.keras.models.load_model(modelo_path, compile=False)
    print("Modelo carregado com sucesso!")

    # Lista todos os arquivos na pasta de origem
    arquivos = os.listdir(pasta_origem)
    
    contador = 0
    for arquivo in arquivos:
        if arquivo.endswith(".npy"):
            caminho_npy = os.path.join(pasta_origem, arquivo)
            
            try:
                # Carrega o array de teste (esperado 256, 256, 4)
                imagem_array = np.load(caminho_npy)
                
                # Adiciona a dimensão do batch para a predição (1, 256, 256, 4)
                input_array = np.expand_dims(imagem_array, axis=0)
                
                # Faz a predição
                predicao = modelo.predict(input_array, verbose=0)
                
                # A saída tem formato (1, 256, 256, 1), removemos as dimensões extras
                mascara = np.squeeze(predicao)
                
                # Aplica um threshold de 0.5 para tornar a imagem nítida (preto e branco absoluto)
                # Valores > 0.5 viram 255 (branco/rio), e <= 0.5 viram 0 (preto/fundo)
                mascara_img = (mascara > 0.5).astype(np.uint8) * 255
                
                imagem = Image.fromarray(mascara_img, mode='L') # 'L' para grayscale (tons de cinza)
                
                # Define o nome e salva o arquivo de saída como PNG
                nome_png = arquivo.replace(".npy", "_pred.png")
                caminho_png = os.path.join(pasta_destino, nome_png)
                
                imagem.save(caminho_png)
                print(f"Predição concluída: {arquivo} -> {nome_png}")
                contador += 1
                
            except Exception as e:
                print(f"Erro ao processar o arquivo {arquivo}: {e}")

    print(f"\nConcluído! {contador} predições salvas em '{pasta_destino}'.")

if __name__ == "__main__":
    # Caminho do modelo treinado
    caminho_modelo = "./artifacts/river256_subset/models/resunet_256_subset_best.keras"
    
    # Pasta onde estão as imagens de teste (.npy)
    pasta_teste = "./test"
    
    # Pasta onde as predições serão salvas
    pasta_predicoes = "./predictions_png" 
    
    realizar_predicoes_locais(pasta_teste, pasta_predicoes, caminho_modelo)
