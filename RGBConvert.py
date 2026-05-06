import os
import numpy as np
from PIL import Image

def converter_npy_para_png(pasta_origem, pasta_destino):
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)
        print(f"Pasta criada: {pasta_destino}")

    # Lista todos os arquivos na pasta de origem
    arquivos = os.listdir(pasta_origem)
    
    contador = 0
    for arquivo in arquivos:
        if arquivo.endswith(".npy"):
            caminho_npy = os.path.join(pasta_origem, arquivo)
            
            try:
                imagem_array = np.load(caminho_npy)
                
                # 1. CORREÇÃO DE COR (BGR para RGB)
                if len(imagem_array.shape) == 3 and imagem_array.shape[2] == 3:
                    imagem_array = imagem_array[:, :, ::-1] 

                # 2. MELHORIA DE CONTRASTE E BRILHO
                if imagem_array.dtype != np.uint8 or imagem_array.max() > 255.0:
                    p_min, p_max = np.percentile(imagem_array, (2, 98))
                    
                    imagem_array = np.clip(imagem_array, p_min, p_max)
                    
                    # Normaliza de volta para 0 a 255
                    if p_max - p_min != 0:
                        imagem_array = (255 * (imagem_array - p_min) / (p_max - p_min))
                    
                imagem_array = imagem_array.astype(np.uint8)

                imagem = Image.fromarray(imagem_array)
                
                if imagem.mode != 'RGB':
                    imagem = imagem.convert('RGB')
                
                # Define o nome e salva o arquivo de saída
                nome_png = arquivo.replace(".npy", ".png")
                caminho_png = os.path.join(pasta_destino, nome_png)
                
                imagem.save(caminho_png)
                print(f"Convertido: {arquivo} -> {nome_png}")
                contador += 1
                
            except Exception as e:
                print(f"Erro ao converter o arquivo {arquivo}: {e}")

    print(f"\nConversão concluída! {contador} imagens salvas em '{pasta_destino}'.")

pasta_origem = "./images" 

# Diretório onde as imagens serão salvas (pode ser o caminho absoluto '/imagesRGB' 
pasta_destino = "./imagesRGB" 

converter_npy_para_png(pasta_origem, pasta_destino)