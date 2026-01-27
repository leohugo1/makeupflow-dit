    💄 MakeupFlow-DiT: Latent Makeup Transfer with Flow Matching
Este repositório contém a implementação oficial do MakeupFlow-DiT, uma arquitetura de transferência de maquiagem baseada em Diffusion Transformers (DiT) e treinada via Flow Matching (I2I). O sistema utiliza um espaço latente comprimido para transferir estilos de maquiagem de uma imagem de referência para um rosto alvo em alta resolução (512x512).

 ---
    🚧 Status do Projeto: Treinamento Multietapa

O projeto utiliza um script unificado (train.py) para gerenciar as três fases críticas de desenvolvimento:

**Fase 1: Treinamento do VAE** – Otimização da reconstrução facial 512px usando CelebA-HQ com Perceptual Loss (LPIPS).

**Fase 2: Treinamento do StyleVit** – Extração de embeddings de maquiagem via Vision Transformer e Triplet Margin Loss no dataset FFHQ-Makeup.

**Fase 3: Treinamento do DiT (Flow Matching)** – Aprendizado da trajetória linear entre o rosto limpo e maquiado.

 ---
    🚀 Arquitetura Técnica
* **Diffusion Transformer (DiT)**: Implementado com blocos de Cross-Attention e ativações SwiGLU (SwiGLUMP) para maior eficiência e estabilidade no aprendizado de fluxos.

* **StyleViT**: Um Vision Transformer que utiliza PEG (Positional Encoding Generator) para capturar texturas de maquiagem.

* **VAE (Variational Autoencoder)**: Estrutura robusta com ResnetBlocks e AttnBlocks. Utiliza um fator de escala latente de 0.18215 para compatibilidade com fluxos de difusão modernos.

---
    🛠️ Tecnologias e Bibliotecas

A stack tecnológica foi selecionada para alta performance em GPUs como a RTX 3060:

**PyTorch 2.1+:** Suporte nativo a torch.compile e torch.amp (Mixed Precision).

**MLflow:** Rastreamento de experimentos e métricas como SSIM e Loss em tempo real.

**LPIPS:** Cálculo de similaridade perceptual baseado em VGG para reconstruções de VAE ultra-nítidas.

**Hugging Face Datasets:** Pipelines de dados eficientes para FFHQ-Makeup e CelebA-HQ.

**Einops:** Manipulação de tensores via einsum para operações de patch e unpatch no DiT.

---



    📦 Instalação
 ```
Bash
# Clone o repositório
git clone https://github.com/leohugo1/makeupflow-dit.git
cd makeupflow-dit

# Instale as dependências otimizadas
pip install -r requirements.txt
```
    🏋️ Como Executar
O treinamento é controlado via argumentos de fase:
```
Bash
# Para treinar o VAE (Fase 1)
python train.py --phase train_vae

# Para treinar o Style Encoder (Fase 2)
python train.py --phase train_style_vit

# Para treinar o DiT via Flow Matching (Fase 3)
python train.py --phase train_dit
```
    Inferência e Pesos
Os pesos pré-treinados estarão disponíveis no Hugging Face assim que as etapas de treinamento forem concluídas: 🔗 Hugging Face: [MakeupFlow-DiT](https://huggingface.co/leonardohugo134/makeupflow-dit)