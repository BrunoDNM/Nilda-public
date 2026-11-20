
# MiniCare — somente o arquivo INT8 original preservado

Esta publicação distribui **um único artefato MiniCare**: `Ready-main/minicare/model/dados/modelCNN-minicare-int8.tflite`, preservado do projeto original e verificado com SHA-256 `3644b03d92ee446514c2df052c8285d7107a1fd33c3551e49ac1b9933cbd821a`.

Os registros de auditoria no computador mostram VP=8, FN=2, FP=0, VN=10 nas 20 trajetórias sintéticas; acurácia 90,0%, precisão 100,0%, recall 80,0%, F1 88,9%. Não se incluem modelos MiniCare H5/FP32, nem conversões posteriores, nem se misturam seus resultados nesta versão.

Entrada 1: imagem `1×64×64×1` em `uint8`, escala 0,003586668754, zero 0. Entrada 2: quatro características `1×4` em `uint8`, escala 0,013695535250, zero 118. Saída: escalar `1×1`, `uint8`, escala 0,00390625, zero 0. Limiar de decisão: `p > 0,5`.

**Limites:** avaliação feita em host com TFLite. A paridade do pré-processamento e o consumo de memória/latência/precisão no firmware ESP32 não foram demonstrados experimentalmente. A população é inteiramente sintética.
