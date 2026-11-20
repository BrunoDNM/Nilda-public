
# Escopo do protocolo e dados

## Entradas de classificação

- A fonte de dados [9] disponibiliza 200 trajetórias sintéticas de treino/validação interna e 20 trajetórias sintéticas de avaliação reservada. A procedência e independência do procedimento manual de criação das trajetórias exigem cautela.
- Os dois arquivos sintéticos estão incluídos com hashes de integridade, com procedência e atribuição documentadas em `THIRD_PARTY.md`. Não contêm dados de pacientes.
- Imagens base têm raster 602 × 369, com amostragem bilinear do `scikit-image` para 128×128 no NewCare e 64×64 no MiniCare, além de quatro atributos (tortuosidade, dimensão fractal, contagem de reversões, trocas de quadrante).
- O escalonador de quatro características foi ajustado somente nos 150 exemplos de treino. No script de inferência é carregado do arquivo verificado.
- O limiar binário é estritamente acima de 0,5; confiança numérica não equivale a calibração ou desempenho clínico.

## Sensores UWB

- Treze distâncias de referência, cada qual resumindo quinze leituras declaradas, de 25 a 450 cm. Total de 195 leituras individuais **não arquivadas**.
- O MAE é a média aritmética dos erros absolutos **das médias** por distância, e não MAE calculado sobre 195 observações individuais.

## Propriedade intelectual

- O repositório distribui somente os dois arquivos JSON sintéticos essenciais do SmartCare, com atribuição em `THIRD_PARTY.md`. Não inclui código, notebook ou modelo original SmartCare; a publicação de Oliveira et al. permanece citada na bibliografia.
- O pré-processamento aqui é uma implementação nova, conferida por equivalência numérica com o protocolo usado nos testes históricos.
