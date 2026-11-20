# Resultados de classificação utilizados no manuscrito JHI

**Somente o MiniCare INT8 original preservado é apresentado como MiniCare neste pacote.** A avaliação foi executada em CPU em 18 de novembro de 2026, com 20 trajetórias sintéticas (10 por classe), limiar estrito `p > 0,5` e os modelos originais identificados pelos hashes em `research/results/evaluation.json`.

| Modelo efetivamente avaliado | VP/FN/FP/VN | Acurácia, IC95% | Precisão, IC95% | Recall, IC95% | F1, IC95% |
| --- | --- | --- | --- | --- | --- |
| NewCare (modelo original) | 9/1/0/10 | 95,0 [57,7–99,9] | 100,0 [58,9–100,0] | 90,0 [50,9–99,9] | 94,7 [54,6–99,9] |
| **MiniCare INT8 original preservado** | **8/2/0/10** | **90,0 [52,3–99,1]** | **100,0 [53,1–100,0]** | **80,0 [40,1–98,3]** | **88,9 [45,7–99,1]** |

Os intervalos são Clopper–Pearson de 97,5% para sensibilidade e especificidade, projetados nas métricas com ajuste conservador de Bonferroni para cobertura conjunta de pelo menos 95% por modelo. Esses intervalos pressupõem independência entre as sessões simuladas e o desenho balanceado; não estabelecem desempenho clínico.

O INT8 original é o arquivo `Ready-main/minicare/model/dados/modelCNN-minicare-int8.tflite` identificado por SHA-256 em `research/package/verificar_artefatos.py`. A classificação **90%** foi observada em inferência no computador. **Não houve comprovação dessas métricas em execução no ESP32 nem validação ponta a ponta.**

No manuscrito anterior constavam outras métricas para o MiniCare; elas foram substituídas pelos resultados verificados do **INT8 original**. As médias históricas de confiança e curvas de treinamento que não puderam ser reproduzidas não devem ser tratadas como evidência de funcionamento do arquivo INT8 original.

A íntegra do relatório computacional auditado, reduzido aos dois modelos do Nilda, está em `research/results/evaluation.json`. Os scripts de avaliação reproduzem apenas NewCare e MiniCare INT8 preservado; o SmartCare é trabalho anterior citado no manuscrito, sem distribuição ou carregamento do seu H5.
