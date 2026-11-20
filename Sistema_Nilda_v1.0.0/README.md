# Sistema Nilda — código e materiais de pesquisa

Implementação e materiais para o estudo **“Monitoramento de deambulação na doença de Alzheimer: avaliação preliminar de sensores e classificadores”**, preparado para o *Journal of Health Informatics*.

O estudo analisa, **separadamente**, medições de distância com sensores UWB e a classificação de **20 trajetórias sintéticas** (10 com deambulação, 10 sem). Não foram realizados testes com pessoas, validação clínica, medição da precisão de trilateração nem teste ponta a ponta entre sensores, localização, classificação e alerta.

## Resultados experimentais

| Modelo de pesquisa | VP / FN / FP / VN | Acurácia | Precisão | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| NewCare H5 original | 9 / 1 / 0 / 10 | 95,0% | 100,0% | 90,0% | 94,7% |
| **MiniCare INT8 original preservado** | **8 / 2 / 0 / 10** | **90,0%** | **100,0%** | **80,0%** | **88,9%** |

O MiniCare distribuído é exclusivamente o modelo **INT8 original** (`Ready-main/minicare/model/dados/modelCNN-minicare-int8.tflite`), não uma versão H5, FP32 ou posteriormente recalibrada. O desempenho informado foi obtido em inferência no computador; não constitui medição no ESP32. A obra anterior **SmartCare**, de Oliveira *et al.* (2022), é referência bibliográfica e linha comparativa do artigo, **não um modelo distribuído neste repositório**.

Em teste independente, 13 distâncias UWB foram representadas pelas médias declaradas de 15 leituras por distância; o erro médio absoluto das **13 médias** é **3,28 cm**. As 195 leituras individuais não estão disponíveis.

## Conteúdo

- `Ready-main/newcare/model/dados/`: 200 trajetórias de treinamento, 20 trajetórias independentes de avaliação, scaler e pesos NewCare H5.
- `Ready-main/minicare/model/dados/`: somente o binário MiniCare INT8 original preservado.
- `Ready-main/comparar_modelos.py`: avaliação dos dois modelos Nilda, matrizes de confusão, ICs e probabilidades por sessão.
- `Ready-main/newcare/newcare/`: pré-processamento independente de imagens e características geométricas.
- `research/`: protocolo, dados agregados UWB, resultados, testes e informações de treinamento/quantização.
- `REFERENCIAS.md`: as 17 referências científicas citadas no manuscrito.
- `THIRD_PARTY.md`: procedência e atribuição dos dados sintéticos SmartCare.
- `CITATION.cff`: metadados de citação do software.

## Reprodução computacional

Ambiente de referência: Debian 13, Python 3.12.14, TensorFlow CPU 2.21.0. Na raiz do repositório:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r Ready-main/requirements-jhi.lock
python research/package/verificar_artefatos.py
PYTHONPATH=Ready-main python -m unittest discover -s research/tests -v
python Ready-main/comparar_modelos.py --output-dir ./resultado-avaliacao
python research/uwb/verificar_agregados.py
sha256sum -c SHA256SUMS
```

`--output-dir` exige um diretório novo. O script grava `evaluation.json` e `predictions.json`. Os scripts de treinamento NewCare estão presentes para investigação, mas **retreinar pode produzir pesos e curvas diferentes** dos arquivos preservados. A tabela acima corresponde aos modelos preservados e não deve ser substituída por resultados de novo treinamento. Os valores históricos também estão arquivados em `research/results/evaluation.json`.

## Métodos, fontes e limites

Os JSONs sintéticos foram obtidos do [repositório da equipe SmartCare](https://github.com/Unilasalle-SmartCare/smartcare-machinelearning/tree/1d479627158cfb46bef837afedc0f63b6c8bbbe0/model/dados) na revisão indicada em `THIRD_PARTY.md`, com conteúdo verificado por SHA-256. O projeto distribui **somente esses dois conjuntos de trajetórias** como insumo externo necessário à avaliação dos modelos Nilda; não reproduz o classificador, os pesos, os notebooks ou o código original SmartCare. O processamento distribuído aqui foi reimplementado e conferido contra os dados de origem. A fonte bibliográfica central é [Oliveira et al., 2022](https://doi.org/10.1007/s42979-022-01091-3).

A métrica UWB é calculada sobre 13 médias, não sobre cada uma das 195 leituras individuais. Os ICs da classificação pressupõem desenho sintético balanceado e independência entre sessões. Os resultados não demonstram desempenho clínico. Detalhes adicionais: `research/PROTOCOL.md`, `research/RESULTS.md`, `research/QUANTIZATION.md` e `research/uwb/README.md`.

## Citação e licença

O GitHub utiliza `CITATION.cff` para exibir **Cite this repository**. O DOI, quando emitido pelo Zenodo para uma release, identifica uma versão arquivada e deve ser citado no artigo. O arquivo de citação registra o identificador público do autor; a grafia acadêmica completa pode ser atualizada nos metadados da versão publicada.

O código original do Nilda é distribuído sob a licença MIT (`LICENSE`). Atribuições e condições aplicáveis aos dois conjuntos de dados sintéticos de terceiros estão em `THIRD_PARTY.md`; a licença MIT **não se aplica automaticamente aos dados desses terceiros**. Documentos científicos do projeto permanecem atribuídos aos seus autores.
