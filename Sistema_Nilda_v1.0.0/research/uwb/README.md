# Medições de distância por UWB

O manuscrito descreve um ensaio entre pares de sensores fixos, com distância de referência medida por trena. O protocolo relata 15 repetições por distância em 13 distâncias entre 25 e 450 cm. O [CSV arquivado](medicoes_agregadas_uwb.csv) contém uma linha por distância, com a referência, a média da distância detectada, o erro absoluto e o número de repetições declarado.

## Recalcular o MAE das médias

Na raiz do pacote, execute o comando abaixo com Python 3. O script usa apenas a biblioteca padrão.

```bash
python research/uwb/verificar_agregados.py
```

O script verifica os erros absolutos das 13 linhas contra a diferença entre a distância média detectada e a referência. A soma dos erros é **42,6 cm**. Dividida pelas 13 distâncias, resulta em **3,276923… cm**, arredondados para **3,28 cm**, conforme o manuscrito. Cada distância tem o mesmo peso nesse cálculo.

## Limites da verificação

As 195 leituras individuais previstas pelo protocolo não estão disponíveis. O campo `repeticoes_declaradas` registra o número relatado no manuscrito. As médias arquivadas permitem conferir o erro por distância e o MAE agregado. Para conferir as médias, a dispersão entre repetições, intervalos de confiança ou o MAE das leituras individuais, seriam necessários os registros brutos. Desvios positivos e negativos podem se cancelar antes do cálculo do erro absoluto de uma média.

O ensaio mediu a distância entre dois sensores fixos. A precisão da trilateração, a localização da pessoa em ambientes residenciais e o desempenho do sistema integrado requerem outros ensaios.
