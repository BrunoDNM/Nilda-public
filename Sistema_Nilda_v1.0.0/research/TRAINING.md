
# Procedimento de treinamento documentado

O modelo principal NewCare foi treinado com 150 sessões sintéticas, com validação interna em 50 sessões (divisão estratificada semente 42). As 20 sessões separadas foram usadas para a avaliação final. Os dois JSONs originais do SmartCare necessários à reprodução estão no pacote; consulte `THIRD_PARTY.md` para sua origem e verificação por SHA-256.

Scripts de reconstrução do pipeline de treinamento NewCare: `Ready-main/newcare/model/extract_features.py` e `Ready-main/newcare/model/train_multiinput.py`. Dependências estão fixadas em `Ready-main/requirements-jhi.lock`. O modelo NewCare distribuído é o H5 original preservado (SHA-256 no `research/package/verificar_artefatos.py`); retreinar pode produzir pesos diferentes. Não é necessário treinar para executar a avaliação com os pesos preservados.

O MiniCare desta versão é **somente o TFLite INT8 original preservado**, não uma nova execução de treinamento ou quantização. Esta publicação não inclui nem avalia os arquivos H5/FP32 do MiniCare.
