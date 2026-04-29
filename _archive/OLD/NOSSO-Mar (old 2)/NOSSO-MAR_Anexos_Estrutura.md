
# Estrutura de anexos — Projeto NOSSO-MAR

## Anexo A — Visão por fases
Texto com objetivo, hipótese, entradas, saídas, dependências, riscos e entregáveis de cada fase.

## Anexo B — Especificação dos módulos
Definição formal do Módulo 1 (propagação), Módulo 2 (interação onda-estrutura), contratos de interface e estratégia de acoplamento.

## Anexo C — Estratégia de dados
Plano de geração de dados, amostragem, convenções de ficheiros, metadados, curadoria, treino, validação e versionamento.

## Anexo D — Solvers de referência
Justificativa técnica para uso de Capytaine, HAMS-MREL, NEMOH e solvers de propagação por fase do projeto.

## Anexo E — Inventário de scripts por fase

### Fase 0
- `create_nosso_mar_repo_fixed.py`
- `scripts/run_scenario.py`
- `scripts/train.py`
- `scripts/train_marl.py`

### Fase 1
- geração e treino do módulo de resposta WEC
- pós-processamento de curvas hidrodinâmicas
- validação BEM vs surrogate

### Fase 2
- geração dos casos de propagação
- treino do operador de propagação
- avaliação espacial e espectral

### Fase 3
- acoplamento iterativo M1–M2
- métricas de convergência
- comparação com solver de referência

### Fase 4
- número variável de WECs
- grafos/conjuntos/atenção
- validação de layouts

### Fase 5
- integração com algoritmo genético
- função objetivo multiobjetivo
- ranking de layouts

### Fase 6
- transferência entre sites
- fine-tuning por domínio
- quantificação de incerteza

### Fase 7
- assimilação de dados
- digital twin
- operação e atualização online

## Anexo F — Plano de publicações
Mapa de artigos por fase, pergunta científica, experimento central e venue-alvo.

## Anexo G — Validação e benchmarks
Casos de referência, datasets externos, benchmarks internos e estratégia de validação cruzada.
