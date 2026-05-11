# 15. Benchmark Compatibility Rules

## Meta

Definir o que pode e o que não pode ser comparado entre níveis de modelo, datasets e benchmarks no NOSSO-MAR.

## Estado

- status: planned
- execution_scope: specification_only
- phase1_blocking: no

## Problema que este documento resolve

Projetos multifidelidade falham com frequência porque misturam observáveis, hipóteses e níveis físicos incompatíveis. Este documento impede que benchmarks sejam usados fora do seu domínio de validade.

## Regra mestra

Toda comparação deve responder explicitamente a quatro perguntas:

1. As duas fontes representam a mesma classe física?
2. Os observáveis comparados são semanticamente equivalentes?
3. As escalas temporal, espacial e espectral são compatíveis?
4. A hipótese física mais fraca do par ainda suporta a comparação proposta?

Se qualquer resposta for negativa, a comparação é proibida ou deve ser classificada como apenas qualitativa.

## Matriz principal

| Origem A | Origem B | Comparação permitida | Observação |
|---|---|---|---|
| Analítico | BEM | parcial | apenas em regimes simplificados e grandezas compatíveis |
| BEM | BEM | sim | desde que DOFs, geometria e convenções coincidam |
| BEM | Espectral | parcial | estatísticas ou forcing, não campo fase a fase |
| BEM | Fase resolvida | parcial | WSI agregado, não interface livre completa |
| BEM | Experimental | parcial | coeficientes ou resposta agregada, respeitando escala |
| Espectral | Fase resolvida | parcial | estatísticas e conteúdo energético, não fase local completa |
| Fase resolvida | NWT/CFD | sim parcial | campos comparáveis após alinhamento e remapeamento |
| Fase resolvida | Experimental | sim parcial | superfície livre, séries e regiões equivalentes |
| CFD/FSI | Experimental | sim parcial | campos e cargas com protocolo forte de alinhamento |

## Comparações explicitamente proibidas

1. Usar BEM linear como verdade completa para superfície livre não linear.
2. Usar modelo espectral como benchmark de fase instantânea da superfície.
3. Comparar potência ou força sem alinhar convenções de escala, fase e sinal.
4. Comparar campos espaciais sem declarar interpolação, máscara e resolução efetiva.
5. Misturar dados experimentais sem relatório mínimo de incerteza.

## Regras por observável

### Added mass, damping e excitation force

Comparáveis entre:
- BEM
- experimento dedicado
- modelos hidrodinâmicos compatíveis

Não comparáveis diretamente com:
- produto espectral puro sem modelo estrutural equivalente

### RAO e movimento do corpo

Comparáveis entre:
- BEM
- modelos acoplados compatíveis
- experimento de tanque com instrumentação adequada

### Superfície livre fase a fase

Comparáveis entre:
- modelo fase resolvida
- NWT
- CFD de superfície livre
- experimento espacial ou séries temporais equivalentes

Não comparáveis diretamente com:
- BEM puro
- modelo espectral puro

### Estatísticas de onda

Comparáveis entre:
- espectral
- fase resolvida após pós-processamento
- experimento após análise estatística

## Regras de alinhamento

Toda comparação quantitativa deve declarar:

- remapeamento espacial usado
- remapeamento temporal usado
- filtro espectral usado
- interpolação ou agregação usada
- máscara espacial aplicada
- janelas válidas de comparação

## Níveis de comparação

### Nível C1. Comparação direta

Mesmo observável, mesma escala, mesma hipótese central.

### Nível C2. Comparação derivada

Observáveis diferentes, mas redutíveis a quantidade comum.

### Nível C3. Comparação qualitativa

Apenas morfologia, tendência ou regime. Não usar para erro quantitativo principal.

## Regras para reports futuros

Todo relatório de benchmark deve incluir:

- benchmark_level_pair
- compatibility_class C1, C2 ou C3
- justificativa física
- transformações aplicadas
- limitações reconhecidas

## Regra específica para a Fase 1 agora

A validação executável da Fase 1 continua ancorada em Capytaine. Outras comparações são planeadas, mas nesta fase não redefinem o critério principal de viabilidade.

## Downstream impact

Este documento deve ser referenciado por:

- todos os novos datasets multifidelidade
- futuros relatórios de benchmark
- futuras losses de alinhamento multifidelidade
- integração de dados experimentais
