# 11. Mathematical Model Foundations

## Meta

Formalizar a base matemática comum do NOSSO-MAR para que funções de perda, benchmarks, datasets e regras de comparabilidade sejam derivados da física do problema e não do nome do solver numérico.

## Estado

- status: planned
- execution_scope: specification_only
- phase1_blocking: no
- owners: scientific_lead, software_architecture

## Objetivos normativos

Este documento deve responder a cinco perguntas:

1. Quais problemas físicos o projeto representa em cada fase.
2. Quais equações governantes são consideradas canónicas para cada classe de problema.
3. Quais hipóteses físicas separam um nível de modelo de outro.
4. Quais observáveis são fundamentais, derivados ou opcionais.
5. Como cada operador neural se relaciona com essas formulações.

## Princípio central

A formulação matemática é a fonte primária da loss física. O solver numérico é fonte de dados, baseline e benchmark. Logo, a loss pode ser independente da ferramenta numérica específica, mas não é independente das hipóteses físicas do modelo que está a ser representado.

## Notação global obrigatória

Toda grandeza usada noutros specs deve ser declarada aqui.

### Campos e estados

- \(\eta(x,y,t)\): elevação da superfície livre
- \(u(x,y,z,t), v(x,y,z,t), w(x,y,z,t)\): componentes de velocidade
- \(h(x,y)\): profundidade local
- \(H = h + \eta\): coluna de água total em modelos rasos
- \(p\): pressão
- \(S(f,\theta)\): espectro direcional
- \(\mathbf{q}\): vetor de estado genérico

### Variáveis de WEC

- \(m\): massa estrutural
- \(A(\omega)\): added mass
- \(B(\omega)\): radiation damping
- \(F_{exc}(\omega)\): força de excitação
- \(\xi(\omega)\): resposta complexa
- \(c_{pto}, k_{pto}\): coeficientes do PTO
- \(P_{abs}\): potência absorvida

### Convenções obrigatórias

- sistema SI
- ângulos em radianos internamente
- frequência em Hz ou rad/s sempre declarada
- coordenadas cartesianas e orientação explicitadas por dataset

## Classes de modelo físico

### Classe M1. WSI linear em domínio da frequência

Uso principal: F1A.

#### Formulação canónica

Para um grau de liberdade, a forma base é:

\[
\left[-\omega^2(m + A(\omega)) + i\omega(B(\omega) + c_{pto}) + (k_h + k_{pto})\right]\xi(\omega) = F_{exc}(\omega)
\]

Para múltiplos graus de liberdade:

\[
\Big[-\omega^2(\mathbf{M}+\mathbf{A}(\omega)) + i\omega(\mathbf{B}(\omega)+\mathbf{C}_{pto}) + (\mathbf{K}_h+\mathbf{K}_{pto})\Big]\boldsymbol{\xi}(\omega)=\mathbf{F}_{exc}(\omega)
\]

#### Hipóteses

- fluido incompressível
- escoamento potencial linear
- pequenas amplitudes
- linearização da superfície livre
- resposta harmónica

#### Observáveis fundamentais

- added mass
- radiation damping
- excitation force
- RAO de movimento
- potência absorvida

#### Limites

- não resolve efeitos viscosos fortes
- não representa breaking
- não representa não linearidade de grandes amplitudes

### Classe M2. Ondas espectrais

Uso principal: referência de fronteira e benchmarks estatísticos.

#### Formulação canónica

Equação de balanço de ação em forma genérica:

\[
\frac{\partial N}{\partial t} + \nabla_x \cdot (\mathbf{c}_x N) + \frac{\partial (c_\theta N)}{\partial \theta} + \frac{\partial (c_\sigma N)}{\partial \sigma} = \frac{S_{tot}}{\sigma}
\]

#### Observáveis fundamentais

- \(H_s\)
- \(T_p\)
- direção média
- fluxo de energia espectral

#### Limites

- não entrega fase local da superfície livre
- não substitui benchmark fase a fase

### Classe M3. Ondas fase resolvida

Uso principal: F1B e benchmark de campos.

#### Formulações possíveis

- Boussinesq ou não-hidrostático simplificado
- Navier Stokes de superfície livre em NWT
- modelos de potencial não linear, conforme caso

#### Observáveis fundamentais

- \(\eta(x,y,t)\)
- \(u, v\)
- reflexão, transmissão, shoaling, difração
- energia por banda ou por região

#### Limites

Dependem da família do modelo. Cada dataset deve declarar a formulação concreta usada.

### Classe M4. CFD ou FSI de alta fidelidade

Uso principal: benchmark seletivo e calibração de casos críticos.

#### Observáveis fundamentais

- campo de pressão
- campo de velocidade
- cargas distribuídas
- vorticidade
- superfície livre não linear

#### Limites

- alto custo computacional
- cobertura paramétrica limitada

### Classe M5. Dados experimentais

Uso principal: benchmark externo e verificação de transferibilidade.

#### Observáveis possíveis

- séries temporais de sondas
- superfície livre reconstruída
- movimentos do corpo
- forças e potência
- espectros e envelopes

#### Limites

- ruído
- incerteza instrumental
- diferenças de escala
- imperfeições de contorno e geração de onda

## Mapeamento para operadores

| Operador | Papel principal | Classe de modelo associada |
|---|---|---|
| DeepONet / PI-DeepONet | resposta funcional WSI | M1 |
| GNO | geometria irregular, múltiplos corpos | M1, M3 |
| FNO / WNO | campos em grelha | M3 |
| RINO | transferência entre resoluções e fidelidades | M2, M3, M4 |

## Regras normativas

1. Toda nova loss física deve apontar para uma equação ou restrição declarada aqui.
2. Todo novo benchmark deve declarar qual classe M1-M5 representa.
3. Todo dataset novo deve listar observáveis fundamentais e derivados com referência a este documento.
4. Toda comparação entre fontes de dados deve ser validada por `15_BENCHMARK_COMPATIBILITY_RULES.md`.

## Downstream impact

Este documento alimenta diretamente:

- `12_PHYSICS_LOSS_LIBRARY.md`
- `13_VALIDATION_DATASETS_BY_FIDELITY.md`
- `15_BENCHMARK_COMPATIBILITY_RULES.md`
- futuras implementações em `src/nossomar/physics/` e `src/nossomar/loss/`
