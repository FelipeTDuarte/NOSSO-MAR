# 12. Physics Loss Library

## Meta

Definir a biblioteca conceitual de funções de perda físicas do NOSSO-MAR, separando claramente:

- loss derivada da física
- loss derivada de dados
- loss derivada de acoplamento
- loss derivada de alinhamento multifidelidade

## Estado

- status: planned
- execution_scope: specification_only
- phase1_blocking: no
- immediate_training_activation: no

## Princípio central

Loss física é função de equações, restrições e observáveis. Não é função do solver por si só. Porém, a disponibilidade e a forma discreta dos termos dependem dos observáveis que cada nível de benchmark ou dataset fornece.

## Forma geral de loss híbrida

A forma canónica, a adaptar por operador, é:

\[
\mathcal{L}_{total} = \lambda_{data}\mathcal{L}_{data} + \lambda_{pde}\mathcal{L}_{pde} + \lambda_{bc}\mathcal{L}_{bc} + \lambda_{ic}\mathcal{L}_{ic} + \lambda_{cons}\mathcal{L}_{cons} + \lambda_{cpl}\mathcal{L}_{cpl} + \lambda_{mf}\mathcal{L}_{mf} + \lambda_{exp}\mathcal{L}_{exp}
\]

Os pesos podem ser fixos, adaptativos, curriculares ou multiobjetivo. A política concreta de ponderação fica para um spec futuro de treino avançado.

## Taxonomia oficial das losses

### L-00. Data fidelity loss

#### Objetivo
Ajustar predição a dados de referência.

#### Formas permitidas
- MSE
- MAE
- relative L2
- complex L2 para saídas em frequência
- Huber quando houver ruído experimental

#### Observação
Não é loss física, mas é parte do sistema híbrido.

### L-10. Boundary condition loss

#### Objetivo
Impor condições de fronteira geométricas, cinemáticas ou dinâmicas.

#### Aplicações
- parede rígida
- fronteira radiativa
- entrada prescrita
- superfície livre linearizada quando apropriado

### L-11. Initial condition loss

#### Objetivo
Impor estado inicial em problemas transitórios.

### L-20. Governing PDE residual loss

#### Objetivo
Minimizar residual forte ou fraco da PDE governante.

#### Formas permitidas
- residual forte por autodiferenciação
- residual forte por diferenças finitas sobre grelha predita
- residual fraco por integração contra funções teste

#### Regra de engenharia
Para campos oscilatórios ou com fronteiras não periódicas, a forma fraca ou híbrida deve ser considerada para reduzir rigidez numérica e erros de derivação local.

### L-21. Conservation loss

#### Objetivo
Impor conservação global ou local.

#### Exemplos
- massa
- energia
- fluxo integrado
- balanço entre entrada, extração e dissipação parametrizada

### L-30. WSI equation-of-motion loss

#### Objetivo
Garantir consistência entre coeficientes hidrodinâmicos, forças e resposta do corpo.

#### Forma base em frequência

\[
\mathcal{L}_{eom} = \left\|\Big[-\omega^2(\mathbf{M}+\mathbf{A}(\omega)) + i\omega(\mathbf{B}(\omega)+\mathbf{C}_{pto}) + (\mathbf{K}_h+\mathbf{K}_{pto})\Big]\boldsymbol{\xi}(\omega)-\mathbf{F}_{exc}(\omega)\right\|^2
\]

#### Uso recomendado
- F1A com DeepONet
- F1A com GNO para arranjos ou geometrias mais ricas

#### Requisitos
- coerência de unidades
- representação consistente de números complexos
- tratamento explícito de frequência e DOFs

### L-31. Passivity and positivity loss

#### Objetivo
Desencorajar saídas fisicamente absurdas.

#### Exemplos
- added mass negativa fora de regimes admissíveis
- damping radiativo negativo onde não faz sentido
- potência absorvida não física sob convenções definidas

#### Observação
Isto é loss de regularização física, não substitui benchmark.

### L-32. Smooth frequency-response loss

#### Objetivo
Evitar curvas não físicas com serrilhado espúrio em \(A(\omega)\), \(B(\omega)\), \(F_{exc}(\omega)\) ou RAOs.

#### Forma sugerida
Penalização de derivada segunda excessiva ou regularização espectral local.

#### Cuidado
Não pode apagar picos ressonantes reais.

### L-40. Free-surface field residual loss

#### Objetivo
Impor consistência do campo de ondas com a formulação escolhida.

#### Casos previstos
- forma rasa simplificada
- forma não-hidrostática simplificada
- residual de modelo fase resolvida quando a formulação estiver definida no dataset

#### Regra
Esta loss só pode ser ativada quando a formulação matemática concreta do benchmark estiver declarada em `11_MATHEMATICAL_MODEL_FOUNDATIONS.md` e o dataset disponibilizar os observáveis necessários.

### L-41. Spectral consistency loss

#### Objetivo
Alinhar estatísticas e conteúdo espectral entre predição e referência.

#### Casos previstos
- perda em \(S(f,\theta)\)
- perda em energia por banda
- perda em momentos espectrais

#### Uso recomendado
- ligação entre benchmarks espectrais e fase resolvida
- supervisão parcial quando campo fase a fase não está disponível

### L-50. Coupling consistency loss

#### Objetivo
Assegurar coerência entre F1A e F1B no acoplamento F1C.

#### Exemplos
- balanço energia incidente versus extraída
- coerência entre condições locais de onda e resposta do dispositivo
- consistência entre campo modificado e potência prevista

#### Estado
Planeada. Não ativar agora como requisito para encerrar a Fase 1.

### L-60. Multi-fidelity alignment loss

#### Objetivo
Aprender correções ou pontes entre níveis de fidelidade.

#### Exemplos
- alinhamento de observáveis comuns
- correção residual baixa para alta fidelidade
- consistência entre resoluções

### L-70. Experimental consistency loss

#### Objetivo
Usar dados experimentais como supervisão parcial ou regularização externa.

#### Formas previstas
- loss em séries temporais de superfície livre
- loss em envelopes ou espectros
- loss em movimentos medidos
- loss ponderada por incerteza instrumental

## Política de ativação por fase

| Loss | Fase 1 agora | Fase 1 futura | Fase 2+ |
|---|---|---|---|
| L-00 Data fidelity | ativa | ativa | ativa |
| L-10 BC | opcional | ativa conforme caso | ativa |
| L-11 IC | rara | opcional | ativa conforme modelo |
| L-20 PDE residual | opcional | progressiva | ativa |
| L-21 Conservation | opcional | progressiva | ativa |
| L-30 WSI EOM | altamente recomendada | ativa | ativa |
| L-31 Passivity/positivity | recomendada | ativa | ativa |
| L-32 Smooth frequency response | recomendada | ativa | ativa |
| L-40 Free-surface field residual | não obrigatória | piloto | ativa |
| L-41 Spectral consistency | opcional | piloto | ativa |
| L-50 Coupling consistency | não obrigatória | piloto | ativa |
| L-60 Multi-fidelity alignment | não | piloto | ativa |
| L-70 Experimental consistency | não | piloto | ativa |

## Regras de discretização para PDE e EOM

### Para saídas de campo

- preferir não dimensionalização antes do cálculo do residual
- documentar claramente a ordem da derivada espacial e temporal
- declarar se o residual usa autodiferenciação, diferenças finitas ou forma fraca
- tratar fronteiras não periódicas com cuidado especial

### Para saídas em frequência

- usar convenção explícita para número complexo
- armazenar real e imaginário ou amplitude e fase de modo consistente
- calcular residual da EOM em forma matricial e por frequência
- reportar resíduos por DOF e por banda de frequência

## Tabela mínima por loss

Toda loss nova deve ser documentada assim:

- id da loss
- equação de origem
- observáveis necessários
- unidade implícita ou não dimensionalização
- operador alvo
- benchmark compatível
- risco de misuse
- relação com outras losses

## Downstream impact

Este documento governa futuras extensões em:

- `src/nossomar/loss/physics_losses.py`
- `src/nossomar/physics/residuals_torch.py`
- configs de treino e curricula
- documentação de benchmarks e datasets
