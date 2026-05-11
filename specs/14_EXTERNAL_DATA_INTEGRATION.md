# 14. External Data Integration

## Meta

Definir como dados externos, laboratoriais, observacionais e de campanhas experimentais devem ser integrados no NOSSO-MAR no futuro, sem ativação operacional imediata nesta fase.

## Estado

- status: planned
- execution_scope: specification_only
- phase1_blocking: no
- immediate_external_ingestion_required: no

## Motivação

Dados externos são essenciais para:

- validação fora do ecossistema de simulação
- benchmark independente
- calibração de losses físicas em regimes reais
- avaliação de robustez e transferibilidade

## Tipos de dados aceites

### E-1. Laboratório de tanque de ondas

Exemplos:
- sondas resistivas ou capacitivas
- reconstrução de superfície livre
- medição de movimentos do corpo
- forças em amarração ou PTO instrumentado

### E-2. Técnicas ópticas ou laser

Exemplos:
- laser sheet
- reconstrução estéreo
- visão calibrada para interface livre
- sistemas de medição espacial da superfície

### E-3. Sensores metoceânicos externos

Exemplos:
- boias
- ADCP
- maregrafos
- altimetria ou produtos derivados

## Requisitos mínimos de aceitação

Todo dataset externo deve declarar:

- origem institucional
- escala física ou escala de modelo
- protocolo experimental
- incerteza instrumental
- frequência de amostragem
- sincronização temporal
- referência espacial e orientação
- convenção de fase e de frequência quando aplicável
- processamento e filtragem aplicados

## Pipeline conceitual de integração

1. aquisição
2. validação de metadados
3. harmonização de unidades
4. alinhamento temporal
5. alinhamento espacial
6. mapeamento para observáveis comuns
7. quantificação de incerteza
8. publicação em formato interno compatível

## Contrato obrigatório de metadados

Campos mínimos:

- provider
- campaign_id
- facility_name
- scaling_law
- sensor_type
- measured_variable
- sampling_rate_hz
- coordinate_reference
- uncertainty_std
- calibration_date
- preprocessing_pipeline
- missing_data_policy

## Regras por tipo de uso

### Uso como benchmark externo

- dados brutos preservados
- dados processados versionados
- incerteza explícita
- janelas válidas de comparação definidas

### Uso como treino parcial

- permitido apenas com spec complementar
- incerteza deve entrar no weighting da loss
- gaps e filtros precisam de ser declarados

### Uso como validação final

- preferido nesta fase futura
- não deve contaminar treino sem decisão explícita

## Superfície livre reconstruída

Quando o dado externo for um campo de superfície livre reconstruído:

- declarar resolução espacial efetiva
- declarar resolução temporal efetiva
- declarar zona cega e máscaras
- declarar método de reconstrução
- declarar erro médio e erro máximo conhecidos

## Escalonamento e similitude

Para dados de tanque de ondas, declarar explicitamente:

- lei de escala adotada
- grandezas preservadas
- distorções esperadas
- limitações viscosa, capilar ou estrutural

## Incerteza e pesos

Todo uso quantitativo em benchmark ou loss deve incluir:

- peso por incerteza quando aplicável
- filtros de qualidade por amostra
- possibilidade de exclusão de sensores instáveis

## Artefatos mínimos por campanha externa

Cada campanha integrada deve conter:

- `campaign.metadata.yaml`
- `campaign_protocol.md`
- `sensor_map.json` ou equivalente
- `uncertainty_report.md`

## Regra de segurança científica

Dados externos não podem ser tratados como ground truth absoluto sem relatório de incerteza e protocolo de aquisição.

## Downstream impact

Este documento governa futuras integrações com:

- FEUP e outros laboratórios
- open data observacional
- benchmarks experimentais em `reports/`
- losses ponderadas por incerteza
