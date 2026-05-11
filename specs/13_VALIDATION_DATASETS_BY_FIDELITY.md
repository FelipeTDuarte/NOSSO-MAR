# 13. Validation Datasets by Fidelity

## Meta

Definir os contratos dos datasets de benchmark e validação por nível de fidelidade, sem obrigar a produção imediata desses datasets nesta fase.

## Estado

- status: planned
- execution_scope: specification_only
- phase1_blocking: no
- current_operational_focus: Capytaine_only

## Princípio central

Cada nível de modelo gera um dataset diferente. Esses datasets podem coexistir num ecossistema multifidelidade apenas se tiverem:

- observáveis claramente definidos
- metadados compatíveis
- regras explícitas de alinhamento
- limites claros do que pode ser comparado

## Taxonomia oficial

### D-L1. Analítico ou semi-analítico

#### Objetivo
Harness leve, sanity checks, manufactured solutions ou aproximações fechadas.

#### Inputs típicos
- parâmetros geométricos simples
- frequência ou regime reduzido
- parâmetros do sistema

#### Outputs típicos
- respostas simplificadas
- soluções fabricadas
- séries ou curvas de referência leve

#### Formato preferido
JSON pequeno, CSV ou NetCDF simples.

### D-L2. BEM linear

#### Objetivo
Benchmark principal da Fase 1 agora.

#### Ferramenta âncora atual
Capytaine.

#### Inputs mínimos
- geometria ou parâmetros equivalentes
- frequências
- profundidade quando aplicável
- direção de onda quando aplicável
- configuração de DOFs

#### Outputs mínimos
- added mass
- radiation damping
- excitation force
- resposta ou quantidades necessárias para a EOM

#### Formato preferido
Zarr ou NetCDF para lotes maiores, JSON apenas para protótipos pequenos.

### D-L3. Modelo espectral

#### Objetivo
Fornecer boundary conditions e benchmarks estatísticos.

#### Inputs mínimos
- bathymetry
- vento se aplicável
- fronteiras espectrais
- malha ou domínio

#### Outputs mínimos
- \(H_s\)
- \(T_p\)
- direção média
- espectro ou momentos espectrais

### D-L4. Modelo fase resolvida ou NWT

#### Objetivo
Benchmark de campo espaço-temporal.

#### Inputs mínimos
- domínio e bathymetry
- condições de geração de onda
- condições de fronteira
- discretização temporal e espacial

#### Outputs mínimos
- \(\eta(x,y,t)\)
- velocidades relevantes
- medidas integradas de energia quando disponíveis

### D-L5. CFD ou FSI de alta fidelidade

#### Objetivo
Benchmark seletivo para casos críticos e calibração local.

#### Outputs mínimos
- campos locais
- forças detalhadas
- pressões ou tensões quando aplicável
- superfície livre não linear quando disponível

### D-L6. Dados experimentais externos

#### Objetivo
Validação externa e teste de transferibilidade.

#### Outputs possíveis
- séries temporais de sonda
- superfície livre reconstruída
- movimentos
- forças
- potência
- espectros medidos

## Esquema mínimo de metadados

Todo dataset deve conter:

- dataset_id
- fidelity_level
- physics_class
- source_type
- source_tool_or_lab
- geometry_description
- coordinate_system
- units_system
- input_schema_version
- output_schema_version
- temporal_resolution
- spatial_resolution
- uncertainty_model
- preprocessing_steps
- license_or_usage_restriction

## Observáveis comuns

Todo dataset deve mapear as suas saídas para um vocabulário comum quando aplicável:

- free_surface_elevation
- velocity_field
- pressure_field
- added_mass
- radiation_damping
- excitation_force
- body_motion
- absorbed_power
- spectral_density
- wave_statistics

## Regras de armazenamento

### JSON
Apenas para datasets pequenos, protótipos, contratos e amostras.

### NetCDF
Preferido para campos em grelha, séries temporais estruturadas e interoperabilidade científica.

### Zarr
Preferido para lotes grandes, treino em cloud ou acesso paralelo.

## Estado por nível na situação atual

| Nível | Estado agora | Ação imediata |
|---|---|---|
| D-L1 | possível | manter como harness |
| D-L2 | ativo | continuar com Capytaine |
| D-L3 | planeado | só especificar |
| D-L4 | planeado | só especificar |
| D-L5 | planeado | só especificar |
| D-L6 | planeado | só especificar |

## Templates mínimos

Cada futuro dataset deve ter dois ficheiros:

1. `dataset_name.metadata.yaml`
2. `dataset_name.schema.md`

### Conteúdo de `schema.md`

- propósito do dataset
- inputs
- outputs
- malha e resolução
- convenções físicas
- limitações
- benchmarks compatíveis

## Regra de ouro

Nenhum dataset novo entra no projeto sem declarar:

- qual nível de fidelidade representa
- quais observáveis são comparáveis com outros níveis
- qual o benchmark compatível
- qual o spec de compatibilidade associado

## Downstream impact

Este documento governa:

- futuras extensões em `src/nossomar/data/`
- geração de bases de benchmark
- integração com `multifidelity.py`
- relatórios de validação e scripts de benchmark
