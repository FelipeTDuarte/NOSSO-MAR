# NOSSO-MAR Architecture Overview

```
src/nosso_mar/
├── operators/
│   ├── fno/       FNO2d, FNO3d, GeoFNO, F-FNO, SpectralConv2d/3d
│   ├── gno/       GraphNeuralOperator, WaveInteractionLayer, mesh_utils
│   ├── wno/       WaveletNeuralOperator, WaveletConv2d (Haar DWT)
│   ├── deeponet/  DeepONet, POD-DeepONet, PI-DeepONet
│   ├── adaptive/  AMROperator, refinement criteria, AdaptiveMesh2D
│   └── meshfree/  RBFOperator, KANOperator
├── modules/
│   ├── wave/      WavePropagationNO, dispersion, JONSWAPSpectrum
│   ├── fsi/       WecFSIModule, BEMSurrogate, EOM, PTOModel
│   ├── hydrodynamics/  TidalModule (Phase 3)
│   ├── morphodynamics/ MorphodynamicsModule (Phase 4)
│   └── tracers/        TracerModule (Phase 5)
├── assimilation/  EnKF, 4D-Var (AD), observation operators, state vector
├── digital_twin/  DigitalTwinManager, sensors, StateEstimator, AnomalyDetector
├── reinforcement/ MADDPG+WECLayoutAgent, MAPPO+PTOAgent, WaveFarmEnv
├── hpc/           DDP trainer, SLURM launcher, mixed precision, checkpointing
├── training/      NOTrainer, MARLTrainer, WaveLoss, BEMLoss, PhysicsLoss
├── data/          CapytaineDataset, OceanWave3DDataset, LHS sampler, Zarr
├── core/          ScenarioEngine, CouplingManager, ModelFactory, FieldSchema
└── utils/         seed, metrics, checkpoint, logging, visualisation
```
