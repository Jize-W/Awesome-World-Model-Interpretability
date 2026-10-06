# Awesome World Model Interpretability

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Focus](https://img.shields.io/badge/Focus-JEPA%20%26%20Latent%20Worlds-blue)

A curated list of papers on **world model interpretability**, **Joint-Embedding Predictive Architectures (JEPAs)**, and related explainable AI methods.

Contributions and corrections are welcome through issues and pull requests.

- [Awesome World Model Interpretability](#awesome-world-model-interpretability)
  - [Category Overview](#category-overview)
  - [General XAI Toolbox](#general-xai-toolbox)
  - [Latent Space World Models](#latent-space-world-models)
    - [General Models and Predictive Architectures](#general-models-and-predictive-architectures)
    - [Downstream JEPA Applications](#downstream-jepa-applications)
      - [Robotics and Embodied Control](#robotics-and-embodied-control)
      - [3D Perception and Autonomous Driving](#3d-perception-and-autonomous-driving)
      - [Medicine and Physiological Sensing](#medicine-and-physiological-sensing)
      - [Genomics, Cells, and Collider Physics](#genomics-cells-and-collider-physics)
      - [Graphs, Time Series, and Physical Fields](#graphs-time-series-and-physical-fields)
      - [Remote Sensing, Audio, and Language](#remote-sensing-audio-and-language)
  - [Related Work in XAI Involving World Models](#related-work-in-xai-involving-world-models)
    - [Explicit Structure and Interpretable Model Design](#explicit-structure-and-interpretable-model-design)
    - [World Representations in Transformers and LLMs](#world-representations-in-transformers-and-llms)
    - [World Models for Explaining Agents](#world-models-for-explaining-agents)
    - [Transferable Methods and Behavioral Evaluation](#transferable-methods-and-behavioral-evaluation)
  - [Interpreting Learned World Models](#interpreting-learned-world-models)

## Category Overview

| Category | What It Covers | Relationship to World Model Interpretability |
| :--- | :--- | :--- |
| **General XAI Toolbox** | Attribution, concepts, probes, sparse features, circuits, causal interventions, and steering. | **Methodological foundations.** Provides tools that can be adapted to world models; their effectiveness for latent predictive dynamics requires validation. |
| **Latent Space World Models** | General predictive architectures and domain-specific JEPA applications. | **Models to interpret.** Defines the model families and application settings that motivate interpretation; downstream performance alone does not explain internal mechanisms. |
| **Related Work in XAI Involving World Models** | Explicitly structured model design, emergent world representations in Transformers and LLMs, and world models used to explain agents. | **Weakly related / indirect relevance.** Primarily studies structured model design, emergent world representations, or agent explanations, rather than directly interpreting the internal predictive mechanisms of learned world models, especially JEPAs. Provides contextual references and potentially transferable ideas. |
| **Interpreting Learned World Models** | Analyses, infrastructure, and interventions targeting the internal representations and mechanisms of learned world models. | **Direct analysis and supporting tools.** Includes JEPA representation studies, interpretability infrastructure, and non-JEPA methods with potential for transfer to JEPAs. |

## General XAI Toolbox

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Axiomatic Attribution for Deep Networks**](https://arxiv.org/abs/1703.01365) <br> *Mukund Sundararajan, Ankur Taly, Qiqi Yan* <br> ICML 2017 | **Integrated Gradients.** Attributes a chosen output to input features. | [Paper](https://arxiv.org/abs/1703.01365) |
| [**“Why Should I Trust You?”: Explaining the Predictions of Any Classifier**](https://arxiv.org/abs/1602.04938) <br> *Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin* <br> KDD 2016 | **LIME.** Fits a local surrogate around a prediction. | [Paper](https://arxiv.org/abs/1602.04938) |
| [**A Unified Approach to Interpreting Model Predictions**](https://arxiv.org/abs/1705.07874) <br> *Scott Lundberg, Su-In Lee* <br> NeurIPS 2017 | **SHAP.** Provides additive feature attributions. | [Paper](https://arxiv.org/abs/1705.07874) |
| [**Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR**](https://arxiv.org/abs/1711.00399) <br> *Sandra Wachter, Brent Mittelstadt, Chris Russell* <br> Preprint 2017 | **Counterfactual explanations.** Asks which input changes would alter a decision. | [Paper](https://arxiv.org/abs/1711.00399) |
| [**Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)**](https://arxiv.org/abs/1711.11279) <br> *Been Kim et al.* <br> ICML 2018 | **Concept methods.** Connects activations to human-defined concepts and measures directional sensitivity. | [Paper](https://arxiv.org/abs/1711.11279) |
| [**Understanding Intermediate Layers Using Linear Classifier Probes**](https://arxiv.org/abs/1610.01644) <br> *Guillaume Alain, Yoshua Bengio* <br> Preprint 2016 | **Probing.** Tests whether information is linearly recoverable from hidden representations. | [Paper](https://arxiv.org/abs/1610.01644) |
| [**Sparse Autoencoders Find Highly Interpretable Features in Language Models**](https://arxiv.org/abs/2309.08600) <br> *Hoagy Cunningham et al.* <br> Preprint 2023; ICLR 2024 version | **Sparse features.** Decomposes activations into an overcomplete set of sparsely active features. | [Paper](https://arxiv.org/abs/2309.08600) |
| [**Zoom In: An Introduction to Circuits**](https://distill.pub/2020/circuits/zoom-in/) <br> *Chris Olah et al.* <br> Distill 2020 | **Circuit analysis.** Studies how internal features combine through network connections. | [Article](https://distill.pub/2020/circuits/zoom-in/) |
| [**Locating and Editing Factual Associations in GPT**](https://arxiv.org/abs/2202.05262) <br> *Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov* <br> NeurIPS 2022 | **Causal tracing and editing.** Uses activation restoration to locate influential computations, followed by targeted weight edits. | [Paper](https://arxiv.org/abs/2202.05262) <br> [Project](https://rome.baulab.info/) |
| [**Steering Language Models with Activation Engineering**](https://arxiv.org/abs/2308.10248) <br> *Alexander Matt Turner et al.* <br> Preprint 2023 | **Activation steering / ActAdd.** Adds contrastive activation directions during inference. | [Paper](https://arxiv.org/abs/2308.10248) |

## Latent Space World Models

### General Models and Predictive Architectures

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**World Models**](https://arxiv.org/abs/1803.10122) <br> *David Ha, Jürgen Schmidhuber* <br> 2018 | **Non-JEPA foundation.** Learns compressed spatial representations and recurrent dynamics for control. | [Paper](https://arxiv.org/abs/1803.10122) <br> [Project](https://worldmodels.github.io/) |
| [**Mastering Diverse Control Tasks through World Models**](https://www.nature.com/articles/s41586-025-08744-2) <br> *Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap* <br> Nature 2025 | **DreamerV3; non-JEPA.** Learns latent dynamics and optimizes behavior through imagined futures. | [Paper](https://www.nature.com/articles/s41586-025-08744-2) |
| [**Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**](https://arxiv.org/abs/2301.08243) <br> *Mahmoud Assran et al.* <br> CVPR 2023 | **I-JEPA.** Predicts masked image-region embeddings rather than pixels. | [Paper](https://arxiv.org/abs/2301.08243) <br> [GitHub](https://github.com/facebookresearch/ijepa) |
| [**MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features**](https://arxiv.org/abs/2307.12698) <br> *Adrien Bardes, Jean Ponce, Yann LeCun* <br> Preprint 2023 | **Motion and content.** Jointly learns visual content and motion features, evaluated on optical flow and segmentation. | [Paper](https://arxiv.org/abs/2307.12698) |
| [**Revisiting Feature Prediction for Learning Visual Representations from Video**](https://arxiv.org/abs/2404.08471) <br> *Adrien Bardes et al.* <br> TMLR 2024 | **V-JEPA.** Learns video representations through masked feature prediction. Frozen evaluations assess appearance and motion representations. | [Paper](https://arxiv.org/abs/2404.08471) |
| [**V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**](https://arxiv.org/abs/2506.09985) <br> *Mido Assran et al.* <br> Preprint 2025 | **Video understanding and action-conditioned control.** V-JEPA 2-AC extends the video representation to robot-action-conditioned prediction and planning. | [Paper](https://arxiv.org/abs/2506.09985) <br> [GitHub](https://github.com/facebookresearch/vjepa2) |
| [**seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models**](https://arxiv.org/abs/2505.03176) <br> *Hafez Ghaemi, Eilif Muller, Shahab Bakhtiari* <br> NeurIPS 2025 | **Invariant and equivariant representations.** Processes observation–transformation sequences and separates aggregate invariant representations from equivariant per-view representations. | [Paper](https://arxiv.org/abs/2505.03176) |
| [**LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**](https://arxiv.org/abs/2603.19312) <br> *Lucas Maes et al.* <br> Preprint 2026 | **LeWM.** Trains action-conditioned latent prediction from pixels with a prediction loss and Gaussian-distribution regularization. Also evaluates physical-quantity probing and surprise detection. | [Paper](https://arxiv.org/abs/2603.19312) |

### Downstream JEPA Applications

#### Robotics and Embodied Control

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**ACT-JEPA: Joint-Embedding Predictive Architecture Improves Policy Representation Learning**](https://arxiv.org/abs/2501.14622) <br> *Aleksandar Vujinovic, Aleksandar Kovacevic* <br> Preprint 2025 | **Application:** Imitation learning through joint action-sequence and latent-observation prediction. | [Paper](https://arxiv.org/abs/2501.14622) <br> [Project](https://act-jepa.github.io/) |
| [**VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model**](https://arxiv.org/abs/2602.10098) <br> *Jingwen Sun et al.* <br> Preprint 2026 | **Application:** Latent state-prediction pretraining followed by action-head fine-tuning; manipulation on LIBERO, LIBERO-Plus, SimplerEnv, and real robots. | [Paper](https://arxiv.org/abs/2602.10098) |
| [**Is Forward Prediction Enough? Physical State Grounding for JEPA World Models**](https://arxiv.org/abs/2608.06799) <br> *Haodong Yan et al.* <br> Preprint 2026 | **PSG-JEPA:** Adds training objectives for proprioceptive state and multi-horizon joint-angle changes, evaluated through probes, planning, and policy learning. | [Paper](https://arxiv.org/abs/2608.06799) |

#### 3D Perception and Autonomous Driving

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Cloud**](https://arxiv.org/abs/2404.16432) <br> *Ayumu Saito, Prachi Kudeshia, Jiju Poovvancheri* <br> Preprint 2024 | **Application:** Predicts point-cloud patch embeddings for 3D representation learning. | [Paper](https://arxiv.org/abs/2404.16432) |
| [**3D-JEPA: A Joint Embedding Predictive Architecture for 3D Self-Supervised Representation Learning**](https://arxiv.org/abs/2409.15803) <br> *Naiwen Hu et al.* <br> Preprint 2024 | **Application:** Context-to-target latent prediction for 3D point-cloud representations. | [Paper](https://arxiv.org/abs/2409.15803) |
| [**LOCATE 3D: Real-World Object Localization via Self-Supervised Learning in 3D**](https://proceedings.mlr.press/v267/mcvay25a.html) <br> *Paul McVay, Sergio Arnaud et al.* <br> ICML 2025 | **Application:** Its 3D-JEPA pretraining supports language-guided object localization from RGB-D observations, followed by a language-conditioned decoder. | [Paper](https://proceedings.mlr.press/v267/mcvay25a.html) <br> [GitHub](https://github.com/facebookresearch/locate-3d) |
| [**Self-Supervised Representation Learning with Joint Embedding Predictive Architecture for Automotive LiDAR Object Detection**](https://arxiv.org/abs/2501.04969) <br> *Haoran Zhu et al.* <br> AAAI 2026 | **AD-L-JEPA:** Predicts BEV embeddings for LiDAR pretraining; evaluated on KITTI3D, Waymo, and ONCE object detection. | [Paper](https://arxiv.org/abs/2501.04969) <br> [Project](https://ad-l-jepa.github.io/) |

#### Medicine and Physiological Sensing

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**A Vision Foundation Model for Cataract Surgery Using Joint-Embedding Predictive Architecture**](https://proceedings.mlr.press/v301/shah26b.html) <br> *Nisarg A. Shah et al.* <br> MIDL; proceedings 2026 | **JHU-VPT(JEPA):** Surgical-video pretraining for step recognition, feedback, and skill assessment. | [Paper](https://proceedings.mlr.press/v301/shah26b.html) |
| [**US-JEPA: A Joint Embedding Predictive Architecture for Medical Ultrasound**](https://arxiv.org/abs/2602.19322) <br> *Ashwath Radhachandran et al.* <br> Preprint 2026 | **Application:** Frozen-teacher latent prediction for ultrasound representations, evaluated on UltraBench classification tasks. | [Paper](https://arxiv.org/abs/2602.19322) |
| [**mmJEPA-ECG: Cross-Posture Robust Contactless Electrocardiogram Monitoring via Millimeter Wave Radar Sensing**](https://ojs.aaai.org/index.php/AAAI/article/download/41242/45203) <br> *Ziyang Liu et al.* <br> AAAI 2026 | **Application:** Physiology-guided JEPA pretraining followed by diffusion-based ECG reconstruction from radar. | [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/41242/45203) |

#### Genomics, Cells, and Collider Physics

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**JEPA-DNA: Grounding Genomic Foundation Models through Joint-Embedding Predictive Architectures**](https://arxiv.org/abs/2602.17162) <br> *Ariel Larey et al.* <br> Preprint 2026 | **Application:** Combines generative objectives with prediction of genomic-segment embeddings; supervised and zero-shot genomic tasks. | [Paper](https://arxiv.org/abs/2602.17162) |
| [**Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics**](https://arxiv.org/abs/2602.02093) <br> *Ali ElSheikh et al.* <br> Preprint 2026 | **Application:** Learns cell representations from partial gene observations; cell-type transfer and perturbation evaluation. | [Paper](https://arxiv.org/abs/2602.02093) |
| [**HEP-JEPA: A Foundation Model for Collider Physics Using Joint Embedding Predictive Architecture**](https://arxiv.org/abs/2502.03933) <br> *Jai Bardhan et al.* <br> Preprint 2025 | **Application:** JetClass pretraining, jet classification, top tagging, and quark/gluon discrimination. | [Paper](https://arxiv.org/abs/2502.03933) <br> [Project](https://hep-jepa.github.io/) |

#### Graphs, Time Series, and Physical Fields

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Graph-Level Representation Learning with Joint-Embedding Predictive Architectures**](https://arxiv.org/abs/2309.16014) <br> *Geri Skenderi et al.* <br> TMLR; preprint first released 2023 | **Graph-JEPA:** Graph representation learning with latent prediction. | [Paper](https://arxiv.org/abs/2309.16014) |
| [**MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction**](https://arxiv.org/abs/2602.04643) <br> *Yanan He et al.* <br> Preprint 2026 | **Application:** Multi-resolution objectives and a soft codebook for anomaly early warning. | [Paper](https://arxiv.org/abs/2602.04643) |
| [**ScaleAware-JEPA: Latent Representation for Discovery in Multiscale Physical Fields**](https://arxiv.org/abs/2606.29723) <br> *Guang-Xing Li* <br> Preprint 2026 | **Application:** Learns scale-aware latent coordinates and structural atlases for MHD turbulence, molecular gas, and urban nighttime-light fields. | [Paper](https://arxiv.org/abs/2606.29723) <br> [GitHub](https://github.com/gxli/SA-JEPA) |

#### Remote Sensing, Audio, and Language

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities**](https://arxiv.org/abs/2412.14123) <br> *Guillaume Astruc et al.* <br> CVPR 2025 | **Application:** JEPA-based multimodal earth-observation representations for classification and segmentation across scales. | [Paper](https://arxiv.org/abs/2412.14123) |
| [**Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning**](https://arxiv.org/abs/2507.02915) <br> *Ludovic Tuncay et al.* <br> Preprint 2025 | **Application:** Masked spectrogram-embedding prediction, evaluated on speech, music, and environmental sound. | [Paper](https://arxiv.org/abs/2507.02915) |
| [**LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures**](https://arxiv.org/abs/2509.14252) <br> *Hai Huang, Yann LeCun, Randall Balestriero* <br> Preprint 2025 | **Application:** JEPA-style objectives for language-model pretraining and fine-tuning. | [Paper](https://arxiv.org/abs/2509.14252) |
| [**VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language**](https://arxiv.org/abs/2512.10942) <br> *Delong Chen et al.* <br> Preprint first released 2025 | **Application:** Predicts target-text embeddings; supports classification, retrieval, and VQA with selective text decoding. | [Paper](https://arxiv.org/abs/2512.10942) |

## Related Work in XAI Involving World Models

### Explicit Structure and Interpretable Model Design

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Learning Cognitive Maps from Transformer Representations for Efficient Planning in Partially Observed Environments**](https://proceedings.mlr.press/v235/dedieu24a.html) <br> *Antoine Dedieu et al.* <br> ICML 2024 | **TDB; structured design.** Discrete bottlenecks yield cognitive maps that can be queried by an external planner. | [Paper](https://proceedings.mlr.press/v235/dedieu24a.html) |
| [**Learning Causal Dynamics Models in Object-Oriented Environments**](https://proceedings.mlr.press/v235/yu24j.html) <br> *Zhongwei Yu, Jingqing Ruan, Dengpeng Xing* <br> ICML 2024 | **OOCDM; causal model design.** Shares causal dependencies and parameters across object classes and learns causal dynamics. | [Paper](https://proceedings.mlr.press/v235/yu24j.html) |
| [**Causal-JEPA: Learning World Models through Object-Level Latent Masking**](https://arxiv.org/abs/2602.11389) <br> *Heejeong Nam et al.* <br> ICML 2026 | **C-JEPA; JEPA model design.** Object-level latent masking encourages context-dependent relational prediction. | [Paper](https://arxiv.org/abs/2602.11389) |
| [**Dyn-O: Building Structured World Models with Object-Centric Representations**](https://arxiv.org/abs/2507.03298) <br> *Zizhao Wang et al.* <br> Preprint 2025 | **Object-centric dynamics.** Separates object features into dynamics-aware and dynamics-agnostic components and enables imagined-trajectory manipulation. | [Paper](https://arxiv.org/abs/2507.03298) |
| [**PoE-World: Compositional World Modeling with Products of Programmatic Experts**](https://arxiv.org/abs/2505.10819) <br> *Wasu Top Piriyakulkij et al.* <br> Preprint 2025 | **Programmatic models.** Represents world dynamics through products of LLM-synthesized programmatic experts. | [Paper](https://arxiv.org/abs/2505.10819) <br> [Project](https://topwasu.github.io/poe-world) |

### World Representations in Transformers and LLMs

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Transformers Use Causal World Models in Maze-Solving Tasks**](https://arxiv.org/abs/2412.11867) <br> *Alex F. Spies et al.* <br> Preprint 2024; ICLR World Models Workshop 2025 | **SAEs, circuits, and feature intervention.** Combines sparse-feature discovery and attention-pattern analysis, then tests causal roles of maze features. | [Paper](https://arxiv.org/abs/2412.11867) |
| [**A Causal World Model Underlying Next Token Prediction: Exploring GPT in a Controlled Environment**](https://proceedings.mlr.press/v267/yehezkel-rohekar25a.html) <br> *Raanan Yehezkel Rohekar et al.* <br> ICML 2025 | **Causal structure learning.** Derives a causal interpretation of GPT attention and a confidence score, evaluated in Othello and Chess. | [Paper](https://proceedings.mlr.press/v267/yehezkel-rohekar25a.html) |
| [**Linear Spatial World Models Emerge in Large Language Models**](https://arxiv.org/abs/2506.02996) <br> *Matthieu Tehenan et al.* <br> Preprint 2025 | **Spatial probing and interventions.** Decodes object positions from contextual embeddings and tests geometric consistency and functional use. | [Paper](https://arxiv.org/abs/2506.02996) |
| [**Revisiting the Othello World Model Hypothesis**](https://arxiv.org/abs/2503.04421) <br> *Yifei Yuan, Anders Søgaard* <br> Preprint 2025 | **Board-state grounding.** Broadens probing across language-model families to examine learned Othello-board representations. | [Paper](https://arxiv.org/abs/2503.04421) |
| [**When Do Neural Networks Learn World Models?**](https://proceedings.mlr.press/v267/zhang25j.html) <br> *Tianren Zhang, Guanyu Chen, Feng Chen* <br> ICML 2025 | **Theoretical context.** Studies recovery of latent data-generating variables under architectural and low-degree assumptions. | [Paper](https://proceedings.mlr.press/v267/zhang25j.html) |
| [**General Agents Need World Models**](https://proceedings.mlr.press/v267/richens25a.html) <br> *Jonathan Richens, Tom Everitt, David Abel* <br> ICML 2025 | **Theoretical context.** Establishes conditions under which goal-directed generalization requires an extractable predictive model. | [Paper](https://proceedings.mlr.press/v267/richens25a.html) |

### World Models for Explaining Agents

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Explainable Reinforcement Learning via a Causal World Model**](https://arxiv.org/abs/2305.02749) <br> *Zhongwei Yu, Jingqing Ruan, Dengpeng Xing* <br> Preprint first released 2023 | Learns causal dynamics and explains long-term action effects through chains connecting actions, state variables, and rewards. | [Paper](https://arxiv.org/abs/2305.02749) |
| [**Explainable Reinforcement Learning Agents Using World Models**](https://arxiv.org/abs/2505.08073) <br> *Madhuri Singh, Amal Alabdulkarim, Gennie Mansi, Mark O. Riedl* <br> Preprint 2025 | Uses forward and reverse world models to construct counterfactual trajectories and states that would make an agent prefer an alternative action. | [Paper](https://arxiv.org/abs/2505.08073) |

### Transferable Methods and Behavioral Evaluation

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**On Mechanistic Knowledge Localization in Text-to-Image Generative Models**](https://proceedings.mlr.press/v235/basu24b.html) <br> *Samyadeep Basu et al.* <br> ICML 2024 | **LocoGen / LocoEdit; method transfer.** Localizes visual-attribute computations through cross-attention interventions and enables editing. | [Paper](https://proceedings.mlr.press/v235/basu24b.html) |
| [**DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders**](https://arxiv.org/abs/2602.05859) <br> *Xu Wang et al.* <br> Preprint 2026 | **Method transfer.** Studies sparse features and interventions across diffusion-language denoising steps. | [Paper](https://arxiv.org/abs/2602.05859) |
| [**How Far Is Video Generation from World Model: A Physical Law Perspective**](https://proceedings.mlr.press/v267/kang25g.html) <br> *Bingyi Kang et al.* <br> ICML 2025 | **Behavioral evaluation.** Tests video predictors under in-distribution, out-of-distribution, and compositional physical scenarios. | [Paper](https://proceedings.mlr.press/v267/kang25g.html) |
| [**WorldModelBench: Judging Video Generation Models as World Models**](https://arxiv.org/abs/2502.20694) <br> *Dacheng Li et al.* <br> Preprint 2025 | **Behavioral benchmark.** Evaluates instruction following and physics adherence using human labels and a learned judge. | [Paper](https://arxiv.org/abs/2502.20694) <br> [Project](https://worldmodelbench-team.github.io/) |

## Interpreting Learned World Models

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [![GitHub stars](https://img.shields.io/github/stars/facebookresearch/vjepa2.svg?style=social&label=Star)](https://github.com/facebookresearch/vjepa2) <br> [**V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**](https://arxiv.org/abs/2506.09985) <br> *Mido Assran et al.* <br> Preprint 2025 <br> **V-JEPA 2 Probing** | **Role: JEPA analysis.** Frozen-representation evaluations establish accessible appearance, motion, and action information. | [Paper](https://arxiv.org/abs/2506.09985) <br> [GitHub](https://github.com/facebookresearch/vjepa2) |
| [**Probing the Latent World: Emergent Discrete Symbols and Physical Structure in Latent Representations**](https://arxiv.org/abs/2603.20327) <br> *Liu Hung Ming* <br> Preprint 2026 <br> **AIM Probing** | **Role: JEPA analysis.** AI Mother Tongue (AIM) passively quantizes frozen V-JEPA 2 latents into vocabulary-free symbol sequences; category contrasts examine grasp angle, object geometry, and temporal motion structure. | [Paper](https://arxiv.org/abs/2603.20327) <br> [Author-linked GitHub](https://github.com/cyrilliu1974/JEPA) |
| [![GitHub stars](https://img.shields.io/github/stars/Bhavith-Chandra/WorldModelLens.svg?style=social&label=Star)](https://github.com/Bhavith-Chandra/WorldModelLens) <br> [**One Lens, Many Worlds: A Capability-Typed Interface for World-Model Interpretability**](https://arxiv.org/abs/2606.09936) <br> *Bhavith Chandra Challagundla et al.* <br> Preprint 2026 <br> **WorldModelLens** | **Role: JEPA + non-JEPA infrastructure.** A capability-typed interface supports activation access, caching, probing, attribution, sparse-feature analysis, patching, and intervention replay. The paper's main empirical demonstration is on I-JEPA. | [Paper](https://arxiv.org/abs/2606.09936) <br> [GitHub](https://github.com/Bhavith-Chandra/WorldModelLens) |
| [**Steering Robustness into World Action Models via Mechanistic Interpretability and Optimal Control**](https://arxiv.org/abs/2607.14943) <br> *Jihoon Hong, Julian Skifstad, Qiyue Dai, Alice Chan, Glen Chou* <br> Preprint 2026 <br> **WAM Steering** | **Role: non-JEPA transferable method.** Analyzes robustness-related activation geometry in Cosmos-Policy, DiT4DiT, and LingBot-VA, then uses contrastive directions and World-Action LQR (WA-LQR) for feedback steering. | [Paper](https://arxiv.org/abs/2607.14943) |
