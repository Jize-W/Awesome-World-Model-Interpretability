# Awesome World Model Interpretability

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Focus](https://img.shields.io/badge/Focus-JEPA%20%26%20Latent%20Worlds-blue)
![Scope](https://img.shields.io/badge/Scope-Curated%20Perspective-green)

A curated reading list on **world model interpretability**, with an emphasis on **Joint-Embedding Predictive Architectures (JEPAs)** and the internal worlds learned in latent space.

This repository accompanies the position paper **“World Model Interpretability: From Decoding Representations to Understanding Latent Worlds.”** It brings together representative papers, transferable tools, and open questions. It is a selective perspective rather than an exhaustive survey.

> **Our position:** JEPA interpretability should move beyond decoding latent representations toward understanding, verifying, and acting on the model's internal world. Recovering a variable from a representation does not establish how the predictor uses it.

The four literature categories below follow **Figure 1** of the position paper. The four representative entries in **Table 1** are collected in [Interpreting Learned World Models](#interpreting-learned-world-models). The other categories provide methodological foundations, model context, or adjacent evidence; they should not be read as a larger body of direct JEPA interpretability results.

Contributions and corrections are welcome through issues and pull requests.

- [Scope and Reading Guide](#scope-and-reading-guide)
- [General XAI Toolbox](#general-xai-toolbox)
- [Latent Space World Models](#latent-space-world-models)
  - [General Models and Predictive Architectures](#general-models-and-predictive-architectures)
  - [Downstream JEPA Applications](#downstream-jepa-applications)
- [Related Work in XAI Involving World Models](#related-work-in-xai-involving-world-models)
  - [Explicit Structure and Interpretable Model Design](#explicit-structure-and-interpretable-model-design)
  - [World Representations in Transformers and LLMs](#world-representations-in-transformers-and-llms)
  - [World Models for Explaining Agents](#world-models-for-explaining-agents)
  - [Transferable Methods and Behavioral Evaluation](#transferable-methods-and-behavioral-evaluation)
- [Interpreting Learned World Models](#interpreting-learned-world-models)
- [Perspective: Ground, Structure, Verify, Act](#perspective-ground-structure-verify-act)
- [Contributing](#contributing)

## Scope and Reading Guide

We use **latent world** to mean the internal variables, relations, and dynamics realized by a predictive model. For an action-conditioned JEPA, the object of interpretation is the encoder and predictor together: `z_t = E(x_t)` and `ẑ_(t+1) = P(z_t, a_t)`. Interpreting the encoder alone leaves the learned transition mechanism unresolved.

| Category | What it contributes | Relationship to JEPA interpretability |
| :--- | :--- | :--- |
| **General XAI Toolbox** | Attribution, concepts, probes, sparse features, circuits, causal interventions, and steering. | Transferable methods; application to JEPA predictive dynamics still needs validation. |
| **Latent Space World Models** | General predictive architectures and domain-specific JEPA applications. | Models to interpret and motivation for interpretation; task performance alone is not mechanistic evidence. |
| **Related Work in XAI Involving World Models** | Structured model design, world representations in sequence models, and world models used to explain agents. | Adjacent or weakly related work with useful ideas, but different primary research objectives. |
| **Interpreting Learned World Models** | Analysis or intervention targeting the internal representations and mechanisms of learned world models. | The four representative entries in the position paper: two JEPA analyses, one infrastructure contribution, and one non-JEPA transferable method. |

**Reading the tables.** Each entry provides a title, authors, publication information, a short introduction, and useful links, following the layout of [Awesome-Anything](https://github.com/VainF/Awesome-Anything). Author lists are shortened with *et al.* for readability. A paper-only link does not imply that code is unavailable. Publication labels follow the linked source; preprints are explicitly labeled. The interpretation questions and relevance judgments are our perspective, not claims made by every cited paper.

## General XAI Toolbox

These methods provide building blocks for interpreting learned predictors. Their inclusion does **not** establish that they have already explained a JEPA latent world. In particular, attribution to an input region, recovery of a concept, and causal participation in a multi-step rollout are different claims.

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Axiomatic Attribution for Deep Networks**](https://arxiv.org/abs/1703.01365) <br> *Mukund Sundararajan, Ankur Taly, Qiqi Yan* <br> ICML 2017 | **Integrated Gradients.** Attributes a chosen output to input features. For JEPA, the target could be a latent prediction score or a grounded readout; interpreting an arbitrary latent coordinate requires care. | [Paper](https://arxiv.org/abs/1703.01365) |
| [**“Why Should I Trust You?”: Explaining the Predictions of Any Classifier**](https://arxiv.org/abs/1602.04938) <br> *Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin* <br> KDD 2016 | **LIME.** Fits a local surrogate around a prediction. A possible diagnostic for downstream JEPA heads, with sensitivity to the perturbation distribution and the chosen explanation target. | [Paper](https://arxiv.org/abs/1602.04938) |
| [**A Unified Approach to Interpreting Model Predictions**](https://arxiv.org/abs/1705.07874) <br> *Scott Lundberg, Su-In Lee* <br> NeurIPS 2017 | **SHAP.** Provides additive feature attributions. Useful for input/output diagnostics, but does not by itself recover latent relations or predictive dynamics. | [Paper](https://arxiv.org/abs/1705.07874) |
| [**Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR**](https://arxiv.org/abs/1711.00399) <br> *Sandra Wachter, Brent Mittelstadt, Chris Russell* <br> Preprint 2017 | **Counterfactual explanations.** Asks which input changes would alter a decision. For world models, useful counterfactuals should respect feasible states, actions, and temporal evolution. | [Paper](https://arxiv.org/abs/1711.00399) |
| [**Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)**](https://arxiv.org/abs/1711.11279) <br> *Been Kim et al.* <br> ICML 2018 | **Concept methods.** Connects activations to human-defined concepts and measures directional sensitivity. A starting point for grounding objects, motion, contact, or task stages. | [Paper](https://arxiv.org/abs/1711.11279) |
| [**Understanding Intermediate Layers Using Linear Classifier Probes**](https://arxiv.org/abs/1610.01644) <br> *Guillaume Alain, Yoshua Bengio* <br> Preprint 2016 | **Probing.** Tests whether information is linearly recoverable from hidden representations. Supports representational accessibility, which should be distinguished from causal use by the predictor. | [Paper](https://arxiv.org/abs/1610.01644) |
| [**Sparse Autoencoders Find Highly Interpretable Features in Language Models**](https://arxiv.org/abs/2309.08600) <br> *Hoagy Cunningham et al.* <br> Preprint 2023; ICLR 2024 version | **Sparse features.** Decomposes activations into an overcomplete set of sparsely active features. A potential way to identify distributed latent factors; feature meaning, fidelity, and temporal stability need separate tests. | [Paper](https://arxiv.org/abs/2309.08600) |
| [**Zoom In: An Introduction to Circuits**](https://distill.pub/2020/circuits/zoom-in/) <br> *Chris Olah et al.* <br> Distill 2020 | **Circuit analysis.** Studies how internal features combine through network connections. Motivates tracing computations that connect grounded world variables to predicted states. | [Article](https://distill.pub/2020/circuits/zoom-in/) |
| [**Locating and Editing Factual Associations in GPT**](https://arxiv.org/abs/2202.05262) <br> *Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov* <br> NeurIPS 2022 | **Causal tracing and editing.** Uses activation restoration to locate influential computations, followed by targeted weight edits. A methodological precedent for Verify and Act, demonstrated on language-model factual recall. | [Paper](https://arxiv.org/abs/2202.05262) <br> [Project](https://rome.baulab.info/) |
| [**Steering Language Models with Activation Engineering**](https://arxiv.org/abs/2308.10248) <br> *Alexander Matt Turner et al.* <br> Preprint 2023 | **Activation steering / ActAdd.** Adds contrastive activation directions during inference. Transfer to JEPA requires checking that edits preserve plausible latent states and have the intended effects on prediction or control. | [Paper](https://arxiv.org/abs/2308.10248) |

## Latent Space World Models

This category describes **what we may want to interpret**. It includes established latent dynamics models and the broader JEPA family. Some JEPAs learn image, graph, or cross-modal representations rather than explicit temporal world dynamics; we retain them as architectural and application context without equating all latent prediction with a full world model.

### General Models and Predictive Architectures

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**World Models**](https://arxiv.org/abs/1803.10122) <br> *David Ha, Jürgen Schmidhuber* <br> 2018 | **Non-JEPA foundation.** Learns compressed spatial representations and recurrent dynamics for control. Establishes the latent-model setting; visually inspectable reconstructions do not alone explain the predictive mechanism. | [Paper](https://arxiv.org/abs/1803.10122) <br> [Project](https://worldmodels.github.io/) |
| [**Mastering Diverse Control Tasks through World Models**](https://www.nature.com/articles/s41586-025-08744-2) <br> *Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap* <br> Nature 2025 | **DreamerV3; non-JEPA.** Learns latent dynamics and optimizes behavior through imagined futures. Provides a neighboring model family for comparing representations, rollouts, and intervention tools. | [Paper](https://www.nature.com/articles/s41586-025-08744-2) |
| [**Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**](https://arxiv.org/abs/2301.08243) <br> *Mahmoud Assran et al.* <br> CVPR 2023 | **I-JEPA.** Predicts masked image-region embeddings rather than pixels. A foundational JEPA representation learner; it does not itself model action-conditioned temporal evolution. | [Paper](https://arxiv.org/abs/2301.08243) <br> [GitHub](https://github.com/facebookresearch/ijepa) |
| [**MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features**](https://arxiv.org/abs/2307.12698) <br> *Adrien Bardes, Jean Ponce, Yann LeCun* <br> Preprint 2023 | **Motion and content.** Jointly learns visual content and motion features, evaluated on optical flow and segmentation. Motivates asking how motion information is organized and used, beyond its downstream accessibility. | [Paper](https://arxiv.org/abs/2307.12698) |
| [**Revisiting Feature Prediction for Learning Visual Representations from Video**](https://arxiv.org/abs/2404.08471) <br> *Adrien Bardes et al.* <br> TMLR 2024 | **V-JEPA.** Learns video representations through masked feature prediction. Frozen evaluations reveal useful appearance and motion content, while leaving predictive mechanisms to be investigated. | [Paper](https://arxiv.org/abs/2404.08471) |
| [**V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**](https://arxiv.org/abs/2506.09985) <br> *Mido Assran et al.* <br> Preprint 2025 | **Video understanding and action-conditioned control.** V-JEPA 2-AC extends the video representation to robot-action-conditioned prediction and planning. Its probing analyses are discussed separately in the direct interpretation category. | [Paper](https://arxiv.org/abs/2506.09985) <br> [GitHub](https://github.com/facebookresearch/vjepa2) |
| [**seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models**](https://arxiv.org/abs/2505.03176) <br> *Hafez Ghaemi, Eilif Muller, Shahab Bakhtiari* <br> NeurIPS 2025 | **Invariant and equivariant representations.** Processes observation–transformation sequences and separates aggregate invariant representations from equivariant per-view representations. Relevant to studying which latent factors remain stable or transform with actions. | [Paper](https://arxiv.org/abs/2505.03176) |
| [**LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**](https://arxiv.org/abs/2603.19312) <br> *Lucas Maes et al.* <br> Preprint 2026 | **LeWM.** Trains action-conditioned latent prediction from pixels with a prediction loss and Gaussian-distribution regularization. Physical-quantity probes and surprise evaluation provide additional representation and behavioral evidence; the main contribution is world-model learning. | [Paper](https://arxiv.org/abs/2603.19312) |

### Downstream JEPA Applications

These representative applications demonstrate why an interpretable latent interface matters across domains. The **Question** in each entry is a proposed research question, not a result established by that application paper. Spatial or cross-modal prediction is kept distinct from learned temporal dynamics.

#### Robotics and Embodied Control

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**ACT-JEPA: Joint-Embedding Predictive Architecture Improves Policy Representation Learning**](https://arxiv.org/abs/2501.14622) <br> *Aleksandar Vujinovic, Aleksandar Kovacevic* <br> Preprint 2025 | **Application:** Imitation learning through joint action-sequence and latent-observation prediction. <br> **Question:** Which parts of the predictive representation support action selection, and which merely track appearance? | [Paper](https://arxiv.org/abs/2501.14622) <br> [Project](https://act-jepa.github.io/) |
| [**VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model**](https://arxiv.org/abs/2602.10098) <br> *Jingwen Sun et al.* <br> Preprint 2026 | **Application:** Latent state-prediction pretraining followed by action-head fine-tuning; manipulation on LIBERO, LIBERO-Plus, SimplerEnv, and real robots. <br> **Question:** Does improved robustness arise from control-relevant transitions, and which internal features mediate it? | [Paper](https://arxiv.org/abs/2602.10098) |
| [**Is Forward Prediction Enough? Physical State Grounding for JEPA World Models**](https://arxiv.org/abs/2608.06799) <br> *Haodong Yan et al.* <br> Preprint 2026 | **PSG-JEPA:** Adds training objectives for proprioceptive state and multi-horizon joint-angle changes, evaluated through probes, planning, and policy learning. <br> **Question:** Does the predictor causally use the grounded variables? This is closely related model design, rather than one of the four Table 1 entries. | [Paper](https://arxiv.org/abs/2608.06799) |

V-JEPA 2-AC, listed above, is another central robotics example: its DROID-based training and image-goal planning motivate interpreting object state, gripper motion, contact, and action-conditioned transitions.

#### 3D Perception and Autonomous Driving

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Point-JEPA: A Joint Embedding Predictive Architecture for Self-Supervised Learning on Point Cloud**](https://arxiv.org/abs/2404.16432) <br> *Ayumu Saito, Prachi Kudeshia, Jiju Poovvancheri* <br> Preprint 2024 | **Application:** Predicts point-cloud patch embeddings for 3D representation learning. <br> **Question:** Which latent factors correspond to geometry, and how sensitive are they to sampling density or nuisance transformations? | [Paper](https://arxiv.org/abs/2404.16432) |
| [**3D-JEPA: A Joint Embedding Predictive Architecture for 3D Self-Supervised Representation Learning**](https://arxiv.org/abs/2409.15803) <br> *Naiwen Hu et al.* <br> Preprint 2024 | **Application:** Context-to-target latent prediction for 3D point-cloud representations. <br> **Question:** Are spatial relationships preserved beyond what a classification head can decode? This paper is distinct from the 3D-JEPA algorithm introduced in LOCATE 3D. | [Paper](https://arxiv.org/abs/2409.15803) |
| [**LOCATE 3D: Real-World Object Localization via Self-Supervised Learning in 3D**](https://proceedings.mlr.press/v267/mcvay25a.html) <br> *Paul McVay, Sergio Arnaud et al.* <br> ICML 2025 | **Application:** Its 3D-JEPA pretraining supports language-guided object localization from RGB-D observations, followed by a language-conditioned decoder. <br> **Question:** How does the representation organize object identity, position, and relations such as “between” or “closest”? | [Paper](https://proceedings.mlr.press/v267/mcvay25a.html) <br> [GitHub](https://github.com/facebookresearch/locate-3d) |
| [**Self-Supervised Representation Learning with Joint Embedding Predictive Architecture for Automotive LiDAR Object Detection**](https://arxiv.org/abs/2501.04969) <br> *Haoran Zhu et al.* <br> AAAI 2026 | **AD-L-JEPA:** Predicts BEV embeddings for LiDAR pretraining; evaluated on KITTI3D, Waymo, and ONCE object detection. <br> **Question:** Do detections rely on meaningful geometry or on sensor-density and dataset shortcuts? This is perception pretraining, not evidence of learned driving dynamics. | [Paper](https://arxiv.org/abs/2501.04969) <br> [Project](https://ad-l-jepa.github.io/) |

#### Medicine and Physiological Sensing

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**A Vision Foundation Model for Cataract Surgery Using Joint-Embedding Predictive Architecture**](https://proceedings.mlr.press/v301/shah26b.html) <br> *Nisarg A. Shah et al.* <br> MIDL; proceedings 2026 | **JHU-VPT(JEPA):** Surgical-video pretraining for step recognition, feedback, and skill assessment. <br> **Question:** Are predictions supported by instrument–anatomy interactions and procedure progression, rather than surgeon, site, or camera cues? | [Paper](https://proceedings.mlr.press/v301/shah26b.html) |
| [**US-JEPA: A Joint Embedding Predictive Architecture for Medical Ultrasound**](https://arxiv.org/abs/2602.19322) <br> *Ashwath Radhachandran et al.* <br> Preprint 2026 | **Application:** Frozen-teacher latent prediction for ultrasound representations, evaluated on UltraBench classification tasks. <br> **Question:** Which features reflect anatomy and pathology, and which reflect speckle or acquisition conditions? | [Paper](https://arxiv.org/abs/2602.19322) |
| [**mmJEPA-ECG: Cross-Posture Robust Contactless Electrocardiogram Monitoring via Millimeter Wave Radar Sensing**](https://ojs.aaai.org/index.php/AAAI/article/download/41242/45203) <br> *Ziyang Liu et al.* <br> AAAI 2026 | **Application:** Physiology-guided JEPA pretraining followed by diffusion-based ECG reconstruction from radar. <br> **Question:** Are latent factors tied to cardiac dynamics rather than posture artifacts, and which component explains reconstruction robustness? | [Paper](https://ojs.aaai.org/index.php/AAAI/article/download/41242/45203) |

#### Genomics, Cells, and Collider Physics

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**JEPA-DNA: Grounding Genomic Foundation Models through Joint-Embedding Predictive Architectures**](https://arxiv.org/abs/2602.17162) <br> *Ariel Larey et al.* <br> Preprint 2026 | **Application:** Combines generative objectives with prediction of genomic-segment embeddings; supervised and zero-shot genomic tasks. <br> **Question:** Does latent organization reflect regulatory and functional context beyond local nucleotide statistics? “Grounding” in the title does not establish causal interpretation. | [Paper](https://arxiv.org/abs/2602.17162) |
| [**Cell-JEPA: Latent Representation Learning for Single-Cell Transcriptomics**](https://arxiv.org/abs/2602.02093) <br> *Ali ElSheikh et al.* <br> Preprint 2026 | **Application:** Learns cell representations from partial gene observations; cell-type transfer and perturbation evaluation. <br> **Question:** Which latent factors reflect cellular programs rather than dropout or technical noise, and do they explain perturbation effects? | [Paper](https://arxiv.org/abs/2602.02093) |
| [**HEP-JEPA: A Foundation Model for Collider Physics Using Joint Embedding Predictive Architecture**](https://arxiv.org/abs/2502.03933) <br> *Jai Bardhan et al.* <br> Preprint 2025 | **Application:** JetClass pretraining, jet classification, top tagging, and quark/gluon discrimination. <br> **Question:** Which latent factors correspond to particle kinematics and jet substructure, and do classifiers use physically meaningful relations? | [Paper](https://arxiv.org/abs/2502.03933) <br> [Project](https://hep-jepa.github.io/) |

#### Graphs, Time Series, and Physical Fields

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Graph-Level Representation Learning with Joint-Embedding Predictive Architectures**](https://arxiv.org/abs/2309.16014) <br> *Geri Skenderi et al.* <br> TMLR; preprint first released 2023 | **Graph-JEPA:** Graph representation learning with latent prediction. <br> **Question:** Do latent factors preserve interpretable graph structure and relationships? Predicting graph embeddings is distinct from modeling graph evolution over time. | [Paper](https://arxiv.org/abs/2309.16014) |
| [**MTS-JEPA: Multi-Resolution Joint-Embedding Predictive Architecture for Time-Series Anomaly Prediction**](https://arxiv.org/abs/2602.04643) <br> *Yanan He et al.* <br> Preprint 2026 | **Application:** Multi-resolution objectives and a soft codebook for anomaly early warning. <br> **Question:** Do codes correspond to system regimes and precursor events, and which transitions cause an anomaly forecast? | [Paper](https://arxiv.org/abs/2602.04643) |
| [**ScaleAware-JEPA: Latent Representation for Discovery in Multiscale Physical Fields**](https://arxiv.org/abs/2606.29723) <br> *Guang-Xing Li* <br> Preprint 2026 | **Application:** Learns scale-aware latent coordinates and structural atlases for MHD turbulence, molecular gas, and urban nighttime-light fields. <br> **Question:** Which physical structures organize the atlas, and how would that correspondence be verified beyond visual coherence? | [Paper](https://arxiv.org/abs/2606.29723) <br> [GitHub](https://github.com/gxli/SA-JEPA) |

#### Remote Sensing, Audio, and Language

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities**](https://arxiv.org/abs/2412.14123) <br> *Guillaume Astruc et al.* <br> CVPR 2025 | **Application:** JEPA-based multimodal earth-observation representations for classification and segmentation across scales. <br> **Question:** Do features align with land-cover and environmental structure across sensors and resolutions? | [Paper](https://arxiv.org/abs/2412.14123) |
| [**Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning**](https://arxiv.org/abs/2507.02915) <br> *Ludovic Tuncay et al.* <br> Preprint 2025 | **Application:** Masked spectrogram-embedding prediction, evaluated on speech, music, and environmental sound. <br> **Question:** Which factors reflect sound sources and temporal events rather than recording conditions? | [Paper](https://arxiv.org/abs/2507.02915) |
| [**LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures**](https://arxiv.org/abs/2509.14252) <br> *Hai Huang, Yann LeCun, Randall Balestriero* <br> Preprint 2025 | **Application:** JEPA-style objectives for language-model pretraining and fine-tuning. <br> **Question:** What semantic organization does the objective induce? Improved language-task accuracy alone does not identify an internal physical world model. | [Paper](https://arxiv.org/abs/2509.14252) |
| [**VL-JEPA: Joint Embedding Predictive Architecture for Vision-Language**](https://arxiv.org/abs/2512.10942) <br> *Delong Chen et al.* <br> Preprint first released 2025 | **Application:** Predicts target-text embeddings; supports classification, retrieval, and VQA with selective text decoding. <br> **Question:** Which visual and linguistic concepts are aligned, and does the resulting interface faithfully describe the visual representation? | [Paper](https://arxiv.org/abs/2512.10942) |

## Related Work in XAI Involving World Models

This category distinguishes several kinds of adjacency: building models with explicit structure, interpreting emergent world representations in Transformers, using a world model to explain an agent, and transferring methods from other architectures. These contributions can be valuable without being direct analyses of JEPA predictive mechanisms.

### Explicit Structure and Interpretable Model Design

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Learning Cognitive Maps from Transformer Representations for Efficient Planning in Partially Observed Environments**](https://proceedings.mlr.press/v235/dedieu24a.html) <br> *Antoine Dedieu et al.* <br> ICML 2024 | **TDB; structured design.** Discrete bottlenecks yield cognitive maps that can be queried by an external planner. A precedent for extracting explicit latent-world structure, enabled by an architectural bottleneck rather than a general post-hoc JEPA method. | [Paper](https://proceedings.mlr.press/v235/dedieu24a.html) |
| [**Learning Causal Dynamics Models in Object-Oriented Environments**](https://proceedings.mlr.press/v235/yu24j.html) <br> *Zhongwei Yu, Jingqing Ruan, Dengpeng Xing* <br> ICML 2024 | **OOCDM; causal model design.** Shares causal dependencies and parameters across object classes and learns causal dynamics. Offers a structured modeling reference, rather than analysis of a pretrained, unstructured JEPA latent space. | [Paper](https://proceedings.mlr.press/v235/yu24j.html) |
| [**Causal-JEPA: Learning World Models through Object-Level Latent Masking**](https://arxiv.org/abs/2602.11389) <br> *Heejeong Nam et al.* <br> ICML 2026 | **C-JEPA; JEPA model design.** Object-level latent masking encourages context-dependent relational prediction. Closely related through object-centric structure and counterfactual-like training queries; training-time masking should be distinguished from post-hoc causal verification of a learned predictor. | [Paper](https://arxiv.org/abs/2602.11389) |
| [**Dyn-O: Building Structured World Models with Object-Centric Representations**](https://arxiv.org/abs/2507.03298) <br> *Zizhao Wang et al.* <br> Preprint 2025 | **Object-centric dynamics.** Separates object features into dynamics-aware and dynamics-agnostic components and enables imagined-trajectory manipulation. Useful architectural context for compositional structure; outside the JEPA-focused Table 1 selection. | [Paper](https://arxiv.org/abs/2507.03298) |
| [**PoE-World: Compositional World Modeling with Products of Programmatic Experts**](https://arxiv.org/abs/2505.10819) <br> *Wasu Top Piriyakulkij et al.* <br> Preprint 2025 | **Programmatic models.** Represents world dynamics through products of LLM-synthesized programmatic experts. Provides an explicitly inspectable alternative to distributed neural latents, with a different representation and learning setup. | [Paper](https://arxiv.org/abs/2505.10819) <br> [Project](https://topwasu.github.io/poe-world) |

### World Representations in Transformers and LLMs

Here, “world model” is often an **emergent internal representation** in a sequence predictor. The studies provide probing and intervention precedents, but do not directly interpret a JEPA world model trained for physical prediction or control.

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Transformers Use Causal World Models in Maze-Solving Tasks**](https://arxiv.org/abs/2412.11867) <br> *Alex F. Spies et al.* <br> Preprint 2024; ICLR World Models Workshop 2025 | **SAEs, circuits, and feature intervention.** Combines sparse-feature discovery and attention-pattern analysis, then tests causal roles of maze features. A useful precedent for separating feature accessibility from functional use. | [Paper](https://arxiv.org/abs/2412.11867) |
| [**A Causal World Model Underlying Next Token Prediction: Exploring GPT in a Controlled Environment**](https://proceedings.mlr.press/v267/yehezkel-rohekar25a.html) <br> *Raanan Yehezkel Rohekar et al.* <br> ICML 2025 | **Causal structure learning.** Derives a causal interpretation of GPT attention and a confidence score, evaluated in Othello and Chess. Its assumptions and controlled sequence-model setting differ from JEPA latent dynamics. | [Paper](https://proceedings.mlr.press/v267/yehezkel-rohekar25a.html) |
| [**Linear Spatial World Models Emerge in Large Language Models**](https://arxiv.org/abs/2506.02996) <br> *Matthieu Tehenan et al.* <br> Preprint 2025 | **Spatial probing and interventions.** Decodes object positions from contextual embeddings and tests geometric consistency and functional use. Motivates grounding spatial variables and verifying their role, within a language-model setting. | [Paper](https://arxiv.org/abs/2506.02996) |
| [**Revisiting the Othello World Model Hypothesis**](https://arxiv.org/abs/2503.04421) <br> *Yifei Yuan, Anders Søgaard* <br> Preprint 2025 | **Board-state grounding.** Broadens probing across language-model families to examine learned Othello-board representations. Stronger representation evidence does not automatically characterize an action-conditioned physical predictor. | [Paper](https://arxiv.org/abs/2503.04421) |
| [**When Do Neural Networks Learn World Models?**](https://proceedings.mlr.press/v267/zhang25j.html) <br> *Tianren Zhang, Guanyu Chen, Feng Chen* <br> ICML 2025 | **Theoretical context.** Studies recovery of latent data-generating variables under architectural and low-degree assumptions. Helps frame identifiability and generalization questions, rather than supplying an empirical JEPA interpretation method. | [Paper](https://proceedings.mlr.press/v267/zhang25j.html) |
| [**General Agents Need World Models**](https://proceedings.mlr.press/v267/richens25a.html) <br> *Jonathan Richens, Tom Everitt, David Abel* <br> ICML 2025 | **Theoretical context.** Establishes conditions under which goal-directed generalization requires an extractable predictive model. Motivates studying internal models; it does not show that their variables are human-interpretable. | [Paper](https://proceedings.mlr.press/v267/richens25a.html) |

### World Models for Explaining Agents

The explanatory target in these papers is principally **agent behavior**. A world model acts as an explanation mechanism; interpreting the world model's own learned latent computation is a separate objective.

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**Explainable Reinforcement Learning via a Causal World Model**](https://arxiv.org/abs/2305.02749) <br> *Zhongwei Yu, Jingqing Ruan, Dengpeng Xing* <br> Preprint first released 2023 | Learns causal dynamics and explains long-term action effects through chains connecting actions, state variables, and rewards. A reference for temporally grounded explanations of policies. | [Paper](https://arxiv.org/abs/2305.02749) |
| [**Explainable Reinforcement Learning Agents Using World Models**](https://arxiv.org/abs/2505.08073) <br> *Madhuri Singh, Amal Alabdulkarim, Gennie Mansi, Mark O. Riedl* <br> Preprint 2025 | Uses forward and reverse world models to construct counterfactual trajectories and states that would make an agent prefer an alternative action. Explains the policy through modeled scenarios rather than exposing JEPA mechanisms. | [Paper](https://arxiv.org/abs/2505.08073) |

### Transferable Methods and Behavioral Evaluation

These are **weakly related methodological or evaluation references**. Text-to-image or diffusion-language interpretability should not be presented as direct world-model evidence. Behavioral physics tests reveal failures but do not alone identify their internal causes.

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [**On Mechanistic Knowledge Localization in Text-to-Image Generative Models**](https://proceedings.mlr.press/v235/basu24b.html) <br> *Samyadeep Basu et al.* <br> ICML 2024 | **LocoGen / LocoEdit; method transfer.** Localizes visual-attribute computations through cross-attention interventions and enables editing. Included in the Figure 1 landscape as a localization precedent; its target is image generation, not learned world dynamics. | [Paper](https://proceedings.mlr.press/v235/basu24b.html) |
| [**DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders**](https://arxiv.org/abs/2602.05859) <br> *Xu Wang et al.* <br> Preprint 2026 | **Method transfer.** Studies sparse features and interventions across diffusion-language denoising steps. Motivates analysis of iterative computation, without establishing that diffusion-language features correspond to a physical latent world. | [Paper](https://arxiv.org/abs/2602.05859) |
| [**How Far Is Video Generation from World Model: A Physical Law Perspective**](https://proceedings.mlr.press/v267/kang25g.html) <br> *Bingyi Kang et al.* <br> ICML 2025 | **Behavioral evaluation.** Tests video predictors under in-distribution, out-of-distribution, and compositional physical scenarios. Supplies controlled failure cases and motivation for distinguishing physical structure from shortcuts. | [Paper](https://proceedings.mlr.press/v267/kang25g.html) |
| [**WorldModelBench: Judging Video Generation Models as World Models**](https://arxiv.org/abs/2502.20694) <br> *Dacheng Li et al.* <br> Preprint 2025 | **Behavioral benchmark.** Evaluates instruction following and physics adherence using human labels and a learned judge. Complements internal analysis; output quality and physical plausibility are not themselves mechanistic explanations. | [Paper](https://arxiv.org/abs/2502.20694) <br> [Project](https://worldmodelbench-team.github.io/) |

## Interpreting Learned World Models

**Start here for the four representative entries in Table 1.** They have different roles: V-JEPA 2 probing and AIM analyze JEPA representations; WorldModelLens supplies infrastructure; WAM steering offers a transferable method from non-JEPA world action models. This selection does not claim that only four related papers exist, or that all four fully interpret JEPA dynamics.

| Title & Authors | Intro | Useful Links |
| :--- | :--- | :--- |
| [![GitHub stars](https://img.shields.io/github/stars/facebookresearch/vjepa2.svg?style=social&label=Star)](https://github.com/facebookresearch/vjepa2) <br> [**V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**](https://arxiv.org/abs/2506.09985) <br> *Mido Assran et al.* <br> Preprint 2025 <br> **Table 1: V-JEPA 2 Probing** | **Role: JEPA analysis.** Frozen-representation evaluations establish accessible appearance, motion, and action information. <br> **Perspective:** Provides partial grounding and structural evidence, but downstream decodability does not establish which variables the latent predictor causally uses. Robot planning capability is distinct from interpretation-guided intervention. | [Paper](https://arxiv.org/abs/2506.09985) <br> [GitHub](https://github.com/facebookresearch/vjepa2) |
| [**Probing the Latent World: Emergent Discrete Symbols and Physical Structure in Latent Representations**](https://arxiv.org/abs/2603.20327) <br> *Liu Hung Ming* <br> Preprint 2026 <br> **Table 1: AIM Probing** | **Role: JEPA analysis.** AI Mother Tongue (AIM) passively quantizes frozen V-JEPA 2 latents into vocabulary-free symbol sequences; category contrasts examine grasp angle, object geometry, and temporal motion structure. <br> **Perspective:** Symbol-distribution differences provide grounding evidence. Discretization and category association alone do not verify that the predictor uses those symbols or recover its transition mechanism. | [Paper](https://arxiv.org/abs/2603.20327) <br> [Author-linked GitHub](https://github.com/cyrilliu1974/JEPA) |
| [![GitHub stars](https://img.shields.io/github/stars/Bhavith-Chandra/WorldModelLens.svg?style=social&label=Star)](https://github.com/Bhavith-Chandra/WorldModelLens) <br> [**One Lens, Many Worlds: A Capability-Typed Interface for World-Model Interpretability**](https://arxiv.org/abs/2606.09936) <br> *Bhavith Chandra Challagundla et al.* <br> Preprint 2026 <br> **Table 1: WorldModelLens** | **Role: JEPA + non-JEPA infrastructure.** A capability-typed interface supports activation access, caching, probing, attribution, sparse-feature analysis, patching, and intervention replay. The paper's main empirical demonstration is on I-JEPA. <br> **Perspective:** Makes Ground and Verify investigations easier to implement; interface coverage across architectures should be distinguished from demonstrated discovery of their latent-world mechanisms. | [Paper](https://arxiv.org/abs/2606.09936) <br> [GitHub](https://github.com/Bhavith-Chandra/WorldModelLens) |
| [**Steering Robustness into World Action Models via Mechanistic Interpretability and Optimal Control**](https://arxiv.org/abs/2607.14943) <br> *Jihoon Hong, Julian Skifstad, Qiyue Dai, Alice Chan, Glen Chou* <br> Preprint 2026 <br> **Table 1: WAM Steering** | **Role: non-JEPA transferable method.** Analyzes robustness-related activation geometry in Cosmos-Policy, DiT4DiT, and LingBot-VA, then uses contrastive directions and World-Action LQR (WA-LQR) for feedback steering. <br> **Perspective:** Connects activation analysis to interventions and downstream robustness. Its architecture-dependent results motivate Act; they do not establish equivalent steerability in JEPAs. | [Paper](https://arxiv.org/abs/2607.14943) |

### Coverage in the Position Paper

The mapping below reproduces **Table 1's assessment**, rather than re-scoring the papers. **✓** = explicit coverage; **◐** = partial coverage; **—** = not addressed. Coverage refers to the position paper's four-stage blueprint, not a ranking of overall paper quality. For infrastructure, coverage records support for analysis rather than completion of a scientific explanation.

| Approach | Role | Model Scope | Ground | Structure | Verify | Act |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| V-JEPA 2 Probing | Method | JEPA | ◐ | ◐ | — | — |
| AIM Probing | Method | JEPA | ◐ | — | — | — |
| WorldModelLens | Infrastructure | JEPA + non-JEPA | ◐ | — | ◐ | — |
| WAM Steering | Transferable method | Non-JEPA | ◐ | ◐ | ◐ | ✓ |

**The gap:** The JEPA analyses primarily characterize accessible latent content. Infrastructure and non-JEPA steering suggest ways forward, while systematic recovery of JEPA relations, dynamics, causal dependencies, and interpretation-guided improvements remains an open research agenda.

## Perspective: Ground, Structure, Verify, Act

The literature categories describe **where work sits in the landscape**. The following stages describe **what evidence an interpretation provides**. They are separate organizing axes.

| Stage | Core Question | Evidence to Seek | What Is Insufficient on Its Own |
| :--- | :--- | :--- | :--- |
| **Ground** | What do latent factors correspond to in the external world? | Controlled correspondences with objects, position, velocity, geometry, contact, actions, events, or domain-specific variables; probe baselines and cross-context stability. | A readable label, visually appealing cluster, or unconstrained decoder. |
| **Structure** | How are these factors related, and how do they evolve? | Relational organization, transition rules, action dependence, and consistency across prediction horizons. | Independent decoding of several variables without analyzing their interactions or transitions. |
| **Verify** | Does the predictor causally rely on the proposed factors and mechanisms? | Controlled activation or factor interventions with appropriate baselines, plausible latent states, and checks for unintended changes. | Correlation, attention visualization, or a prediction change caused by an uncontrolled latent edit. |
| **Act** | Can verified understanding diagnose failures and improve behavior? | Targeted steering, editing, or training changes tied to a tested mechanism, with measured effects on prediction, robustness, planning, or control. | Task improvements whose connection to the proposed interpretation has not been established. |

For example, decoding **velocity** is a Ground result. Recovering how velocity and contact determine position changes moves toward Structure. Changing a velocity-related factor while preserving other relevant information and testing the predictor's response addresses Verify. Using that verified mechanism to correct a rollout or planning failure addresses Act.

The variables depend on the domain. Robot manipulation may require objects, gripper pose, contact, and action effects; surgical video may require instruments, anatomy, interactions, and procedural stages; genomic prediction may require regulatory context and functional relationships. A common framework need not impose a universal ontology or a one-neuron–one-concept correspondence.

### Open Research Directions

- **Build controlled testbeds.** Vary world factors independently, record known relations and dynamics, and evaluate explanations beyond probe accuracy.
- **Trace the predictive system.** Compare encoder states, predictor computations, action inputs, and multi-step rollouts to separate state, dynamics, interaction, and accumulated-error failures.
- **Validate latent interventions.** Check whether edits change the intended factor, preserve other relevant information, and remain consistent with plausible world states.
- **Test stability and shortcuts.** Compare interpretations across objects, viewpoints, backgrounds, environments, model scales, and training stages.
- **Connect analysis to useful action.** Test whether mechanism-based diagnosis and intervention improve prediction or planning, and report where transfer fails.

Decoding remains useful and can be sufficient for a representation-level question. Stronger claims about an internal predictive world require stronger evidence. Internal analysis should complement behavioral evaluation, robustness tests, and uncertainty assessment.

## Contributing

Please suggest additions with the **paper title, authors, publication year or venue, primary-source link, relevant category, and a short explanation of its connection to latent-world interpretability**. Include an author-linked code or project page when available.

Distinguish direct JEPA analysis, non-JEPA analysis, infrastructure, structured model design, and transferable tools. Claims about Ground, Structure, Verify, or Act should point to the supporting experiment or analysis. Model accuracy, an object-centric architecture, or an interpretable visualization alone should not be presented as full mechanistic understanding.

The list is curated from the position paper and its supporting research notes and downstream-application spreadsheet, with bibliographic details checked against linked primary sources. Format inspired by [Awesome-Anything](https://github.com/VainF/Awesome-Anything).
