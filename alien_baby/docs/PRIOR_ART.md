# Prior Art Survey: RL Locomotion, Motor Learning, and the Taylor Perception Theory

*April 2026 — companion to `GAP_ANALYSIS.md`*

---

## Why this doc exists

David's question, paraphrased: "We keep hitting promising results that fall apart, and another model just told us a basic architectural choice (the MirrorWrapper) was something we should have been doing from day one. Are we reinventing wheels? What has the field actually solved? Has anyone tested Taylor's theory another way?"

This document answers those three questions. Its companion, `GAP_ANALYSIS.md`, turns the same field-knowledge inward: which standard tools, MuJoCo features, and training practices we have been omitting that the literature treats as table-stakes.

## TL;DR

- **Touch-the-ball as a pure RL/locomotion task is solved many times over.** Standard recipe: SAC or PPO + dense shaped reward + symmetry priors + (when pixels are in the obs) a CNN encoder + domain randomization. MuJoCo Reacher converges in <100K steps. OpenAI's Shadow Hand solved a 24-DOF in-hand cube reorientation. We are not at the frontier of difficulty.
- **The legged-locomotion field moved fast through environment engineering, not new algorithms.** Hwangbo (ANYmal), Lee (rough terrain), Margolis (Mini Cheetah at 3.9 m/s) all used standard PPO. What made them work: better actuator models, domain randomization, curriculum, careful reward shaping.
- **A platform shift is available and cheap.** MuJoCo Playground (Jan 2025, JAX/MJX on GPU) gives 100×–1000× our current iteration speed using the *same* MuJoCo physics. Most of its tasks train in under 10 minutes on a single GPU.
- **Brain-grounded motor learning exists in pieces, not as a unified framework.** CPGs (Ijspeert), MyoSuite (muscle-actuated), active inference (Friston/Pezzulo), internal models (Wolpert/Kawato). None of them frame motor learning as *conditioning* in a sim, and none operationalize what Taylor specifically meant.
- **Taylor's Π has no computational implementation in the literature.** The closest living relatives — sensorimotor contingency theory (O'Regan & Noë), Sidman stimulus equivalence, active inference — share family resemblance but none asks Taylor's specific cross-modal interpenetration question via a load-bearing-vision test. The research framing is genuinely novel; the *plumbing* (touch-ball baseline) is not.

---

## 1. RL Locomotion: Where the Field Actually Is

### The standard benchmarks

MuJoCo locomotion environments (Ant, HalfCheetah, Humanoid, Hopper, Walker2d) have been the workhorse benchmark for continuous-control RL since roughly 2016. Three algorithms dominate the current literature:

- **SAC (Soft Actor-Critic)** [Haarnoja et al. 2018] is the de facto default for off-policy continuous control. It learns efficiently because entropy regularization prevents premature collapse to suboptimal policies. SAC routinely solves HalfCheetah to reward >10,000 in 1–3 million environment steps; Ant and Humanoid take 3–10 million steps depending on implementation.

- **TD3 (Twin Delayed DDPG)** [Fujimoto et al. 2018] is a close competitor for sample efficiency and often edges out SAC on deterministic tasks; slightly more brittle to hyperparameter choices.

- **PPO (Proximal Policy Optimization)** [Schulman et al. 2017] is on-policy and generally less sample-efficient than SAC/TD3 on locomotion, but it parallelizes cleanly, is easier to debug, and is the algorithm most often used in massively-parallel GPU pipelines (IsaacGym, Brax). PPO is the dominant algorithm in the sim-to-real legged-robot literature because those pipelines run tens of thousands of parallel environments where off-policy replay is less of an advantage.

A practical benchmark comparison [Atlantis Press 2022] found SAC and TD3 converge faster than PPO on standard MuJoCo tasks, but PPO closes the gap when parallel environments are available. On a single-GPU machine with 16–64 environments, all three are workable; the choice matters less than the environment design and reward signal.

**Sample counts you should expect:** A reach/touch task in a low-DOF setting (2–4 joints, stationary target, dense reward) should converge under 500K steps with SAC. If you are burning multiple million steps without clear progress, the reward signal or environment design is the issue, not algorithm choice. The standard recipe: SAC or TD3 + dense shaped reward + domain randomization. That's it.

### Sim-to-real quadruped and bipedal locomotion

The legged-robot community has taken this from "benchmark curiosity" to production hardware in roughly five years.

**Hwangbo et al. (2019)** — *Learning agile and dynamic motor skills for legged robots* [Science Robotics] — is the canonical entry point for sim-to-real quadruped work. The key insight was not a new algorithm but a better simulator: they trained a neural-network actuator model from real hardware data, dramatically reducing the sim-to-real gap for ANYmal's series elastic actuators. Policy training in sim then transferred zero-shot to the real robot. This paper established the template: (1) improve simulator fidelity, (2) add domain randomization, (3) transfer policy without fine-tuning.

**Lee et al. (2020)** — *Learning Quadrupedal Locomotion over Challenging Terrain* [Science Robotics] — extended this to rough terrain (mud, rubble, snow) using blind proprioceptive control. The controller generalizes zero-shot to terrains never seen in training because the policy learns to be robust to proprioceptive surprises rather than to model specific terrain types.

**Margolis et al. (2022/2023)** — *Rapid Locomotion via Reinforcement Learning* [IJRR, arXiv:2205.02824] — trained MIT Mini Cheetah to run at 3.9 m/s on grass, ice, and gravel, using an adaptive curriculum on velocity commands and online system identification for sim-to-real. This demonstrates that the reach of simulation-trained policies is now genuinely athletic, not just plodding locomotion.

**Cassie (Berkeley)** — Li, Cheng, Peng et al. [ICRA 2021] trained parameterized walking controllers on the Cassie bipedal robot that vary speed and height while tolerating large external perturbations, outperforming traditional hybrid-zero-dynamics controllers on resilience. The IJRR 2024 version extends to dynamic, versatile locomotion.

The common thread: none of this required new algorithms. All of it required careful environment engineering — better actuator models, domain randomization, curriculum, reward shaping. Algorithm choice (PPO vs SAC) is secondary to environment quality.

### Reach and touch tasks

The **MuJoCo Reacher** environment (two-joint arm, touch a moving or stationary target) is one of the simplest non-trivial continuous-control tasks, and it is *solved*. SAC converges on Reacher in under 100K steps in standard benchmarks; the task reward climbs reliably within the first few thousand episodes. The dm_control "manipulator" suite [Tassa et al.] adds contact-rich manipulation but remains in the same difficulty class for stationary targets.

**OpenAI Shadow Hand** [Andrychowicz et al. 2020, IJRR; arXiv:1808.00177] solved dexterous in-hand reorientation of a cube with a 24-DOF Shadow Dexterous Hand — arguably the hardest manipulation benchmark in the standard literature — using PPO with massive parallelism (6,000+ CPU cores), 100M+ steps, and automatic domain randomization. If a 24-DOF hand solving a Rubik's cube is solved, "small bilateral creature touches a stationary ball" is almost certainly in the solved regime.

**Bottom line for David's project:** The locomotion/reach task is not the research contribution — it is scaffolding. A small bilateral creature touching a ball should be achievable with SAC + dense reward + 500K–2M steps in standard MuJoCo. If it is not working, the issue is almost certainly environment engineering (reward signal, episode structure, target placement, action space) rather than algorithm. The videos of creatures jumping hurdles and walking like crabs that David has seen online are trained policies from exactly this pipeline.

### Faster platforms

The current generation of massively parallel simulators has made this dramatically cheaper:

- **Brax** [Freeman et al. 2021, Google] runs in JAX on GPU/TPU and scales linearly to 10,000+ parallel environments. Training speed: "locomotion in ten seconds or so" for simple tasks; a humanoid locomotion policy trained for 200M steps on RTX 4090 completes in ~56 minutes [arXiv:2407.05148].

- **MuJoCo Playground** [Google DeepMind, January 2025; arXiv:2502.08844] is an open-source GPU-accelerated suite built on MJX (JAX-based MuJoCo). Most state-based policies train in under 10 minutes on a single GPU. It supports quadrupeds (Go1, Spot, Barkour), humanoids (H1, G1), and manipulation tasks. This is where anyone starting a new project in early 2026 should look first.

- **IsaacGym / IsaacLab** [NVIDIA] trained a Humanoid to threshold reward in under 4 minutes on an A100 with 4,096 parallel agents. Note: IsaacLab (the newer framework) has been reported slower than IsaacGym in early benchmarks; the gap may close as it matures.

For David's task specifically: MuJoCo Playground or Brax would let him run the same experiment 100×–1000× faster than single-environment MuJoCo on Apple Silicon. The per-experiment cost in wall-clock time drops from hours to minutes, meaning he could run the ablations (vision-on vs. vision-off, FOV sweep) in an afternoon.

---

## 2. Neuroscience-Grounded Motor Learning

### Central Pattern Generators

The spinal cord contains neural circuits capable of generating rhythmic locomotor output without supraspinal input — Central Pattern Generators (CPGs). The foundational computational work is **Ijspeert et al. (2001/2007)** on salamander locomotion [Biological Cybernetics; Neural Networks 2008]. Ijspeert built a biologically plausible connectionist model of a CPG that produces both aquatic (traveling wave) and terrestrial (standing wave) gaits, with a two-dimensional biomechanical body simulation controlled by a leaky-integrator neural network. Parameters were found with a genetic algorithm. The core point: the spinal cord's rhythmic output is not computed top-down from a reward signal; it is an oscillatory attractor that higher levels modulate.

This is the ancestor of modern "hierarchical" locomotion controllers: a high-level policy (learned with RL) sends commands to a lower-level CPG or PD controller rather than directly setting joint torques. This hybrid architecture typically learns faster and transfers better to hardware because the lower level encodes hard physical constraints the upper level does not have to rediscover.

### MyoSuite and muscle-actuated control

**MyoSuite** [Caggiano, Wang, Durandau, Sartori, Kumar 2022; arXiv:2205.13600; PMLR 2022] is the closest thing to a physiologically grounded computational motor learning platform. It provides MuJoCo-based musculoskeletal models of the elbow, wrist, and hand, with hundreds of muscles and tendons, contact physics, and support for physiological alterations (tendon transfer surgery, muscle fatigue, sarcopenia). Tasks range from simple postural control to pen-twirling and key-turning. RL policies trained in MyoSuite learn to solve these tasks using muscle-activation commands rather than joint torques, producing motion that looks more like biological movement.

Importantly, MyoSuite does not reframe the learning process as conditioning — it still uses reward-maximization RL. What it changes is the action space and the body model. A 2024 Neuron paper by Caggiano and colleagues extended this to curriculum-based RL for acquiring musculoskeletal skills, finding that curriculum structure matters a great deal for convergence when the action space is 100+ muscle activations.

### Active inference / free energy principle

**Friston's Free Energy Principle** (FEP) and its behavioral implementation, active inference, offer a unified account of perception, action, and learning as approximate Bayesian inference [Friston 2010, Nature Reviews Neuroscience; Parr, Pezzulo, Friston 2022, MIT Press]. In the FEP framework, motor control is not reward maximization but the minimization of prediction error at proprioceptors — the motor system acts to make the world match its predictions. Action and perception are the same computation seen from two sides.

**Pezzulo** has extended active inference to hierarchical motor control and locomotion planning. A 2016 paper in Journal of the Royal Society Interface by Friston and colleagues demonstrated active inference robot control in simulation. The FEP account maps strikingly well onto classical reflex arcs: motor inference is realized at the spinal level, with higher levels setting priors (desired proprioceptive states) that reflexes execute by reducing prediction error.

For David's project: active inference is interesting because it explicitly frames motor learning as a process of building a generative model of sensorimotor contingencies — which is much closer to Taylor's conditioning-based account than reward maximization is. However, active inference has not (as of early 2026) produced locomotion results competitive with PPO or SAC on standard benchmarks. It is a theoretical framework, not yet a practical training algorithm for complex sim tasks.

### Internal models (Wolpert and Kawato)

**Wolpert, Miall, and Kawato (1998)** — *Internal Models in the Cerebellum* [Trends in Cognitive Sciences] — proposed that the cerebellum encodes paired forward (predictor) and inverse (controller) internal models of limb dynamics. The forward model predicts sensory consequences of motor commands; the inverse model computes the motor command needed to achieve a desired state. This framework accounts for skill acquisition, coordination, and the rapid, predictive nature of well-practiced movements.

**Wolpert and Kawato (1998)** — *Multiple Paired Forward and Inverse Models* [Neural Networks] — extended this to a modular architecture (MOSAIC) where multiple paired models compete, enabling context-appropriate motor behavior. This is neurobiologically grounded and computationally explicit, and it has influenced how some researchers structure hierarchical RL architectures.

For Taylor's project specifically: the internal model framework operationalizes something close to what Taylor means by conditioning — the body learns to predict sensory outcomes of its own actions, and that predictive model is the perception. But Wolpert/Kawato frame it in terms of cerebellar signal processing, not behavioral conditioning in the Skinnerian sense.

### Has anyone built a sim where motor learning is framed as classical/operant conditioning?

The short answer is: not exactly, but related things exist.

A 2009 undergraduate research paper at Rice (Thompson, Cox) built a computational model of operant and classical conditioning in a simple agent and showed that RL algorithms can replicate conditioning phenomena such as extinction and spontaneous recovery. The broader connection between RL and conditioning is well-established theoretically [Niv 2009, Cognitive, Affective, and Behavioral Neuroscience — "Reinforcement learning, conditioning, and the brain"]: temporal-difference RL maps naturally onto dopaminergic prediction error signaling, which is operant conditioning at the neural level.

**Spinal conditioning** is an active research area: protocols exist for operantly conditioning the H-reflex and spinal stretch reflex in humans and animals, producing plasticity in motoneurons and spinal interneurons [Wolpaw and colleagues; reviewed in multiple papers]. This is exactly the kind of sub-cortical, conditioning-based motor learning Taylor had in mind. But no one has built a full simulation that implements this spinal conditioning architecture and uses it to train a creature on a locomotion task. The gap between "conditioning as a neural mechanism" and "conditioning as a training algorithm in a simulator" has not been bridged computationally.

---

## 3. Has Taylor's Theory Been Tested Computationally?

### Taylor (1962) in context

James G. Taylor's *The Behavioral Basis of Perception* (Yale University Press, 1962; mathematical appendix by Seymour Papert) proposes that perception is not a passive decoding of sensory input but a learned behavioral disposition. The key concepts: (1) **equivalence classes** — stimuli that reliably produce the same response are collapsed into a single perceptual category through conditioning; (2) **Π / interpenetration** — the structured overlap or mutual dependency between senses, so that conditioning on one modality shapes the response to another. The theory is Skinnerian in foundation but rigorously formal for its era.

Critically, Taylor's theory was almost entirely ignored by mainstream cognitive science after the cognitive revolution of the 1960s, which moved away from behaviorism. There is no identified direct computational implementation of Taylor's Π framework in the RL or computational neuroscience literature. Searching specifically for computational tests of Taylor's interpenetration concept yields nothing. This appears to be genuinely novel territory.

### The closest living relatives

**Sensorimotor Contingency Theory (SMC)** [O'Regan and Noë 2001; Behavioral and Brain Sciences 24:5, 939–973] is the most direct contemporary parallel. O'Regan and Noë argue that seeing is not the activation of an internal representation but the mastery of sensorimotor contingencies — the lawful relationships between actions and their sensory consequences. "Seeing is a way of acting." This is structurally very similar to Taylor: perception is a learned behavioral relationship between organism and environment, not a stored representation.

SMC has been computationally implemented, at least partially. A 2011 paper at IEEE [cited as "A discrete computational model of sensorimotor contingencies for object perception and control of behavior"] built a formal framework inspired by SMC for robot control. A 2012 Springer volume included applications to terrain discrimination in quadruped robots using sensorimotor contingencies. The SMC literature has also intersected with predictive coding.

The key distinction from Taylor: SMC focuses on the action-perception loop within a single modality (mostly vision), while Taylor's Π is specifically about cross-modal conditioning — the way one sense shapes the perceptual categories of another. No computational model of SMC appears to address cross-modal interpenetration in Taylor's sense.

**Ecological perception (Gibson)** [*The Ecological Approach to Visual Perception*, 1979] shares Taylor's rejection of representationalism — Gibson argues that perception is direct pickup of affordances, not internal reconstruction. But Gibson explicitly resists computational description ("the systems do not compute, transform, or enrich information but 'resonate' to ecological information"), which makes it difficult to operationalize. Gibson and Taylor share a behavioral, organism-environment relational framing, but Taylor's conditioning mechanism gives his theory more computational traction.

**Predictive coding** [Rao and Ballard 1999; Nature Neuroscience 2:79–87] proposes that feedforward connections carry prediction errors while feedback connections carry top-down predictions. This is now the dominant framework for linking perception to neural architecture. It captures something of Taylor's idea that perception is a learned model, but predictive coding is largely indifferent to cross-modal conditioning and does not operationalize equivalence classes in Taylor's behavioral sense.

**Active inference / FEP** [Friston 2010; Parr, Pezzulo, Friston 2022] is the most mathematically developed account of perception and action as a unified process. It can, in principle, model cross-modal interaction as the integration of multiple likelihood mappings in a hierarchical generative model. But FEP does not specifically operationalize Taylor's conditioning mechanism, and the equivalence class formation through operant contingency is not part of the standard FEP vocabulary.

**Stimulus equivalence (Sidman)** [Sidman 1971, 1994; reviews in JEAB] — in behavior analysis, equivalence classes form when stimuli become interchangeable via reflexivity, symmetry, and transitivity after conditional discrimination training. Computational models of Sidman equivalence have been built [Tovar et al. 2023, JEAB arXiv:2507.00265; MDPI 2023] using various neural-network architectures. This is the closest behavioral-analytic tradition to Taylor's equivalence-class concept, and it has been computationalized. But Sidman's focus is on symbolic behavior (language, categorization), not multisensory motor learning, and the models are not embodied simulations.

**Bottom line on Taylor:** No computational test of Taylor's theory has been identified in the literature. The theory predates modern RL, was largely unassimilated into computational cognitive science, and the specific claim — that interpenetration of senses arises through conditioning such that one modality's perceptual categories become structurally dependent on another modality's conditioning history — has not been formalized or simulated. The SMC tradition (O'Regan and Noë) is the closest parallel in contemporary philosophy of perception; the stimulus equivalence tradition (Sidman) is closest in behavioral formalism; predictive coding and FEP are closest in computational neuroscience. None of them directly test Π.

---

## 4. Bottom-Line Implications for David's Project

### A note on what "solved" means here

This project has two intertwined goals that the wider literature does not separate:

1. **Plumbing goal:** a small bilateral creature touches a ball.
2. **Research goal:** vision becomes *load-bearing* on the creature's behavior — pixels are read and used in a way that is causally necessary to behavior, not redundant with proprio. The vision-ablation sensitivity metric in PROJECT_STATUS.md operationalizes this.

Goal (1) is solved many times over. Goal (2) has no published precedent. Critically: solving (1) the standard way does **not** automatically deliver (2). The standard recipe trains a policy that works; it does not test whether vision is causally necessary. Most of our current gap is in (1)'s plumbing — see `GAP_ANALYSIS.md` — and once that plumbing is right, the test for (2) becomes runnable. But the test itself, and the Taylor-Π framing it operationalizes, remains our contribution.

### Is the touch-the-ball task (goal 1) solved?

Yes, emphatically. A bilateral creature with a handful of joints touching a stationary ball is below the difficulty threshold of MuJoCo Reacher, which is one of the simplest tasks in the standard benchmark suite. The recipe that will work:

1. **SAC with a dense shaped reward** — distance-to-target negative reward + contact bonus. No exotic reward engineering needed.
2. **500K–2M environment steps** in a single MuJoCo environment on Apple Silicon. With 16 parallel environments (already the default in train_v8.py), you are effectively at 8–32M equivalent steps in wall-clock time.
3. **If it is not working**, the checklist is: (a) confirm reward is non-zero in first episodes; (b) confirm target is reachable from spawn positions; (c) confirm action space is not so large that random policy never accidentally contacts the ball; (d) check observation vector contains relative position to target.

The MirrorWrapper and FOV tuning efforts described in the project notes are environment engineering — which is exactly the right place to spend time. But the algorithm is not the bottleneck.

### Should you switch platforms?

For David's *current* research question — does vision become load-bearing? — the platform switch question is worth taking seriously. Here is the tradeoff:

- **Stay in MuJoCo on Apple Silicon:** familiar, debuggable, full control. Cost: training runs take hours.
- **Move to MuJoCo Playground (MJX + JAX):** same MuJoCo physics, GPU-accelerated, most tasks train in under 10 minutes. Cost: learning a new codebase, JAX ecosystem, some loss of fine-grained control over visualization.
- **Brax:** fastest of all, but the physics fidelity is lower (Brax uses its own physics solver, not MuJoCo's). For a research question about sensory interpenetration, MuJoCo physics fidelity matters.

The honest recommendation: the vision-ablation ablation experiment David wants to run — comparing behavior with and without pixel columns zeroed — requires many policy evaluations across many training checkpoints. That is exactly the experiment profile where a 100× speedup is most valuable. MuJoCo Playground is worth evaluating for stage 2 and beyond.

### Is the Taylor angle genuinely novel?

Yes, with caveats.

The SMC community (O'Regan, Noë, Buhrmann, Di Paolo) has made the same core philosophical claim — that perception is a learned sensorimotor relationship — and has built some computational implementations. But they have not asked Taylor's specific question about cross-modal conditioning, and they have not built an embodied agent that tests whether one modality's perceptual categories become structurally entangled with another's through operant history. That specific question, operationalized as "does zeroing pixel columns change a policy that was trained with vision + proprioception more than one trained without vision," is a clean empirical test of something close to Π, and there is no published version of this experiment.

The closest anybody comes is the sim-to-real literature's work on sensor fusion and ablation, but that work asks "does the policy degrade without sensor X?" as an engineering question, not "has sensor X become constitutive of the creature's behavioral categories?" in Taylor's sense.

The contribution David is building toward is genuinely novel at the level of research question, even if the tools (MuJoCo, SAC, vision-ablation sensitivity) are standard. The theoretical framing — Taylor's Π as a testable hypothesis about embodied sensory conditioning — is not represented in the computational literature.

---

## See also

- **`GAP_ANALYSIS.md`** — the inward-facing companion. Catalogs which standard practices from this survey we are not currently using, ranked by likely impact on the "vision never becomes load-bearing" symptom.
- **`PRIOR_ART_NUMENTA.md`** — focused deep-dive on Numenta / Hawkins / Thousand Brains / Monty. The closest major research program to this work; deep philosophical overlap (sensorimotor coupling as foundational, skepticism of reward-maximization) but methodological divergence (cortical-circuit theory vs. behavioral conditioning; one-shot Hebbian vs. SAC). Sharpest finding: Numenta's multi-column voting is *synchronic* coordination at inference time; Taylor's Π is *diachronic* structural conditioning over learning history. Not the same claim in different vocabularies.
- **`PROJECT_STATUS.md`** — the empirical ground truth for "what isn't working." The vision-ablation sensitivity numbers (frozen 2.29, unfrozen 0.85) and the "7/20 via forward-paddle alone" finding are the data the GAP_ANALYSIS recommendations are trying to explain.

---

## References

[Andrychowicz et al. 2020] Andrychowicz M, Baker B, Chociej M, et al. *Learning Dexterous In-Hand Manipulation*. IJRR 2020. arXiv:1808.00177.

[Caggiano et al. 2022] Caggiano V, Wang H, Durandau G, Sartori M, Kumar V. *MyoSuite: A Contact-rich Simulation Suite for Musculoskeletal Motor Control*. L4DC 2022; PMLR 168. arXiv:2205.13600.

[Freeman et al. 2021] Freeman CD, Frey E, Raichuk A, et al. *Brax — A Differentiable Physics Engine for Large Scale Rigid Body Simulation*. NeurIPS Datasets & Benchmarks 2021.

[Friston 2010] Friston KJ. *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience 11:127–138. DOI:10.1038/nrn2787.

[Fujimoto et al. 2018] Fujimoto S, van Hoof H, Meger D. *Addressing Function Approximation Error in Actor-Critic Methods (TD3)*. ICML 2018. arXiv:1802.09477.

[Gibson 1979] Gibson JJ. *The Ecological Approach to Visual Perception*. Houghton Mifflin.

[Google DeepMind 2025] *MuJoCo Playground*. Technical Report, January 2025. arXiv:2502.08844. https://playground.mujoco.org/

[Haarnoja et al. 2018] Haarnoja T, Zhou A, Abbeel P, Levine S. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning*. ICML 2018. arXiv:1801.01290.

[Hwangbo et al. 2019] Hwangbo J, Lee J, Dosovitskiy A, et al. *Learning agile and dynamic motor skills for legged robots*. Science Robotics 4(26). DOI:10.1126/scirobotics.aau5872.

[Ijspeert et al. 2001] Ijspeert AJ. *A connectionist central pattern generator for the aquatic and terrestrial gaits of a simulated salamander*. Biological Cybernetics 84. DOI:10.1007/s004220000211.

[Ijspeert 2008] Ijspeert AJ. *Central pattern generators for locomotion control in animals and robots: a review*. Neural Networks 21(4). DOI:10.1016/j.neunet.2008.03.014.

[Lee et al. 2020] Lee J, Hwangbo J, Wellhausen L, Koltun V, Hutter M. *Learning Quadrupedal Locomotion over Challenging Terrain*. Science Robotics 5(47). DOI:10.1126/scirobotics.abc5986.

[Li et al. 2021] Li Z, Cheng X, Peng XB, Abbeel P, Levine S, Berseth G, Sreenath K. *Reinforcement Learning for Robust Parameterized Locomotion Control of Bipedal Robots*. ICRA 2021. arXiv:2103.14295.

[Margolis et al. 2022] Margolis GB, Yang G, Paigwar K, Chen T, Agrawal P. *Rapid Locomotion via Reinforcement Learning*. RSS 2022; IJRR 2023. arXiv:2205.02824.

[Niv 2009] Niv Y. *Reinforcement learning, conditioning, and the brain: Successes and challenges*. Cognitive, Affective, and Behavioral Neuroscience 9(4):343–364. DOI:10.3758/CABN.9.4.343.

[O'Regan and Noë 2001] O'Regan JK, Noë A. *A sensorimotor account of vision and visual consciousness*. Behavioral and Brain Sciences 24(5):939–1031. DOI:10.1017/S0140525X01000115. (PMID:12239892)

[Parr, Pezzulo, Friston 2022] Parr T, Pezzulo G, Friston KJ. *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press. ISBN:9780262045353.

[Rao and Ballard 1999] Rao RPN, Ballard DH. *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects*. Nature Neuroscience 2:79–87. DOI:10.1038/4580.

[Schulman et al. 2017] Schulman J, Wolski F, Dhariwal P, Radford A, Klimov O. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.

[Sidman 1971] Sidman M. *Reading and auditory-visual equivalences*. Journal of Speech and Hearing Research 14:5–13.

[Tassa et al. 2018] Tassa Y, Tunyasuvunakool S, Muldal A, et al. *dm_control: Software and tasks for continuous control*. arXiv:2006.12983.

[Taylor 1962] Taylor JG. *The Behavioral Basis of Perception*. Yale University Press. (Mathematical appendix by Seymour Papert.)

[Tovar et al. 2023] Tovar AE et al. *Computational models of stimulus equivalence: An intersection for the study of symbolic behavior*. Journal of the Experimental Analysis of Behavior 119(1). DOI:10.1002/jeab.829.

[Wolpert, Miall, Kawato 1998] Wolpert DM, Miall RC, Kawato M. *Internal models in the cerebellum*. Trends in Cognitive Sciences 2(9):338–347. DOI:10.1016/S1364-6613(98)01221-2. (PMID:21227230)

[Wolpert and Kawato 1998] Wolpert DM, Kawato M. *Multiple paired forward and inverse models for motor control*. Neural Networks 11(7–8):1317–1329. DOI:10.1016/S0893-6080(98)00066-5.
