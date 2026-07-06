# Numenta / Thousand Brains Deep-Dive: Overlap and Divergence with the Taylor Project

*April 2026 — focused supplement to `PRIOR_ART.md`*

---

## 1. What Numenta Is and What It Has Built

Numenta is a research organization founded in 2005 by Jeff Hawkins and Donna Dubinsky. Its central claim, consistent across twenty years and two major theoretical iterations, is that the neocortex runs a single canonical algorithm, and that understanding that algorithm is the right path to general intelligence. The organization has never been in the business of building AI products in the commercial sense; it has been building a theory of cortical computation and prototyping it.

**Phase 1: HTM / NuPIC (2005–2018)**

HTM (Hierarchical Temporal Memory) is Numenta's first implementation of that theory. The formalism draws heavily on Vernon Mountcastle's proposal that the neocortex is a repeating-circuit device — each cortical column does the same thing, the "what" varies only by its inputs and outputs. HTM operationalizes this through two algorithms:

- **Spatial Pooler:** Maps arbitrary input into a *sparse distributed representation* (SDR) — a high-dimensional binary vector where roughly 2% of bits are active at any time. Sparsity is the key property: it enables graceful noise tolerance (a few flipped bits don't shift meaning) and massive overlap capacity (exponentially many distinct patterns storable in the same weights). The Spatial Pooler is an unsupervised Hebbian process, not gradient descent.

- **Temporal Memory (Sequence Memory):** Learns transitions *between* SDRs by growing new synapses onto dendritic segments. The mechanism is local: a neuron learns to predict its own activation by detecting patterns of co-active neurons on its basal dendrites. When those patterns occur again, the neuron enters a "predictive" (slightly depolarized) state before it would fire, effectively issuing a prediction. Surprise — the difference between predicted and actual activation — is what drives new learning. No reward signal. No backprop. No gradient.

NuPIC was the open-source Python/C++ implementation of HTM. It was archived by Numenta in 2018–2019 and is now in legacy-only mode. The community fork **htm.core** (C++17, Python bindings via pybind11) remains nominally active but appears to be a small-community maintenance project with no evidence of significant new algorithmic development as of early 2026.

HTM was applied primarily to **anomaly detection in time-series data** (streaming sensor data, financial streams, IT metrics). Numenta commercialized this briefly. The research contribution — SDR theory, temporal sequence memory at the neuron level — is real and peer-reviewed [Hawkins & Ahmad 2016, PMC4811948]. The biological claims (dendritic prediction, sparse activations) have partial empirical support in neuroscience and are not contested by most theorists. What HTM *did not* address: multi-modal integration, object recognition, space, reference frames, or movement.

**Phase 2: Thousand Brains Theory (2017–present)**

The grid-cell turn. In 2017–2019, Numenta absorbed work from neuroscience on entorhinal grid cells — neurons that encode position in physical space using a hexagonal periodic tiling — and asked: what if the neocortex runs *the same computation* but in object-centered reference frames rather than room-centered ones? This became the Thousand Brains Theory.

Core papers:
- [Hawkins & Ahmad 2017] "A Theory of How Columns in the Neocortex Enable Learning the Structure of the World," *Frontiers in Neural Circuits* [PMC5661005]
- [Hawkins et al. 2019] "A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex," *Frontiers in Neural Circuits*
- [Lewis, Purdy, Ahmad & Hawkins 2019] "Locations in the Neocortex: A Theory of Sensorimotor Object Recognition Using Cortical Grid Cells," *Frontiers in Neural Circuits* [PMC6491744]

The mechanism: each cortical column contains (a) a **location layer** using grid cell-like modules to represent "where on this object am I sensing right now?" and (b) a **sensory input layer** that encodes the feature detected *in the context of that location*. The two layers interact bidirectionally. Sensation updates the location estimate; motor commands (efference copy / path integration) propagate that estimate forward in time. After several sensorimotor steps, the column converges on a unique object-location assignment. Multiple columns, sensing different parts of the same object simultaneously (or the same part across time), vote via lateral connections — the "thousand brains" part — to reach consensus faster than any single column could.

The Thousand Brains book [Hawkins 2021] is an accessible popularization. It is worth reading for its argument structure but does not add technical content beyond the papers above.

**Phase 3: Monty (2024–present)**

In November 2024, Numenta released **Monty** (named for Vernon Mountcastle) as an open-source Python framework implementing Thousand Brains principles. The Thousand Brains Project simultaneously spun out as a 501(c)(3) nonprofit (January 2025), funded in part by a Gates Foundation grant ($2.69M over two years; the Gates Foundation is interested in applications to ultrasound interpretation). The project is led by Dr. Viviane Clay; Hawkins serves as Research Advisor.

**What Monty actually does:** A simulated "surface agent" — essentially a point sensor — moves along the surface of 3D objects in the **Habitat-Sim** simulator. At each step it records the local surface normal, curvature, and RGB value. A learning module (implementing one cortical column) maintains a probabilistic set of hypotheses over (object-identity, pose) pairs, updates them with each new observation, and issues goal states to the motor system. Motor policy is either input-driven (reactive, like a reflex) or hypothesis-driven (active inference: move the sensor to a location that maximally discriminates remaining candidates). Learning is associative/Hebbian — one-shot per object, stored in a graph structure mapping (location, feature) pairs.

The benchmark uses the **YCB object set** (77 real-world objects photogrammetrically scanned and placed in Habitat-Sim). A short benchmark uses a subset of 10 morphologically distinct objects. The system can recognize object ID and pose from a sequence of touch-like surface explorations. Key papers: [Clay, Leadholm et al. 2024, arXiv:2412.18354] and [Leadholm, Clay et al. 2025, arXiv:2507.04494].

**What Numenta has explicitly not done:**
- Locomotion. No physics-simulated creature learning to walk, run, or balance.
- RL with reward signals. The learning paradigm is associative/Hebbian; there is no policy gradient, no value function, no reward.
- Embodied agents in MuJoCo or any full-physics simulator. Habitat-Sim provides rigid-body collision but not physics-based creature locomotion.
- Cross-modal conditioning as a learning *process*. The voting mechanism aggregates modules that have learned different modalities (vision, touch), but the mechanism by which one sense structurally reorganizes another sense's categories is not addressed.
- Vision-ablation sensitivity as a probe. They do not ask "does the agent's policy change when vision is zeroed?"

---

## 2. Conceptual Overlap with the Taylor Project

The overlap is real and runs deeper than surface vocabulary similarity. Three specific alignments:

**A. Sensorimotor coupling as foundational, not add-on**

Both Taylor and Numenta reject the classical information-processing story in which perception is decoding — the environment presses a signal onto a passive receptor, the brain extracts meaning, and movement is an afterthought. Taylor's argument (1962): perceptual categories *are* behavioral response classes; you cannot define one without the other. Numenta's argument (2019): a cortical column that does not have access to motor efference copy cannot build stable object representations, because the same sensory input means completely different things depending on how the sensor got there. These are structurally the same claim. The sequence is: movement → changed sensation → learning update; without movement, the learning doesn't happen.

**B. Categories as relational structures in movement space**

Taylor's equivalence classes collapse stimuli that produce the same behavioral response. Numenta's reference frames assign stimuli to locations in object-centered coordinate space. These are not identical concepts, but they share a critical feature: neither treats a category as a feature-detector. Both treat it as a relation — Taylor, a relation to behavioral output; Numenta, a relation to a spatial coordinate. In both cases, what a stimulus *is* depends on what else is true (the behavioral context, or the sensorimotor trajectory that brought the sensor there). A purely passive system — one that receives a feature vector and outputs a class label — cannot represent this in principle.

**C. Skepticism of reward-maximization as the mechanism of learning**

Numenta's learning mechanism requires no reward signal. It is online, local, and associative. The system learns object structure because movement + sensation is informationally structured (an object has a consistent surface), not because it is incentivized to learn. Taylor's conditioning story is different in mechanism (operant/classical conditioning rather than Hebbian synapse growth), but the shared instinct is that the *environment's structure*, not an external reward gradient, does the work. The "pressure not bribery" principle David has articulated is a policy-design version of this same instinct.

---

## 3. Where They Diverge

**A. Level of analysis**

This is the deepest divergence. Numenta is committed to the cellular/columnar level: their theory makes specific claims about what happens in layer 4, layer 6, basal vs. apical dendrites, grid cell modules in entorhinal cortex. The reference frame mechanism is specifically about how efference copy reaches layer 6 via thalamus and propagates a location estimate. Taylor's theory is at the behavioral/conditioning level: it says nothing about how the brain implements equivalence classes, only that behavior is the criterion for collapsing stimuli into classes. One operates inside the neuron; the other operates at the organism-environment interface.

This difference matters practically. Numenta's implementation choices — sparse distributed representations, Hebbian synapse growth, columnar voting — are motivated by biological realism. David's implementation choices — SAC, dense reward, MuJoCo physics — are motivated by getting a creature to do something observable in reasonable time. These are different engineering regimes.

**B. What the "cross-modal" claim means**

Taylor's Π (interpenetration) is a claim about *conditioning history*: proprioception and vision become structurally entangled because the animal has repeatedly experienced them co-varying during behavior. The visual category for "that object" is partly constituted by the proprioceptive context in which visual encounters with it occurred. This is a diachronic claim — it is about what happens to categories *over time* as a result of sensorimotor experience.

Numenta's multi-modal voting is a synchronic claim — at inference time, learning modules that have learned objects in different modalities (one via touch, one via vision) vote on object identity using a shared message format (pose + features). The voting converges faster because more evidence is available. But neither module's internal representation has been restructured by the other's learning history. Touch's categories and vision's categories were learned independently; voting is coordination, not interpenetration. The distinction is: Taylor's Π changes what the category *is*; Numenta's voting changes how fast you identify which category *applies*.

**C. Learning paradigm and what can be proven**

Numenta's Hebbian/associative learning is one-shot per object (under favorable conditions). It does not require millions of training steps. But its task domain is narrow: static 3D objects in a simulated scanner. David's SAC training requires millions of steps but produces a policy that generalizes to novel physical conditions (new ball positions, perturbations) because it has explored a huge space of (state, action) pairs. These are different in what they buy: Numenta buys rapid acquisition of specific object models; SAC buys robust behavioral generalization. Neither is strictly superior — they are solving different problems.

**D. Vision specifically**

Monty uses vision (RGB + surface normals from a point sensor) and touch interchangeably as inputs to the same learning module architecture. But the question David is asking — *does vision become load-bearing on behavior, or is it decorative?* — requires a creature that can *solve the task without vision* and then being asked whether vision changes what it does. Monty's architecture does not run this probe. It is not designed to ask "what does the policy do when I zero out the pixels?" because there is no policy in the RL sense — there is a recognition/inference process, not a learned behavioral control law. The vision-ablation sensitivity measure David has specified has no analogue in Numenta's framework.

---

## 4. Practical: Is Any of It Usable for Us?

**Monty as a drop-in: No.**

Monty lives in Habitat-Sim and is architected around a point sensor exploring object surfaces. It does not have a natural interface to MuJoCo physics-simulated creatures. Integrating Monty as the "brain" of David's creature would require rewriting both the motor system (Monty's motor output is sensor displacement, not joint torques) and the sensory pipeline (MuJoCo observations are proprioceptive state vectors + pixel arrays, not curvature + surface normals from a surface-following agent). This is not a small integration project; it is a framework replacement.

**htm.core as a drop-in: Also no.**

NuPIC/htm.core was built for streaming temporal sequence prediction (anomaly detection), not for embodied sensorimotor learning with a continuous action space. The Spatial Pooler + Temporal Memory pipeline does not produce a policy; it produces sequence predictions. Adapting it to the creature task would require solving the credit assignment problem (how does a Hebbian learner decide which synaptic changes led to ball-touching?) — which is precisely what RL algorithms are designed to do.

**What IS extractable without committing to the full stack:**

1. **Sparse distributed representations as an observation encoding.** If David wanted to test whether the creature's internal representation has structure analogous to SDRs (distributed, sparse, overlap-based), he could add a bottleneck encoder that enforces sparsity on the observation before it feeds the policy network. This is not standard practice in SAC implementations but is technically straightforward. The Numenta papers on SDR properties [Ahmad & Hawkins 2016] make testable predictions about capacity, noise tolerance, and union representability that could be probed. Whether this would help vision become load-bearing is an open question, but it is a principled intervention.

2. **Reference frame framing as a diagnostic lens.** Even without using Numenta's code, their conceptual vocabulary is useful: ask whether the creature's policy has learned a reference frame (does it represent "ball relative to head"? "ball relative to arm"?) or whether it has learned a lookup table (memorized stimulus-response pairs). Ablation experiments — translate the ball, rotate the creature, perturb the starting state — reveal this. The Numenta papers on what reference frames buy [Lewis et al. 2019] provide a principled account of why reference-frame-structured representations generalize better to novel poses.

3. **Voting / ensemble as an analogy for the vision-ablation test.** The Thousand Brains "voting" intuition clarifies what David is really asking. If the creature has two "columns" — one that processes proprioception and one that processes vision — and they vote on the action, then vision is load-bearing if and only if zeroing the vision column's vote changes the consensus. This is exactly the vision-ablation sensitivity measure. Numenta's framework predicts that if vision and proprioception are seeing the *same* world structure from different angles, they will converge on the same answer independently; if they are not (e.g., the ball is visually ambiguous but proprioceptively unambiguous), vision adds nothing. The experimental prediction: vision becomes load-bearing when the ball is visually distinctive but proprioceptively ambiguous (moving targets, visual occlusion tests, large workspace). This is already implicit in the v9 moving-target plan.

4. **One-shot learning as an aspiration.** Monty learns object models in one or a few exploratory episodes. SAC requires millions of steps. If David's creature eventually needs to learn new objects rapidly (not just touch-the-ball, but "learn a new ball shape in 10 seconds"), Numenta's architecture has something to say about how that could work without retraining the entire policy. This is not an immediate concern but is relevant to longer-term research directions.

---

## 5. Bottom Line

Numenta and the Taylor project are traveling toward related territory from very different starting points and using very different tools.

The **genuine overlap** is philosophical and structural: both reject passive decoding, both treat sensorimotor coupling as constitutive of perception rather than additive to it, both are skeptical that reward maximization is the right account of how categories form. Reading Hawkins's 2019 paper carefully will not feel like reading a stranger's work; it will feel like a different research tradition converging on similar intuitions by a different path.

The **genuine divergence** is methodological and definitional:
- Numenta is a neural circuit theory implemented in a surface-scanning robot proxy. David is a behavioral theory tested in a physics-simulated embodied creature. Different levels.
- Numenta's multi-column voting is synchronic coordination at inference time. Taylor's Π is diachronic structural conditioning over learning history. These are not the same claim in different vocabularies.
- Numenta has never asked whether vision is *load-bearing* on behavior in the specific sense David means. They have not run the ablation. Their framework would predict a testable answer, but they have not done the experiment.

**Practical verdict:** Nothing from Numenta should be bolted onto the current stack. The integration cost is prohibitive and the immediate bottleneck (getting vision to be load-bearing at all) is not what Monty's architecture addresses. The value is conceptual: the reference-frame vocabulary is a useful diagnostic language for interpreting what the policy has learned, the voting analogy clarifies what "vision load-bearing" means at an architectural level, and the SDR literature provides one principled way to structure observations if the current encoding proves to be the limiting factor.

Numenta is the closest major research program to what David is doing. That is genuinely notable — most of the sensorimotor learning literature is either pure neuroscience or pure robotics, and Numenta is explicitly trying to bridge them. But "closest" still means distant enough that David is not duplicating their work, and not close enough that their code is a shortcut.

---

## References

- Hawkins, J. & Ahmad, S. (2016). Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex. *Frontiers in Neural Circuits*. PMC4811948. arXiv:1511.00083
- Hawkins, J., Ahmad, S., & Cui, Y. (2017). A Theory of How Columns in the Neocortex Enable Learning the Structure of the World. *Frontiers in Neural Circuits*. PMC5661005
- Hawkins, J., Lewis, M., Klukas, M., Purdy, S., & Ahmad, S. (2019). A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex. *Frontiers in Neural Circuits*. https://www.numenta.com/resources/research-publications/papers/a-framework-for-intelligence-and-cortical-function-based-on-grid-cells-in-the-neocortex/
- Lewis, M., Purdy, S., Ahmad, S., & Hawkins, J. (2019). Locations in the Neocortex: A Theory of Sensorimotor Object Recognition Using Cortical Grid Cells. *Frontiers in Neural Circuits*. PMC6491744
- Hawkins, J. (2021). *A Thousand Brains: A New Theory of Intelligence.* Basic Books.
- Clay, V., Leadholm, N., et al. (2024). The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence. arXiv:2412.18354
- Leadholm, N., Clay, V., Knudstrup, S., Lee, H., & Hawkins, J. (2025). Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference. arXiv:2507.04494
- thousandbrainsproject/tbp.monty. GitHub. https://github.com/thousandbrainsproject/tbp.monty (MIT License)
- Thousand Brains Project nonprofit: https://thousandbrains.org/ (spun out January 2025, Gates Foundation funded)
