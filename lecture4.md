# Lecture 4: Multi‑Qubit Systems, Mixed States, and Entanglement  
*From Pure States to Density Matrices, Partial Trace, and Quantifying Entanglement*  
*notes prepared with the help of AI*

---

## Outline

- Why we need a more general description than state vectors  
- **Density matrices**: definition, properties, examples  
- **Partial trace** – how to “ignore” part of a system  
- **Reduced density matrices** – what a subsystem looks like alone  
- **Purity** – measuring how mixed a state is  
- **Schmidt decomposition** – the mathematical structure of bipartite states  
  - **Quantifying entanglement** – entropy of entanglement  
- **No‑cloning theorem** – a simple proof  
- **Applications and PennyLane demos**: Teleportation & Superdense coding (recap with density matrices), computing reduced states and entanglement  

---

## Motivation – Pure vs. Mixed States

So far we have described quantum systems by **state vectors** $|\psi\rangle$ (pure states).  

---

But what if:

- We only have **incomplete knowledge**?  

**Example:** “The qubit is in $|0\rangle$ with probability 1/2, and in $|1\rangle$ with probability 1/2” 
  - a **statistical mixture**.

---

And what if
- The system is part of a larger entangled system, and we only have access to one part?  

  **Example:** A Bell state $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)$ – if we look only at the first qubit, what do we see?

---

In both cases, a pure state description is insufficient. That is why we need the **density matrix** (or **density operator**).

---

## Classical uncertainty vs. quantum superposition

**Pure superposition:**  
- Qubit in state $\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$  
- State vector $|\psi\rangle$ 

For the superposition $|+\rangle$, measuring in $X$ basis gives **always** $+1$ (deterministic).  
 
---

**Statistical mixture:** 
- Qubit is either $|0\rangle$ or $|1\rangle$ with 50% probability 
- Density matrix $\rho$ 

For the mixture, 
  - measuring in $X$ basis gives $+1$ and $-1$ each with 50% probability.

Density matrices capture this difference.

---

## Density Matrix – Definition

For a **pure state** $|\psi\rangle$, the density matrix is the **projector** onto that state:

$$ \rho = |\psi\rangle\langle\psi|. $$

---

For a **statistical mixture** of pure states $\{|\psi_i\rangle\}$ with probabilities $p_i$ ($\sum p_i = 1$, $p_i \ge 0$), the density matrix is

$$ \rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|. $$

---

- **Hermitian:** $\rho^\dagger = \rho$  

---

- **Trace 1:** $\text{Tr}(\rho) = 1$ (total probability = 1)  

---
- **Positive semi‑definite:** $\langle\phi|\rho|\phi\rangle \ge 0$ for any $|\phi\rangle$

---

A **pure state** satisfies $\rho^2 = \rho$ (a projector).  
A **mixed state** has $\text{Tr}(\rho^2) < 1$.

---

## Density Matrix – Examples (1)

**Pure $|0\rangle$:**

$$ \rho = |0\rangle\langle0| = \begin{bmatrix}1 & 0 \\
 0 & 0\end{bmatrix}. $$

**Pure $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$:**

$$ \rho = |+\rangle\langle+| = \frac12\begin{bmatrix}1 & 1 \\
1 & 1\end{bmatrix}. $$

---

## Density Matrix – Examples (2)

**50‑50 mixture of $|0\rangle$ and $|1\rangle$** (classical coin flip):

---

$$ \rho = \frac12 |0\rangle\langle0| + \frac12 |1\rangle\langle1| = \frac12 \begin{bmatrix}1&0\\
0&1\end{bmatrix} = \frac{I}{2}. $$

This is the **maximally mixed state**.  
Check purity: $\rho^2 = \frac{I}{4}$, so $\text{Tr}(\rho^2) = \frac12 < 1$ (mixed).

---

**50‑50 mixture of $|+\rangle$ and $|-\rangle$**:

---

$$ \rho = \frac12 |+\rangle\langle+| + \frac12 |-\rangle\langle-| = \frac{I}{2}. $$

Same result! Different mixtures can give the same density matrix.  
The density matrix captures **all observable statistics**.

---

## Expectation values and measurement

The probability of measuring outcome $m$ (with projector $P_m$) is

$$ p(m) = \text{Tr}(P_m \rho). $$

---

The expectation value of an observable $M$ is

$$ \langle M \rangle = \text{Tr}(M \rho). $$

**Why trace?** The trace automatically averages over all possibilities in the mixture.

---

For a pure state $\rho = |\psi\rangle\langle\psi|$, 

$$ \text{Tr}(M\rho) = \langle\psi|M|\psi\rangle $$ 
- consistent with Born rule.


---

## Visualizing single‑qubit density matrices – Bloch ball

A pure state lies on the **surface** of the Bloch sphere.  
A mixed state lies **inside** the sphere.

---

- **Pure states** → surface (radius = 1)  
- **Maximally mixed state** $I/2$ → center (radius = 0)  
- Partially mixed states → somewhere inside

---

The Bloch vector $\vec{r} = (\langle X\rangle, \langle Y\rangle, \langle Z\rangle)$ for a general density matrix satisfies $|\vec{r}| \le 1$.

---

**Bubble analogy (mixed states):**  
- Pure state = one coherent bubble  
- Mixed state = fog (many tiny droplets, no coherence)

---

## Partial Trace – “Forgetting” a Subsystem

When we have a composite system (say two qubits $A$ and $B$) but only care about subsystem $A$, we need to **trace out** $B$. This operation is called the **partial trace**.

---

**Intuition:**  
- You have a joint state of two particles.  
- You only look at particle $A$ and ignore particle $B$ completely.  
- What is the state of $A$ alone? It’s given by the **reduced density matrix** $\rho_A = \text{Tr}_B(\rho_{AB})$.

---

## Partial Trace – Definition

Given a bipartite density matrix 
- $\rho_{AB}$ on $\mathcal{H}_A \otimes \mathcal{H}_B$, 

and let 
- $\{|j\rangle_B\}$ be any orthonormal basis for $\mathcal{H}_B$

the reduced density matrix for subsystem $A$ is

$$ \rho_A = \text{Tr}_B(\rho_{AB}) = \sum_j (I_A \otimes \langle j|_B) \; \rho_{AB} \; (I_A \otimes |j\rangle_B), $$

---

**In matrix terms**: if $\rho_{AB}$ is written with indices $i,i'$ for $A$ and $j,j'$ for $B$, then

$$ (\rho_A)_{i,i'} = \sum_j (\rho_{AB})_{ij,i'j}. $$

- We sum over the index of the traced-out system.

---

## Example – Product state

Let $\rho_{AB} = |0\rangle\langle0|_A \otimes |1\rangle\langle1|_B$.

$\rho_A$?

---

$$ 
\rho_A = \text{Tr}_B\big(|0\rangle\langle0|_A \otimes |1\rangle\langle1|_B\big) = |0\rangle\langle0|_A \cdot \text{Tr}(|1\rangle\langle1|_B). 
$$

Since $\text{Tr}(|1\rangle\langle1|) = 1$, we get

$$ \rho_A = |0\rangle\langle0|_A. $$

So $\rho_A$ is pure $|0\rangle$ – the subsystem is in a definite state, as expected for a product state.

---

## Example – Bell state

Take $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)$. Its density matrix is

$$ 
\rho_{AB} = \frac12\big(|00\rangle\langle00| + |00\rangle\langle11| + |11\rangle\langle00| + |11\rangle\langle11|\big). 
$$

---

Tracing out $B$ (sum over $j=0,1$ for qubit B), we get

$$ 
\rho_A = \frac12\big( |0\rangle\langle0| + |1\rangle\langle1| \big) = \frac{I}{2}. 
$$

The same holds for $\rho_B$.

---

**Key insight:** Even though the joint state $|\Phi^+\rangle$ is **pure**, each qubit alone is in the **maximally mixed state**!  
This is a signature of entanglement: the parts are completely random, but the whole is perfectly correlated.

---

## Purity – Measuring Mixedness

The **purity** is defined as

$$ 
P = \text{Tr}(\rho^2). 
$$

- For a pure state: $P = 1$ (since $\rho^2 = \rho$ and $\text{Tr}(\rho)=1$).  
- For a maximally mixed state of dimension $d$: $P = 1/d$ (for a qubit, $d=2$, so $P = 1/2$).  
- For the Bell state reduced density matrix $\rho_A = I/2$: $P = 1/2$.

---

Purity tells us how “mixed” a state is. 
It’s a quick check for entanglement: if the reduced density matrix of a subsystem has purity < 1, the overall state may be entangled.  

---

- **Caution:** Purity alone doesn't distinguish entanglement from classical mixing

---

**Caution:** A statistical mixture of product states e.g., 
-  50% $|00\rangle$, 50% $|11\rangle$
- This gives a mixed reduced density matrix, but **the overall state is not entangled** (it's a classical mixture). 

Purity alone doesn't distinguish entanglement from classical mixing – we need the Schmidt decomposition for that.

---

## Schmidt Decomposition – The Structure of Bipartite Pure States

Given any pure state $|\psi\rangle_{AB}$ on a bipartite system dimensions $d_A$, $d_B$, 
- there exist orthonormal sets $\{|u_i\rangle_A\}$ in $\mathcal{H}_A$ and $\{|v_i\rangle_B\}$ in $\mathcal{H}_B$, 
- and non‑negative real numbers $\lambda_i$ (**Schmidt coefficients**) such that

$$ |\psi\rangle_{AB} = \sum_{i=1}^{r} \lambda_i \, |u_i\rangle_A \otimes |v_i\rangle_B, $$

- $r \le \min(d_A, d_B)$ is the **Schmidt rank**.  
- The $\lambda_i$ satisfy $\sum_i \lambda_i^2 = 1$.

---

**Key:** The Schmidt decomposition reveals the entanglement structure.

---

## How to find the Schmidt decomposition

1. Compute the reduced density matrix $\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|)$.  
2. Diagonalize $\rho_A$: its eigenvalues are $\lambda_i^2$, and the eigenvectors are $|u_i\rangle_A$.  
3. The $|v_i\rangle_B$ are then determined (they are eigenvectors of $\rho_B$ with the same eigenvalues, up to a phase).

---

**Example – Bell state:**  
$\rho_A = I/2$ → eigenvalues $1/2, 1/2$ → Schmidt rank = 2, $\lambda_1 = \lambda_2 = 1/\sqrt{2}$, $|u_1\rangle=|0\rangle$, $|u_2\rangle=|1\rangle$, $|v_1\rangle=|0\rangle$, $|v_2\rangle=|1\rangle$.

---

**Bubble analogy (entanglement):**  
- Product state = two separate bubbles (each independent)  
- Entangled state = one bubble with two connected lobes (cannot describe one lobe without the other)

---
## Schmidt Rank and Entanglement

- If **Schmidt rank = 1**, the state is a **product state** (not entangled).  
- If **Schmidt rank > 1**, the state is **entangled**.  

---

- The larger the Schmidt rank, the more “degrees of entanglement” (but for two qubits, max rank is 2).

For a Bell state, rank = 2 → maximally entangled.  
For a product state, rank = 1 → no entanglement.

---

## Quantifying Entanglement – Entropy of Entanglement

For a bipartite **pure** state, a natural measure of entanglement is the **von Neumann entropy** of the reduced density matrix:

$$ S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A) = -\sum_i \lambda_i^2 \log_2 \lambda_i^2, $$

where $\lambda_i$ are the Schmidt coefficients.

---

- Product state ($\rho_A$ pure) → $S = 0$.  
- Maximally entangled state (Bell state) → $\rho_A = I/2$ → $S = \log_2 2 = 1$ (one **ebit** of entanglement).  
- Partially entangled state → $0 < S < 1$.

---

## No‑Cloning Theorem (Revisited)

**No‑cloning theorem:** It is impossible to make a perfect copy of an unknown quantum state.

---

**Proof sketch (using linearity):**  
Suppose there exists a unitary $U$ that clones: $U(|\psi\rangle\otimes|0\rangle) = |\psi\rangle\otimes|\psi\rangle$ for all $|\psi\rangle$.  
Consider two orthogonal states $|\psi\rangle$ and $|\phi\rangle$, and their superposition $|\omega\rangle = \frac{1}{\sqrt{2}}(|\psi\rangle+|\phi\rangle)$.  
By linearity, $U(|\omega\rangle|0\rangle) = \frac{1}{\sqrt{2}}(|\psi\rangle|\psi\rangle + |\phi\rangle|\phi\rangle)$.  
But if $U$ cloned, we should get $|\omega\rangle|\omega\rangle = \frac12(|\psi\rangle|\psi\rangle + |\psi\rangle|\phi\rangle + |\phi\rangle|\psi\rangle + |\phi\rangle|\phi\rangle)$, which is different. Contradiction.

---

**Implication:** Quantum information cannot be copied, which is why eavesdropping in BB84 is detectable.

---

## Application – Quantum Teleportation (Recap)

**Goal:** Send an unknown qubit state $|\psi\rangle$ from Alice to Bob using only classical communication and a shared entangled pair.

---

**Protocol:**  
1. Create Bell pair $|\Phi^+\rangle$ between Alice and Bob.  
2. Alice performs a Bell measurement on her two qubits (the unknown $|\psi\rangle$ and her half of the Bell pair).  
3. Alice sends the 2‑bit measurement result to Bob.  
4. Bob applies a Pauli correction ($X$, $Z$, or both) to his qubit, recovering $|\psi\rangle$.

---

**Key point:** The teleported state appears on Bob’s side **without** any qubit traveling from Alice to Bob – only classical bits.  
The shared entanglement is the “quantum resource”.

---

## Why Teleportation Works – Density Matrix View

- After Alice’s Bell measurement, Bob’s qubit is in one of four possible states, each related to $|\psi\rangle$ by a Pauli operator.  

---

- Because the Bell measurement results are random, Bob’s qubit before correction is a **mixture** of these possibilities.  

---

- But once Alice sends the result, Bob knows which correction to apply, and the final state becomes **pure** $|\psi\rangle$.

---

**Reduced density matrix** view: Tracing out Alice’s qubits leaves Bob’s qubit in a maximally mixed state **until** Alice’s measurement outcome is known 
- demonstrates that entanglement alone does not transmit information 
- classical communication is essential.

---

## Application – Superdense Coding (Recap)

**Goal:** Send 2 classical bits using only 1 qubit, with the help of shared entanglement.

---

**Protocol:**  
1. Share a Bell pair $|\Phi^+\rangle$.  
2. Alice encodes her 2 bits by applying $I$, $X$, $Z$, or $iY$ to her qubit.  
3. Alice sends her qubit to Bob.  
4. Bob performs a Bell measurement on both qubits, recovering the 2 bits.

---

**Why it works:** The four Bell states are orthogonal; 
- by applying Pauli operations, 
  - Alice can map the shared Bell state to any of the four, 
  - and Bob can distinguish them.

---

## PennyLane Demo – Creating Bell state and computing reduced density matrix

```python
import pennylane as qml
import numpy as np

dev = qml.device('default.qubit', wires=2)

@qml.qnode(dev)
def bell_state():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0,1])
    return qml.state()

state = bell_state()
print("Bell state vector:\n", state)

# Full density matrix
rho_full = np.outer(state, np.conj(state))
print("\nFull density matrix:\n", rho_full.round(3))

# Reduced density matrix for qubit 0 using partial trace
# Reshape to (2,2,2,2) and trace over qubit 1
rho_full_reshaped = rho_full.reshape(2,2,2,2)
rho_A = np.trace(rho_full_reshaped, axis1=2, axis2=3)
print("\nReduced density matrix for qubit 0:\n", rho_A.round(3))
print("Purity of rho_A:", np.trace(rho_A @ rho_A).real)
```

---

## PennyLane Demo – Entropy of entanglement

```python
from scipy.linalg import eigvalsh

evals = eigvalsh(rho_A)
evals = np.clip(evals, 0, 1)  # remove numerical negatives
entropy = -np.sum(evals * np.log2(evals + 1e-12))
print("Entropy of entanglement:", entropy)
```

For the Bell state, this should be $1$ (up to numerical precision).

---

## PennyLane Demo – Non‑maximally entangled state

```python
@qml.qnode(dev)
def non_max_state():
    qml.RY(1.2, wires=0)          # create non-uniform amplitudes
    qml.CNOT(wires=[0,1])
    return qml.state()

state2 = non_max_state()
rho_full2 = np.outer(state2, np.conj(state2))
rho_A2 = np.trace(rho_full2.reshape(2,2,2,2), axis1=2, axis2=3)
evals2 = eigvalsh(rho_A2)
evals2 = np.clip(evals2, 0, 1)
entropy2 = -np.sum(evals2 * np.log2(evals2 + 1e-12))
print("Entropy of entanglement for non-max state:", entropy2)
```

You’ll get a value between 0 and 1.

---

## Summary

**Density matrix** $\rho$ General description of quantum states (pure or mixed) 
**Partial trace** $\text{Tr}_B$ Operation to ignore a subsystem 
**Reduced density matrix** $\rho_A$  State of subsystem A alone 
**Purity** $\text{Tr}(\rho^2)$  Measures how mixed a state is 
**Schmidt decomposition**  Write bipartite pure state as $ \sum \lambda_i \mid u_i\rangle\mid v_i\rangle $ 
 - **Schmidt rank**  Number of terms; >1 means entangled   

**Entropy of entanglement** $S(\rho_A)$  Measure of entanglement for pure states 
**No‑cloning theorem**  Impossible to clone an unknown quantum state 
**Teleportation / Superdense coding**  Applications of entanglement 

---

## Exercises

1. **Partial trace practice** – For $|\psi\rangle = \frac{1}{\sqrt{3}}(|00\rangle + |01\rangle + |10\rangle)$, compute $\rho_A$ by hand or with NumPy. Is it entangled? Schmidt rank?

2. **Mixture vs. superposition** – Consider $\rho = \frac12 |0\rangle\langle0| + \frac12 |+\rangle\langle+|$. Write $\rho$ as a matrix, compute its purity. Can it be written as $|\phi\rangle\langle\phi|$?

3. **Schmidt decomposition** – Find the Schmidt decomposition of $|\psi\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |01\rangle)$. Is it a product state?

4. **Entropy of entanglement** – For $|\psi\rangle = \sqrt{0.8}|00\rangle + \sqrt{0.2}|11\rangle$, compute $\rho_A$, eigenvalues, entropy. Compare to Bell state.

5. **PennyLane** – Generate random two‑qubit states, compute entropy histogram.

---

## Next Lecture Preview

**Week 5: Introduction to Quantum Algorithms**  
- Deutsch-Jozsa algorithm  
- Bernstein-Vazirani algorithm  
- Simon's algorithm  

**Reading:** Nielsen & Chuang Chapter 3 (intro to algorithms)

*These slides were prepared with the help of AI (DeepSeek). Remember to verify results and understand the mathematics behind the code.*