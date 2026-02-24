# Homework 1: From Bits to Qubits – Math, Numpy, and a First Variational Circuit
*prepared with the help of DeepSeek AI*

**Due date:** given in classroom
**Submission:** Upload a single Jupyter notebook (`.ipynb`) or Python script with your code, plots, and written answers.  
**Group work:** You may work in groups of 1–3. Include all group members' names at the top.

---

## Overview

This assignment has two parts:

- **Part A** focuses on the basic mathematics of qubits using **NumPy**. You will represent quantum states, compute inner products, work with Pauli matrices, and explore measurement probabilities.
- **Part B** asks you to modify the PennyLane single‑qubit predictor from Lecture 2 to fit a different function, and to analyze what functions such a circuit can represent.

You will need Python with `numpy`, `matplotlib`, `pennylane`, `torch`, and `scikit-learn` installed. The lecture notebooks provide all the necessary building blocks.

---

## Part A: Qubit Math with NumPy (70 points)

In this part you will use NumPy to perform basic linear algebra operations that underlie quantum computing. All questions should be answered with code and short explanations.

### A.1 Representing kets and bras

A qubit state is a vector in ℂ². Represent the following states as NumPy arrays (use `np.array` with `dtype=complex`):

- $|0\rangle$
- $|1\rangle$
- $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$
- $|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$
- $|+i\rangle = \frac{1}{\sqrt{2}}(|0\rangle + i|1\rangle)$
- $|-i\rangle = \frac{1}{\sqrt{2}}(|0\rangle - i|1\rangle)$

Print each array.

### A.2 Bras and inner products

For a given ket $|\psi\rangle$, the corresponding bra $\langle\psi|$ is the conjugate transpose (use `np.conj` and `.T`).  
Write a function `inner(psi, phi)` that returns $\langle\psi|\phi\rangle$ for two kets `psi` and `phi`.

Test it by computing:

- $\langle 0 | 1 \rangle$
- $\langle + | + \rangle$
- $\langle + | - \rangle$
- $\langle 0 | + \rangle$

Verify that $\langle + | + \rangle = 1$ and $\langle + | - \rangle = 0$. Explain why these results make sense.

### A.3 Normalization

Check that each of the states in A.1 is normalized (norm = 1). Write a function `norm(psi)` that returns $\sqrt{\langle\psi|\psi\rangle}$ and apply it to each state.

### A.4 Measurement probabilities

Given a state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$, the probability of measuring $|0\rangle$ is $|\alpha|^2$ and of measuring $|1\rangle$ is $|\beta|^2$.

Write a function `probs(psi)` that returns a tuple `(p0, p1)`. Test it on the six states above. For each state, print the probabilities and verify they sum to 1.

### A.5 Pauli matrices

Define the Pauli matrices as NumPy arrays:

```python
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)
```

Print each matrix.

### A.6 Eigenvalues and eigenvectors

For each Pauli matrix ($X$, $Y$, $Z$), compute its eigenvalues and eigenvectors using `np.linalg.eig`.  
For each matrix, print:

- The eigenvalues.
- The corresponding eigenvectors (as kets).
- Verify that the eigenvectors are orthonormal (inner product between different eigenvectors is 0, norm 1).

**Question:** What physical measurement does each Pauli matrix correspond to? (Refer to Lecture 2.)

### A.7 Applying gates

A quantum gate is a unitary matrix. Apply each Pauli gate to the state $|0\rangle$ (i.e., compute $X|0\rangle$, $Y|0\rangle$, $Z|0\rangle$) and interpret the resulting state. Does it match the known actions (e.g., $X$ flips the bit)? Do the same for $|1\rangle$ and $|+\rangle$.

### A.8 Measurement in different bases

Measuring in the $Z$ basis means projecting onto $|0\rangle$ and $|1\rangle$. To measure in the $X$ basis, we project onto the eigenstates of $X$, i.e., $|+\rangle$ and $|-\rangle$. Similarly for $Y$.

For the state $|\psi\rangle = \frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$, compute:

- Probabilities of measuring $+1$ and $-1$ in the $Z$ basis (this is the same as measuring $|0\rangle$ / $|1\rangle$).
- Probabilities of measuring $+1$ and $-1$ in the $X$ basis.
- Probabilities of measuring $+1$ and $-1$ in the $Y$ basis.

*Hint:* For $X$ basis, the probability of $+1$ is $|\langle + | \psi \rangle|^2$, etc.

### A.9 Hermitian and unitary checks

- Show that each Pauli matrix is Hermitian: $M^\dagger = M$. (Use `np.allclose` after computing conjugate transpose.)
- Show that each Pauli matrix is unitary: $M^\dagger M = I$. (Again use `np.allclose`.)
- Explain why quantum gates must be unitary (refer to Lecture 2).

### A.10 Plot measurement probabilities vs. rotation

In Lecture 2 we saw that applying $R_y(\theta)$ to $|0\rangle$ gives $\langle Z \rangle = \cos\theta$. More generally, the probabilities of measuring $|0\rangle$ and $|1\rangle$ are $\cos^2(\theta/2)$ and $\sin^2(\theta/2)$.

Write code to plot these two probabilities as functions of $\theta$ from $0$ to $2\pi$. Label axes and add a legend. (You can use `np.linspace` and `matplotlib`.)

---
### A.11 Random rotations: measurement probabilities and the Bloch sphere

In this exercise you will explore how a random sequence of rotations affects the measurement probabilities in the $X$, $Y$, and $Z$ bases, and connect these probabilities to the expectation values $\langle X \rangle$, $\langle Y \rangle$, $\langle Z \rangle$. You will also visualize the state as a point in the “observable space” and see the Bloch sphere constraint.

#### 1. Recall eigenvalues
From A.6 you already know that:
- $X$, $Y$, $Z$ each have eigenvalues $+1$ and $-1$.
- The eigenstates are respectively:  
  $X$: $|+\rangle$, $|-\rangle$  
  $Y$: $|+i\rangle$, $|-i\rangle$  
  $Z$: $|0\rangle$, $|1\rangle$

#### 2. Rotation gates (explicit matrices)
Using the formulas from the lecture notes, define functions that return the $2\times2$ unitary matrices:

```math
R_x(\theta) = \begin{bmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}
```

```math
R_y(\theta) = \begin{bmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}
```

```math
R_z(\theta) = \begin{bmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{bmatrix}
```

Implement these in NumPy. Verify that we can also write:
```math
R_x(\theta) = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} X
```
```math
R_y(\theta) = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} Y
```
```math
R_z(\theta) = \cos\frac{\theta}{2} I - i\sin\frac{\theta}{2} Z
```
   (You can also use `scipy.linalg.expm` on X, Y, Z matrices with $i\theta$ angle. They give the same results)

#### 3. Random sequence
Pick three random angles $\theta_x, \theta_y, \theta_z$ uniformly from $[0, 2\pi)$.  
Construct the overall unitary $U = R_z(\theta_z)\, R_y(\theta_y)\, R_x(\theta_x)$ (apply $R_x$ first, then $R_y$, then $R_z$).  
Starting from the initial state $|0\rangle$, compute the final state $|\psi\rangle = U|0\rangle$.

#### 4. Expectation values
For each Pauli operator $M \in \{X, Y, Z\}$:
- Compute $\langle M \rangle = \langle \psi | M | \psi \rangle$ directly from the state vector.
- Compute the same quantity as $\langle 0 | U^\dagger M U | 0 \rangle$ and verify they match (use `np.allclose`).
- Print $\langle X \rangle$, $\langle Y \rangle$, $\langle Z \rangle$.

#### 5. Probabilities from expectation values
Because the eigenvalues are $\pm 1$, the probabilities of measuring $+1$ and $-1$ for an observable $M$ are given by:
```math
p_{+1} = \frac{1 + \langle M \rangle}{2}, \qquad p_{-1} = \frac{1 - \langle M \rangle}{2}.
```

For each basis, compute these probabilities and verify that they match the direct inner‑product calculations (e.g., for $Z$: $p_0 = |\langle 0|\psi\rangle|^2$, $p_1 = |\langle 1|\psi\rangle|^2$; for $X$: $p_+ = |\langle +|\psi\rangle|^2$, etc.).

#### 6. Bar chart of probabilities
Create a bar chart with three groups ($Z$ basis, $X$ basis, $Y$ basis). For each group, show two bars: the probability of the $+1$ outcome and the probability of the $-1$ outcome. Label the axes and include a legend. Annotate the bars with the numerical values.

#### 7. Visualizing the state in observable space (Bloch sphere projection)
The three expectation values $(\langle X \rangle, \langle Y \rangle, \langle Z \rangle)$ define a point in $\mathbb{R}^3$. For any pure state, this point lies on the surface of the **Bloch sphere**, meaning $\langle X \rangle^2 + \langle Y \rangle^2 + \langle Z \rangle^2 = 1$ (up to numerical precision).

- Create a 2D scatter plot showing the point $(\langle X \rangle, \langle Z \rangle)$. (You can also plot $(\langle X \rangle, \langle Y \rangle)$ or any pair, but $X$ and $Z$ are often the most intuitive.)
- On the same plot, draw a unit circle (radius 1) centered at $(0,0)$ to represent the Bloch sphere’s equator.
- Add a point for your random state. Label the point with the values of $\langle X \rangle$ and $\langle Z \rangle$.

#### 8. Repeat for multiple random trials
Run the experiment for at least **five different random seeds** (or simply loop over five random angle sets). For each trial, store the expectation values.

- Create a second plot showing all the $(\langle X \rangle, \langle Z \rangle)$ points from these trials, again with the unit circle.
- Optionally, also plot $(\langle X \rangle, \langle Y \rangle)$ or $(\langle Y \rangle, \langle Z \rangle)$ in separate subplots.

Observe that all points lie exactly on the unit circle (within floating‑point error). Why is that? What would happen if the state were mixed? (We haven't studied mixed states (the dots inside the unit circle) yet, but you can reason about it.)

#### 9. Interpretation
Answer the following questions in a markdown cell:
- For one of your random trials, look at the bar chart. Which basis gives the most “deterministic” outcome (one probability close to 1)? What does that tell you about the state in that basis?
- How do the eigenvalues of the Pauli matrices relate to the possible measurement outcomes? Can you see from your plots that the expectation value is always between $-1$ and $+1$?
- Explain why $\langle 0 | U^\dagger Z U | 0 \rangle$ gives the same result as $\langle \psi | Z | \psi \rangle$. (Hint: what is $|\psi\rangle$ in terms of $U$ and $|0\rangle$?)
- In the scatter plot, why do all points lie on the unit circle? What would change if we applied a non‑unitary operation (like measurement) before computing the expectation values?

#### Bonus challenge (optional)
For one of your random sequences, compute the Bloch sphere angles $(\theta, \phi)$ of the final state using:
```math
\theta = \arccos(\langle Z \rangle), \quad \phi = \arg(\langle X \rangle + i\langle Y \rangle)
```
and verify that applying $U$ to $|0\rangle$ yields a state with those angles. (You can reconstruct $|\psi\rangle$ from $\theta$ and $\phi$ and compare with your computed state.)



## Part B: Modifying the PennyLane Predictor (30 points)

In Lecture 2 we built a single‑qubit variational circuit to learn the function $f(x) = \sin(x)$. The circuit was:

1. Start in $|0\rangle$.
2. Encode input $x$ via $R_y(x)$.
3. Apply trainable rotations $R_z(\theta_z)$, $R_y(\theta_y)$, $R_x(\theta_x)$.
4. Measure $\langle Z \rangle$.

We saw that such a circuit can only represent functions of the form $f(x) = A\cos x + B\sin x$ with $A^2+B^2 \le 1$.

### B.1 Modify the target function

Change the target function to $g(x) = \cos(2x)$ for $x \in [0, \pi]$. Generate a dataset of 200 points (using `torch.linspace`) and split into training (80%) and test (20%) sets as in the lecture.

- Train the same circuit (with three trainable parameters) on this new dataset. Use the same hyperparameters (learning rate, number of epochs).
- Plot the training/test loss curves and the final fit (model predictions vs. true $g(x)$).
- Report the final test loss and the trained parameters.

**Question:** Does the circuit manage to learn $\cos(2x)$ well? Why or why not? Relate your answer to the theoretical form $f(x) = A\cos x + B\sin x$. Can this form approximate $\cos(2x)$ over the interval? Explain.

### B.2 Try a different circuit architecture

Modify the circuit to use only two trainable rotations, e.g., $R_y(\theta_1)$ and $R_z(\theta_2)$ (removing $R_x$). Repeat the training for both $\sin(x)$ and $\cos(2x)$. 

- Plot the fits and compare performance.
- What functions can this reduced circuit represent? (Hint: analyze as in the lecture notes.)

### B.3 [Optional challenge] Learn a step function

Try to learn a step function:

```math
h(x) = \begin{cases}
-1 & \text{if } x < 0 \\
+1 & \text{if } x \ge 0
\end{cases}
```

for $x \in [-\pi, \pi]$. Use the original three‑rotation circuit. Can it fit a sharp step? Why? Discuss the limitations.

### B.4 Written summary

Write a short summary (5–10 sentences) explaining what you learned about the expressive power of single‑qubit variational circuits. Include:

- The general form of functions that can be represented.
- The effect of adding more trainable rotations.
- Why understanding these limitations is important for designing quantum machine learning models.

---

## Deliverables

Submit a single Jupyter notebook (or Python script with clear sections) containing:

- All code for Part A and Part B, with appropriate comments.
- The required outputs (printed values, plots).
- Answers to all questions (as markdown cells or comments).

Make sure your notebook runs from top to bottom without errors. You can share a colab file if you want.

---

## Hints and Notes

- Use `np.vdot` or `np.dot` with conjugate for inner products, but be careful with complex numbers. `np.vdot(a,b)` computes the conjugate of `a` times `b`.
- For matrix multiplication, use `@` or `np.matmul`.
- When checking equalities with floating‑point numbers, always use `np.allclose`.
- In Part B, you can start from the code provided in Lecture 2. Copy it into your notebook and modify as needed.
- If you use AI tools (like DeepSeek or ChatGPT) for coding help, note that in your submission and ensure you understand every line you write.

---

## Grading Rubric

A1-A11: each question is 7 points (7 points bonus)
B1-B4: each question is 10 points (10 points bonus).
**Total (without bonus): 100 points**
