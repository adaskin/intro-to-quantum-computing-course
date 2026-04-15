# Introduction to Quantum Information Processing
*Introduction to quantum information and computing with an emphasis on AI*  
*Prepared/written with the help of DeepSeek AI*


### **Course Description**
This course introduces quantum computing and information from a computer science perspective. Using mathematical rigor and algorithmic analysis, we develop quantum mechanics from first principles, establish the quantum circuit model, analyze key algorithms and protocols, and explore applications in optimization and machine learning. After mathematical formalism, computational complexity, and quantum information theory, in the final weeks of the course, the emphasis is given on the application of quantum computing to AI and optimization.

### **Prerequisites**
- Data Structures & Algorithms (basics)
- Linear Algebra (basics)
- Discrete Mathematics or equivalent
- Python programming (assignments include implementation components)

### **Learning Outcomes**
Upon completion, students will be able to:
1. Formalize quantum states, operations, and measurements using Dirac notation and linear algebra
2. Understand fundamental results (no-cloning, teleportation correctness, algorithm bounds)
3. Analyze quantum algorithms' computational complexity and advantages
4. Design quantum circuits for specified computational tasks
5. Compare quantum and classical approaches to optimization, machine learning, and information processing
6. Critique research papers and claims about quantum computing capabilities

### **Textbooks & Resources**
No required textbook. But, lecture notes are mostly based on
- "Quantum Computation and Quantum Information" by Nielsen & Chuang (primary theoretical reference)
- "An Introduction to Quantum Computing" by Kaye, Laflamme, Mosca (CS-focused theory)
- "Quantum Computing for Computer Scientists" by Yanofsky & Mannucci (CS-focused and more basics)  
- "Quantum Computing: A Gentle Introduction" by Eleanor G. Rieffel and Wolfgang H. Polak. 

**Online available** notes-books:
- [A Course on the Theory of Quantum Computing](https://arxiv.org/pdf/2507.11536) by John Watrous, see also his IBM-Qiskit Textbook (free online: [qiskit.org/textbook](https://quantum.cloud.ibm.com/learning/en/courses))
- [Quantum country](https://quantum.country/) by Andy Matuschak and [Michael Nielsen](https://michaelnielsen.org/)  see also his great [introduction to neural networks](http://neuralnetworksanddeeplearning.com/)
- [Pennylane tutorials and documentation](https://pennylane.ai/qml)
- [Quantum Algorithm Zoo](https://quantumalgorithmzoo.org/)
- [From Classical to Quantum Shannon Theory, Mark W. Wilde](https://arxiv.org/pdf/1106.1445)
**Software & Frameworks:**
- Qiskit,PennyLane, PyTorch, NumPy/SciPy 
- You can also use other software packages



### **Grading Policy**
- 40% 3-5 assignments including python implementations (you can work as groups of 1-3 people)
- 20% Midterm
- 40% Final


### **Course Schedule and Lecture Notes**
**Week 1: Classical vs. Quantum Information**
reading: [Thirty years of quantum computing](https://doi.org/10.1088/2058-9565/ade0e4),David P DiVincenzo, Quantum Sci. Technol. 10 (2025) 030501. 
lecture note: [lecture1](lecture1.md)  
- Bits vs. qubits: the need for new computational models
- Stern-Gerlach experiment: sequential measurements as computation
- Polarization: a complete qubit analogy
- Quantum postulates (preview)
- **AI Tool Demo:** Using ChatGPT to generate Stern-Gerlach analogies


**Week 2: qubit representation, rotation gates, variational quantum circuits**
lecture note: [lecture2](lecture2.md)  
- Vector basics
- Postulate of quantum mechanics,
- Bloch sphere, rotation gates
- Simulating rotation gates in python
- Single qubit predictor for y = sin(x) with pennylane-torch
- Multiple qubits, entanglement
- **AI Tool Demo:** Using DeepSeek for writing code to suggest PennyLane syntax for parameterized circuits, implement the train/test split with scikit‑learn and PyTorch, debug the training loop, generate explanatory comments and exercises.
- **Math Supplement:** Complex vector spaces, Hermitian operators
- **Reading (optional physics for curious): [Geometric Quantum mechanics (physics-geometry)](https://arxiv.org/abs/quant-ph/9906086)**

**Week 3-4:  From Vectors to Entanglement – Math Foundations & Quantum Communication**
*Bridging Linear Algebra Review with Quantum Protocols*
lecture note: [lecture3][lecture3.md]  
- **Math Review:** Vectors, inner products, eigenvalues, Hermitian matrices, tensor products
- **Dirac Notation Deep Dive:** Bras, kets, and operators
- **Composite Systems:** Building multi-qubit states
- **Entanglement:** The "spooky" correlation explained mathematically
- **Bell States:** Maximally entangled states
- **Quantum Communication Protocols:**
   - Superdense coding
   - Quantum teleportation
- **Quantum Key Distribution (BB84):** Real-world application
- **AI Tool Demo:** Using Numpy and PennyLane Demos for creating and measuring Bell states and simulating BB84

**Week 5-6: Multi-Qubit Systems & Entanglement**
[lecture4](lecture4.md)
- Why we need a more general description than state vectors  
- **Density matrices**: definition, properties, examples  
- **Partial trace** – how to “ignore” part of a system  
- **Reduced density matrices** – what a subsystem looks like alone  
- **Purity** – measuring how mixed a state is  
- **Schmidt decomposition** – the mathematical structure of bipartite states  
- **Quantifying entanglement** – entropy of entanglement  
- **No‑cloning theorem** – a simple proof  
- **Applications**: Teleportation & Superdense coding (recap with density matrices)  
- **PennyLane demos** – computing reduced states and entanglement  


**Week 8-9: Quantum Algorithms I: Oracle Problems**
[lecture5](lecture5.md)
- Quantum oracle model and query complexity
- Deutsch-Jozsa algorithm: problem, circuit, and analysis
- Bernstein-Vazirani algorithm
- Simon's problem (introduction)
- **AI Tool Demo:** Using AI to generate oracle functions

**Week 9-10: Quantum Fourier Transform**
[lecture6a](lecture6-period-finding-factoring/lecture6a-dft.md)
[lecture6b](lecture6-period-finding-factoring/lecture6b-qft.md)
[lecture6c](lecture6-period-finding-factoring/lecture6c-factoring.md)
- Discrete Fourier transform review
- Quantum Fourier Transform (QFT) circuit construction
- Period finding and the factoring problem
- Shor's algorithm: full circuit analysis
- Continued fractions and correctness proof
- Impact on cryptography and RSA
- **AI Tool Demo:** Using AI to explain periods and number theory concepts, discuss its impact on cryptography

---
The remaining part is to be edited

---


**Week 9: Quantum Search & Optimization**
- Grover's algorithm: geometric analysis
- Optimality proof (quadratic speedup)
- Amplitude amplification generalization
- Applications to unstructured search and SAT

#### **Unit 4: Advanced Topics (Weeks 10-12)**
**Week 10: Quantum Error Correction**
- Classical vs. quantum error models
- Bit-flip and phase-flip codes
- Shor's 9-qubit code
- Stabilizer formalism introduction
- Quantum threshold theorem

**Week 11: Variational Quantum Algorithms**
- Parameterized quantum circuits
- Variational Quantum Eigensolver (VQE)
- Quantum Approximate Optimization Algorithm (QAOA)
- Classical optimization loops
- **AI Tool Demo:** Using AI to suggest optimization strategies

**Week 12: Quantum Machine Learning Foundations**
- Quantum data encoding strategies
- Quantum neural networks as variational circuits
- Quantum kernels and feature spaces
- Barren plateaus and trainability
- **AI Tool Demo:** Using AI to generate quantum dataset encodings
- **Final Project Topics Distributed**

#### **Unit 5: Frontiers & Applications (Weeks 13-15)**
**Week 13: Quantum Complexity Theory**
- Complexity classes: BQP, QMA, and relationships to classical classes
- Quantum supremacy arguments
- Oracle separations
- The hidden subgroup problem

**Week 14: Quantum Information & Entanglement Theory**
- Entanglement measures
- Quantum channels and CPTP maps
- Quantum Shannon theory (overview)

**Week 15: Presentations & Future Directions**
- **Final Project Presentations**



## Academic Integrity
All work must be your own. Collaboration on assignments is allowed but must be documented. AI tool usage must be disclosed as described below.

### **AI Tools in Quantum Computing Education**
This course encourages the **responsible, transparent use of modern AI tools** such as Deepseek, Claude, ChatGPT, Gemini, Github Copilot as learning aids and productivity enhancers. When used properly, these tools can accelerate understanding, help debug code, and provide alternative explanations of complex concepts.

#### **Approved Uses:**
1. **Concept Clarification:** Asking for alternative explanations of quantum phenomena
   - *Example:* "Explain quantum entanglement using a CS analogy"
   - *Example:* "Compare Grover's algorithm to classical search algorithms"

2. **Code Assistance:**
   - Debugging/designing quantum circuits
   - Optimizing classical optimization loops for VQE/QAOA
   - Converting between different quantum programming frameworks

3. **Mathematical Derivation Help:**
   - Step-by-step explanations of quantum algorithm proofs
   - Linear algebra calculations and validations
   - Probability amplitude computations

4. **Literature Review Support:**
   - Summarizing quantum computing research papers
   - Identifying key contributions in quantum ML papers
   - Finding connections between different research areas

#### **Prohibited Uses:**
- Generating complete solutions to assignment problems without understanding each step
- Writing entire proofs without understanding each step
- Submitting AI-generated work as your own without attribution
- Using AI during exams or quizzes

#### **Required Attribution:**
When using AI tools for assignments, you **must** include an "AI Usage Statement" describing:
1. Which tool(s) you used
2. For what specific purposes
3. What you learned from the interaction

*Example statement: "I used DeepSeek to help design and debug my QFT circuit implementation when the phase estimation was giving incorrect results. The AI suggested checking the qubit ordering in my controlled rotations, which helped me identify the indexing error."*

