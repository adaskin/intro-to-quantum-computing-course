# intro-to-quantum-computing-course
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

**Online available** notes-books:
- [A Course on the Theory of Quantum Computing](https://arxiv.org/pdf/2507.11536) by John Watrous, see also his IBM-Qiskit Textbook (free online: [qiskit.org/textbook](https://quantum.cloud.ibm.com/learning/en/courses))
- [Quantum country](https://quantum.country/) by Andy Matuschak and [Michael Nielsen](https://michaelnielsen.org/)  see also his great [introduction to neural networks](http://neuralnetworksanddeeplearning.com/)
- [Pennylane tutorials and documentation](https://pennylane.ai/qml)
  

**Software & Frameworks:**
- Qiskit,PennyLane, PyTorch, NumPy/SciPy 
- You can also use other software packages



### **Grading Policy**
- 40% 3-5 assignments including python implementations
- 20% Midterm
- 40% Final


### **Tentative: Course Schedule (15 Weeks)**

#### **Unit 1: Mathematical Foundations (Weeks 1-3)**
**Week 1: Classical vs. Quantum Information**
- Bits vs. qubits: the need for new computational models
- Stern-Gerlach experiment: sequential measurements as computation
- Polarization: a complete qubit analogy
- Quantum postulates (preview)
- **AI Tool Demo:** Using ChatGPT to generate Stern-Gerlach analogies
- **Assignment 1 Released:** Classical reversible circuits and quantum intuition

**Week 2: Linear Algebra for Quantum Computing**
- Dirac notation: kets, bras, and inner products
- State vectors in ℂ² and ℂⁿ
- Born rule and probability interpretation
- Projective measurements
- **AI Tool Demo:** Using DeepSeek for linear algebra calculations
- **Math Supplement:** Complex vector spaces, Hermitian operators

**Week 3: Quantum Operations & Circuit Model**
- Unitary operators as quantum gates
- Pauli matrices and their algebraic properties
- Hadamard and phase gates
- Quantum circuit diagrams and composition rules
- **AI Tool Demo:** Using GitHub Copilot for Qiskit boilerplate code
- **Assignment 1 Due; Assignment 2 Released:** Single-qubit state evolution and measurements

#### **Unit 2: Quantum Information Theory (Weeks 4-6)**
**Week 4: Multi-Qubit Systems & Entanglement**
- Tensor product spaces
- Entangled states: Bell basis and properties
- Schmidt decomposition
- Partial trace and reduced density operators
- **Math Supplement:** Tensor algebra and density matrices

**Week 5: Quantum Protocols & Fundamental Limits**
- Quantum teleportation: protocol and proof of correctness
- Superdense coding
- No-cloning theorem: proof and implications
- Quantum key distribution (BB84) analysis
- **Assignment 2 Due; Assignment 3 Released:** Entanglement and basic protocols

**Week 6: Quantum Algorithms I: Oracle Problems**
- Quantum oracle model and query complexity
- Deutsch-Jozsa algorithm: problem, circuit, and analysis
- Bernstein-Vazirani algorithm
- Simon's problem (introduction)
- **AI Tool Demo:** Using AI to generate oracle functions
- **Midterm Review Session**

#### **Unit 3: Core Quantum Algorithms (Weeks 7-9)**
**Week 7: Quantum Fourier Transform**
- **Midterm Exam** (covers Weeks 1-6)
- Discrete Fourier transform review
- Quantum Fourier Transform (QFT) circuit construction
- Phase estimation algorithm

**Week 8: Shor's Algorithm**
- Period finding and the factoring problem
- Shor's algorithm: full circuit analysis
- Continued fractions and correctness proof
- Impact on cryptography and RSA
- **AI Tool Demo:** Using AI to explain number theory concepts

**Week 9: Quantum Search & Optimization**
- Grover's algorithm: geometric analysis
- Optimality proof (quadratic speedup)
- Amplitude amplification generalization
- Applications to unstructured search and SAT
- **Assignment 3 Due; Assignment 4 Released:** Quantum algorithm analysis

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
- **Assignment 4 Due; Assignment 5 Released:** Error correction and variational methods

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
- **Assignment 5 Due; Assignment 6 Released:** Complexity and information theory problems

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

