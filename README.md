# intro-to-quantum-computing-course
*Introduction to quantum information and computing with an emphasis on AI*  
*Prepared/written with the help of DeepSeek AI*

## **Quantum Computing & Information: From Qubits to AI**
**Course Numbers:** BIL474/YZM 522 (Undergraduate/Graduate)  
**Instructor:** Ammar Daskin  
**Semester:** Spring 2026  
**Office Hours:** [Day/Time], [Location]  
**Course Website:** [https://github.com/adaskin/intro-to-quantum-computing-course](https://github.com/adaskin/intro-to-quantum-computing-course)

### **Course Description**
This course introduces quantum computing and information from a computer science perspective. Using mathematical rigor and algorithmic analysis, we develop quantum mechanics from first principles, establish the quantum circuit model, analyze key algorithms and protocols, and explore applications in optimization and machine learning. After mathematical formalism, computational complexity, and quantum information theory, in the final weeks of the course, the emphasis is given on the application of quantum computing to AI and optimization.

### **Prerequisites**
- **Required:** Data Structures & Algorithms
- **Required:** Linear Algebra (vector spaces, inner products, eigenvalues, unitary matrices)
- **Required:** Discrete Mathematics or equivalent
- **Required:** Python programming proficiency (assignments include implementation components)

### **Learning Outcomes**
Upon completion, students will be able to:
1. **Formalize** quantum states, operations, and measurements using Dirac notation and linear algebra
2. **Prove** fundamental results (no-cloning, teleportation correctness, algorithm bounds)
3. **Analyze** quantum algorithms' computational complexity and advantages
4. **Design** quantum circuits for specified computational tasks
5. **Compare** quantum and classical approaches to optimization, machine learning, and information processing
6. **Critique** research papers and claims about quantum computing capabilities

### **Textbooks & Resources**
**Primary Textbooks:**
1. **"Quantum Computation and Quantum Information"** by Nielsen & Chuang (primary theoretical reference)
2. **"An Introduction to Quantum Computing"** by Kaye, Laflamme, Mosca (CS-focused theory)
3. **"Quantum Computing for Computer Scientists"** by Yanofsky & Mannucci (accessible approach)

**Software & Frameworks:**
- **Qiskit** (IBM Quantum): Primary quantum programming framework
- **PennyLane** (Xanadu): For quantum machine learning implementations
- **NumPy/SciPy:** For classical simulation components
- **You can also use other software packages**

**Supplementary Resources:**
- Lecture notes and selected research papers
- Qiskit Textbook (free online: [qiskit.org/textbook](https://quantum.cloud.ibm.com/learning/en/courses))
- Pennylane tutorials and documentation


### **Grading Policy**
| Component | Weight | Description |
|-----------|--------|-------------|
| **Problem Sets & Python Simulations** | 30% | 3-5 assignments blending proofs, analysis, and implementations |
| **Midterm Exam** | 20% | covers Weeks 1-8 |
| **Class/Quiz Participation** | 10% | Determined by in-class quiz questions |
| **Final Exam** | 40% | Comprehensive, covers entire semester |

**Late Policy:** Assignments submitted within 24 hours of deadline receive 20% penalty. No submissions accepted after 24 hours without prior arrangement.


### **Course Schedule (15 Weeks)**

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
- Student-led discussions on AI-quantum synergy
- Course synthesis: what's proven, what's plausible, what's speculative
- **Final Papers Due**

### **Assignment Structure**
Each assignment will typically include:

1. **Proof Problems:** Formal proofs of quantum information results
2. **Circuit Analysis:** Designing and analyzing quantum circuits for given tasks
3. **Computational Problems:** Calculating probabilities, state evolutions, etc.
4. **Algorithm Analysis:** Analyzing time/space complexity of quantum algorithms
5. **Python Implementation:** Hands-on experience with Qiskit or PennyLane

**Example Assignment Topics:**
- **A1:** Classical reversible computing and quantum notation (Python: basic gate implementation)
- **A2:** Single-qubit states and measurements (Python: state tomography simulation)
- **A3:** Entanglement and teleportation proofs (Python: teleportation circuit with noise)
- **A4:** Quantum algorithm correctness and complexity (Python: Grover's algorithm implementation)
- **A5:** Error correction and variational methods (Python: VQE for small molecules)
- **A6:** Advanced topics synthesis (Python: quantum kernel method for classification)

### **Final Project Options**
Students may choose from:
1. **Implementation Project:** Implement and analyze a quantum algorithm from recent literature
2. **Survey Paper:** Comprehensive review of a quantum computing subfield
3. **Research Proposal:** Original idea for quantum-AI research with feasibility analysis
4. **Tool Development:** Create educational tools or visualizations for quantum concepts

**Project Timeline:**
- Week 8: Project proposal due
- Week 10: Literature review/design document due
- Week 13: Progress report
- Week 15: Final presentation and paper submission

### **Getting Help**
1. **Office Hours:** [Schedule]
2. **Course Forum:**  classroom.google.com
4. **Programming Help:** TA-led Qiskit/PennyLane troubleshooting sessions


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

