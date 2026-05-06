# Lecture 8a – Recap: Period Finding, Phase Estimation, and Eigenvalues

---

## Welcome Back! (After Break)

- Last time: **Shor's algorithm** factors integers exponentially faster
- Today: Recap the **quantum tools** behind it
- Then: **What breaks** (RSA, Bitcoin, banking)
- Finally: **How the world prepares** (post‑quantum crypto)

---

## Why Should a CS Student Care?

```
Your bank account    WhatsApp messages    Bitcoin wallet
        ↓                   ↓                   ↓
    protected by        protected by       protected by
    RSA / ECC           RSA / ECC           ECC
        ↓                   ↓                   ↓
    Shor's algorithm BREAKS all of them
```

> **A large quantum computer → your digital life is an open book**

---

##  DFT / QFT – Finding Frequencies

### Classical DFT (Discrete Fourier Transform)

```
Input:  signal = [2, 1, 0, 1, 2, 1, 0, 1]  (periodic)
        ↓ DFT
Output: peak at frequency = 4  (the period)
```

**Analogy:**  
A prism splits light into colours – DFT splits a signal into its frequency components.

---

### Quantum Fourier Transform (QFT)

- Works on **quantum amplitudes**, not classical numbers
- Input: superposition $\sum_x a_x |x\rangle$
- Output: superposition where amplitudes are Fourier transformed

**Key property:**  
If the input has **period $r$**, the QFT output has **peaks at multiples of $M/r$**

```
Periodic superposition          After QFT
   |0⟩ + |r⟩ + |2r⟩ + ...   →   peaks at |0⟩, |M/r⟩, |2M/r⟩, ...
```

---

## Visualizing QFT (No Math Overload)

```
Before QFT:   █░░█░░█░░█░░   (period = 3)
After QFT:    ███░░░░░░░░░░   (peak at position related to 3)
```

**Magic lens:**  
QFT "sees" the repeating pattern and makes the distance between repeats light up.

> **Takeaway:** QFT finds periods in **one step** – classically you'd need to check many positions.

---

## Period Finding – The Engine of Shor

### Problem Definition

Given a function $f: \{0,1\}^n \to \{0,1\}^m$ with property:
$$f(x) = f(x+r) \quad \text{for all } x$$
Find the smallest **period** $r > 0$.

---

**Concrete example (modular exponentiation):**
$$f(x) = a^x \bmod N$$
Period $r$ satisfies $a^r \equiv 1 \pmod{N}$.

---

### Quantum Period Finder – Steps (High Level)

```
Step 1: Create superposition over x
        |ψ⟩ = (1/√M) Σ|x⟩|0⟩

Step 2: Compute f(x) into second register
        |ψ⟩ = (1/√M) Σ|x⟩|f(x)⟩

```

---

```
Step 3: Measure second register
        First register collapses to periodic comb: |x₀⟩ + |x₀+r⟩ + |x₀+2r⟩ + ...

Step 4: Apply QFT to first register
        Peaks appear at multiples of M/r

Step 5: Measure → get k·(M/r), use continued fractions to recover r
```

---

## Example: Small Period Finding (Illustrative)

Let $f(x) = (x \bmod 3)$ for $x=0..7$ (period $r=3$)

```
x:    0 1 2 3 4 5 6 7
f(x): 0 1 2 0 1 2 0 1

After measuring f(x)=0, we have: |0⟩ + |3⟩ + |6⟩

Apply QFT (size M=8) → peaks at |0⟩, |8/3⟩≈|2.66⟩, |16/3⟩≈|5.33⟩
Measure → get 3 or 6 (with high prob) → continued fractions → r=3
```

> **No need to compute all 8 x's – one quantum run suffices!**

---

## Part 3: Phase Estimation – Finding Eigenvalues

### What is an Eigenvalue?

For a matrix $U$ and vector $|\psi\rangle$:
$$U|\psi\rangle = \lambda |\psi\rangle$$
- $\lambda$ = eigenvalue (a complex number, often $e^{i\phi}$)
- $|\psi\rangle$ = eigenvector

**Example (Pauli Z):**
$$Z = \begin{bmatrix}1 & 0 \\ 0 & -1\end{bmatrix}, \quad Z|0\rangle = (+1)|0\rangle, \quad Z|1\rangle = (-1)|1\rangle$$

---

### Phase Estimation Problem

Given:
- A unitary $U$ (as a circuit)
- An eigenvector $|\psi\rangle$ (we can prepare)
- Unknown eigenvalue $e^{i 2\pi \theta}$ (with $\theta \in [0,1)$)

**Goal:** Estimate $\theta$ to high precision.

---

### How Phase Estimation Works (Pictorial)

```
Control qubits: |0⟩|0⟩...|0⟩  --(Hadamard)→  superposition of all |x⟩

Target:        |ψ⟩ (eigenvector)

```

---
```
Apply controlled‑U^{2^j} for each qubit j
→ Each |x⟩ gets phase e^{i 2π θ x}

Result: (1/√M) Σ e^{i 2π θ x} |x⟩ ⊗ |ψ⟩

```

---

```

Apply inverse QFT to control register → measures |M·θ⟩

Output: integer closest to M·θ → estimate θ = output / M
```

---

## Generic Matrix Eigenvalue Example

Suppose we have a **large matrix** $H$ (e.g., Hamiltonian of a molecule).  
We want its smallest eigenvalue (ground state energy).

- Build unitary $U = e^{-i H t}$ (quantum simulation)
- Prepare an approximate ground state $|\psi\rangle$
- Run phase estimation → get eigenvalue $\lambda$

---

**Applications:** Drug design, battery materials, high‑temperature superconductors.

> **Why quantum?** Classical eigenvalue algorithms scale poorly for large matrices. Phase estimation gives exponential speedup **in some cases**.

---

## Connection: Period Finding = Special Case of Phase Estimation

Recall $U_a: |y\rangle \mapsto |a y \bmod N\rangle$

- Eigenvectors: $|u_s\rangle = \frac{1}{\sqrt{r}} \sum_{k=0}^{r-1} e^{-2\pi i s k / r} |a^k \bmod N\rangle$
- Eigenvalues: $e^{2\pi i s / r}$

Running phase estimation on $U_a$ with input $|1\rangle$ yields a random $s/r$ → recover $r$.

> So **period finding = eigenvalue estimation** for the modular multiplication operator.

---

## Summary

| Concept | What it does | Why important |
|---------|--------------|----------------|
| **DFT / QFT** | Finds frequencies/periods | Core of Shor's algorithm |
| **Period finding** | Finds $r$ for $f(x+r)=f(x)$ | Reduces factoring to period |
| **Phase estimation** | Finds eigenvalues of $U$ | General tool for many problems |

**Example of phase estimation:**  
Finding vibration modes of a bridge (eigenvalues) → predict where it might break.

---

# Factoring Integers, RSA, and What Breaks

> *“Quantum computers don't break everything – but they break the math behind most of today's secure communication.”*

---

## From Period Finding to Factoring

### The Key Connection

If you can find the period $r$ of
$$f(x) = a^x \bmod N$$
then you can factor $N$ (with high probability).

**Why?**  

---

When $r$ is even:
$$a^{r/2} \not\equiv \pm 1 \pmod{N}$$
Then $\gcd(a^{r/2} - 1, N)$ is a non‑trivial factor of $N$.

---

## Example: Factor $N = 15$

Choose $a = 7$ (coprime to 15)

```
x:      0   1   2   3   4   5   6   7
7^x:    1   7   4  13   1   7   4  13
        ↑               ↑
        period r = 4
```

$r = 4$ (even), $a^{r/2} = 7^2 = 49$

$\gcd(49 - 1, 15) = \gcd(48, 15) = 3$

$15 / 3 = 5$ → factors: $3$ and $5$ ✅

---

## Factoring via Period Finding – Recipe

```
Input:  N (odd composite, not a prime power)
Step 1: Pick random a in [2, N-1]
Step 2: Compute g = gcd(a, N). If g > 1 → g is factor → done.
Step 3: Use quantum period finding to find r = order of a mod N
Step 4: If r is even and a^{r/2} ≠ ±1 mod N:
           p = gcd(a^{r/2} - 1, N)  → factor
        else: pick another a and repeat
```

**Success probability:** High after a few tries.

---

## RSA – How Your Data Is Protected

### RSA Basics (Rivest–Shamir–Adleman)

```
Choose two large primes p, q
Compute N = p × q      (public modulus)
Choose e (public exponent, often 65537)
Compute d = e⁻¹ mod (p-1)(q-1)  (private exponent)

Public key:  (N, e)
Private key: (p, q, d)   or just d
```

---

### Encryption and Decryption

```
Alice (wants to send secret to Bob)

Message m (integer < N)
Ciphertext c = m^e mod N   (using Bob's public key)

Bob (receives c)
m = c^d mod N              (using his private key)
```

**Security:** Without $p$ and $q$, you cannot compute $d$ from $e$ and $N$ – unless you can factor $N$.

---

### RSA in Real Life

```
HTTPS (padlock in browser)   → RSA or ECC
Email encryption (PGP)       → RSA
Digital signatures           → RSA
Banking cards                → RSA
SSH (secure shell)           → RSA
```

> **Every padlock you see** likely relies on the hardness of factoring or discrete logarithms.

---

**Example Mobile App-Server: RSA Handshake**
```text
Step 1: App requests secure connection (e.g., login / get token)
   📱 App ────────────────────────────►  ☁️ Server
        "Hello, I want to talk securely" 
```     
```text
Step 2: Server sends its RSA certificate (contains public key)
   📱 App ◄───────────────────────────── ☁️ Server
          Certificate + Public Key (K_pub)   
```

---
```text
Step 3: App verifies certificate (chain of trust – browser already trusts root)
   📱 App                                         ☁️ Server
       • Check signature using built‑in root CA      
       • Is domain name correct?                     
      • Not expired?                                
                                                    
    [ ✅ Certificate OK ]                

```

---
```text
Step 4: App generates a random symmetric key (e.g., for AES)

   📱 App
       Generate random K_sym (128 bits) 
     – this will encrypt all future messages
```

---
```text
Step 5: App encrypts K_sym with server's RSA public key

   📱 App
    Encrypted_K = RSA_encrypt(K_sym, K_pub)
    (only server can decrypt, because only server has the private key)
```

---
```text
Step 6: App sends Encrypted_K to server

   📱 App ──────────────────────► ☁️ Server
                  Encrypted_K                       
```

---

```text
☁️ Server decrypts using its private key → recovers K_sym
     
      - Both now share same K_sym – secure channel established!

Result: All further communication (messages, photos, tokens) encrypted with AES‑K_sym.
```
---


## Blockchain: Needs a "Digital Handwriting"
**Digital Signatures**
- In blockchain, every transaction must be signed with your **Private Key**.
- Anyone can verify it using your **Public Key**.

**The Challenge:** Traditional RSA signatures are too slow and create keys that are too large for resource-constrained nodes.

---

###  Elliptic Curves: The Math Behind the Magic

**Key Equation:**
\[
y^2 = x^3 + ax + b
\]
(Where \( 4a^3 + 27b^2 \neq 0 \) to avoid singularities)
- \( y^2 = x^3 - x \)
  
**The "Clock Arithmetic" Property:**
- Instead of a straight line (like RSA), we define a special "addition" rule: \( P + Q = R \).
- Draw a line through two points on the curve. It intersects a third point. Mirror it over the x-axis. That is \( P+Q \).

---

**The One-Way Function:**
- **Easy:** Start with a point \( G \). Add it to itself \( k \) times (\( G + G + ... + G \)) to get a new point \( K \).
- **Impossible (so far):** Given \( K \) and \( G \), figure out \( k \).
- *This is called the Elliptic Curve Discrete Logarithm Problem (ECDLP).*

---

###  Why Blockchain Loves ECC (vs. RSA)
Efficiency = Security + Speed

**The Security Comparison:**
| Security Level | RSA Key Size | ECC Key Size |
| :--- | :--- | :--- |
| **Moderate** | 1024 bits | 160 bits |
| **Strong** | 3072 bits | **256 bits** |
| **Paranoid** | 15360 bits | 512 bits |

---

**Why this matters for Blockchain:**
1.  **Smaller Keys:** A Bitcoin private key is just 32 bytes (easy to memorize/backup).
2.  **Faster Signatures:** ECDSA (Elliptic Curve Digital Signature Algorithm) generates signatures much faster than RSA.
3.  **Less Bandwidth:** Small signatures (64 bytes) mean transactions fill fewer blocks.

> **Bottom Line:** ECC secures billions of dollars with the same energy as a light bulb, while RSA would be like mining coal.

---

###  Real Blockchain Example (Bitcoin & Ethereum)

`Private Key (Random #)` --> `Elliptic Curve (secp256k1)` --> `Public Key` --> `Hash` --> `Wallet Address`

**The Specific Curve:** `secp256k1`
- Standardized by Bitcoin's creator (Satoshi).
- Defined as: \( y^2 = x^3 + 7 \)
- Chosen because it is efficient to compute and avoids potential backdoors found in other NIST curves.

---

**How a Transaction Works:**
1.  You hold **Private Key** \( k \).
2.  The network knows your **Public Key** \( k \times G \).
3.  You create a transaction (e.g., "Send 2 BTC to Bob").
4.  You use \( k \) to sign the hash of that transaction → (r, s).
5.  Miners use your Public Key to verify (r, s) **without ever knowing** \( k \).

---

###  The Danger of Bad Randomness (The PlayStation 3 Hack)

- The PS3 used the same ECDSA algorithm as Bitcoin to sign games.
- Sony forgot to use a **random nonce (k)** . They used a constant number.

**The Math Breach:**
- If you sign two transactions with the same \( k \), you can solve for the Private Key using simple algebra.
- Hackers did this, and Sony lost the ability to lock down the PS3.

**Lesson for Blockchain:**
- If your wallet's random number generator is broken (e.g., Android bug in 2013), hackers can steal your funds instantly.
- This is why hardware wallets have physical random number generators.

---

## What a Quantum Computer Breaks

*If we had a quantum computer that solves the ECDLP, would it steal your Bitcoin? (
    Yes, unless you moved it to a **quantum-resistant** address.
    
- Points on the curve form an **abelian group** under “point addition” 

---
### Asymmetric Cryptography (Public Key)

| Algorithm | Based on | Quantum Status |
|-----------|----------|----------------|
| **RSA** | Factoring | ❌ Broken (Shor) |
| **ECC** | Elliptic curve discrete log | ❌ Broken (Shor) |
| **Diffie‑Hellman** | Discrete log | ❌ Broken (Shor) |
| **DSA** | Discrete log | ❌ Broken |

**Impact:** Almost all secure internet communication becomes readable.

---

### Symmetric Cryptography (AES, ChaCha20)

| Key size | Classical security | Quantum (Grover) |
|----------|-------------------|------------------|
| AES‑128 | 2¹²⁸ operations | 2⁶⁴ operations (breakable) |
| AES‑256 | 2²⁵⁶ operations | 2¹²⁸ operations (still secure) |

**Solution:** Double key length → remain safe.

> **Takeaway:** Symmetric crypto survives. Public key crypto dies.

---

### Hash Functions (SHA‑256, SHA‑3)

- Grover's algorithm can find preimages **quadratically** faster
- But 256‑bit hash → 2¹²⁸ quantum operations → still impractical
- No fundamental break (unlike factoring)

**So:** Hash‑based signatures (e.g., SPHINCS+) are **quantum‑resistant**.

---

## Visual Summary: What Breaks vs. What Survives

```
┌─────────────────────────────────────────────────────┐
│                     QUANTUM THREAT                   │
├─────────────────────────┬───────────────────────────┤
│   COMPLETELY BROKEN      │      LARGELY SAFE         │
│   (Shor's algorithm)     │   (with longer keys)      │
├─────────────────────────┼───────────────────────────┤
│ • RSA                    │ • AES‑256                 │
│ • ECC (Bitcoin)          │ • ChaCha20                │
│ • Diffie‑Hellman         │ • SHA‑256/512             │
│ • DSA                    │ • Hash‑based signatures   │
└─────────────────────────┴───────────────────────────┘
```

---

## The Hidden Subgroup Problem (HSP) – Abstraction

**Abelian HSP** (group is commutative)  
- Example: integers under addition modulo N  
- Shor's algorithm solves it → factoring, discrete log  

**Non‑abelian HSP** (group is not commutative, e.g., matrix groups)  
- No known efficient quantum algorithm  
- **Potential source of post‑quantum crypto**

```
Abelian groups      →  vulnerable (Shor)
Non‑abelian groups  →  possibly safe (conjectured)
```

---

## Example: Elliptic Curve Cryptography (ECC) is Abelian

- Elliptic curve points form an **abelian group**
- Shor's algorithm solves discrete log on abelian groups
- **Result:** Bitcoin's signatures are quantum‑vulnerable

**Bitcoin addresses**  
Public key visible on blockchain → quantum computer computes private key → steal coins.

> **But** blockchain can migrate to non‑abelian / lattice / hash‑based signatures.

---
## Abelian vs Non‑Abelian Groups

- **In classical cryptography**  
  RSA, DH, ECC all rely on **abelian** groups ⇒ hard problems like factoring, discrete log.

- **Quantum threat**  
  **Shor’s algorithm** efficiently solves the *abelian* hidden subgroup problem, breaking all such schemes.  
  Non‑abelian group problems are not directly broken by Shor, motivating post‑quantum exploration.

> ⚡ *Why this matters for quantum information: abelian group‑based crypto falls to quantum computers.*
> 
---

## Summary 

| Concept | Status after quantum computer |
|---------|-------------------------------|
| Factoring (RSA) | ❌ Broken |
| ECC (Bitcoin, Signal) | ❌ Broken |
| AES‑256 | ✅ Safe (with double keys) |
| SHA‑256 | ✅ Safe |
| Non‑abelian crypto | ✅ Possibly safe |


# Lecture 8c – Post‑Quantum Cryptography, Blockchain & Industry Impact

*How the world prepares – post‑quantum cryptography, blockchain migration, banking roadmaps.*

---

## Recap: What We Learned So Far

```
QFT finds periods → Period finding → Phase estimation
Period finding → Factoring → RSA/ECC broken
```

**Now:** The world is preparing for the quantum computers.

---

## Abelian vs. Non‑Abelian HSP (High Level)

### Hidden Subgroup Problem (HSP)

Given: Group $G$, function $f$ constant on cosets of a hidden subgroup $H$

Goal: Find $H$

```
Shor solves:  G = Z (integers) or Z_n → factoring, discrete log
              (abelian groups)

Unsolved:     G = non‑abelian (e.g., symmetric group, matrix groups)
```

---

### Why Non‑Abelian Matters

```
Abelian groups (commutative)     Non‑abelian (non‑commutative)
        a + b = b + a                   a * b ≠ b * a
        
        Example: integers               Example: 2×2 matrices
        ↓                               ↓
        Vulnerable to Shor               No known quantum attack
        (RSA, ECC broken)                (possible post‑quantum)
```

> **Opportunity:** Build cryptography on non‑abelian groups → quantum‑resistant?

---

## Blockchain – Why Your Crypto Is at Risk

### How Bitcoin Signatures Work (Simplified)

```
Private key (secret number)  →  Public key (point on elliptic curve)
                                       ↓
                                  hash → Address
```

To spend coins: Sign transaction with private key  
Anyone verifies using public key.

**Attack:** Quantum computer sees public key (on blockchain) → computes private key (Shor on ECC) → steals coins.

---

### Vulnerability Timeline

```
Today:     Public keys are visible on blockchain
           (Bitcoin, Ethereum, all UTXO‑based)

‑2030: Estimated time when first ECC‑breaking quantum computer appears

After that: All wallets with exposed public keys are compromised
```

**Safe wallets:** Those that never reused an address (public key not exposed until spending). But still risky.

---

##  Solutions – Post‑Quantum Cryptography (PQC)

### NIST Post‑Quantum Competition (Started 2016)

```
Finalists selected 2022:

Encryption / Key exchange:
    • CRYSTALS‑Kyber (lattice‑based) – winner
    • Classic McEliece (code‑based)
    • HQC (code‑based)

Digital Signatures:
    • CRYSTALS‑Dilithium (lattice‑based) – winner
    • Falcon (lattice‑based)
    • SPHINCS+ (hash‑based – no mathematical assumptions!)
```

---

### Lattice‑Based Crypto – Visual Intuition

```
Lattice = grid of points in space

Shortest Vector Problem (SVP):
Given: lots of points, find the shortest non‑zero vector

Visual:    •   •   •   •   •
           •   •   •   •   •
           •   •   •   •   •
           •   •   •   •   ?
                ↑ shortest?

No known quantum algorithm solves SVP efficiently.
```

> **Hard problem** even for quantum computers.

---

### Hash‑Based Signatures (SPHINCS+)

```
Based only on security of hash function (e.g., SHA‑256)

One‑time signature: Merkle tree of many keys

Stateful (few signatures) or stateless (many)
```

**Advantage:** No number theory – relies on symmetric crypto.  
**Disadvantage:** Large signatures (~8‑40 KB) vs. RSA (256 bytes).

---

## Industry Migration – Banks, Governments, You

### Banks and Financial Systems

```
Systems that use RSA/ECC today:
    • SWIFT (interbank transfers)
    • Credit card payments (Visa, Mastercard)
    • Online banking (TLS)
    • Stock exchanges

Migration plans:
    • Hybrid mode (classical + PQC) during transition
    • Expected completion: ~2035
    • Central banks testing PQC now
```

---

### Government & Military

```
NIST standardisation completed 2024
NSA announced timeline for National Security Systems:
    • Start migration by 2025
    • Complete by 2035

Applications:
    • Classified communications
    • Nuclear command and control
    • Satellite links
```

---

### Internet and You

```
Browsers (Chrome, Firefox) already testing:
    • X25519 + Kyber hybrid key exchange

Signal / WhatsApp will upgrade when libraries ready

Your future:
    • e‑passports will need PQC
    • Digital IDs (e‑residency) will migrate
    • Car key fobs, smart meters, IoT devices
```

> **Job opportunity:** Post‑quantum crypto engineers are in high demand.

---

##  Other Quantum Applications (Brief)

### Grover's Search (Next lecture)

```
Unstructured search: N items → √N quantum steps

Example: Find a password in N possibilities
         Classical: N tries
         Quantum: √N tries (AES‑256 becomes 2¹²⁸ secure)
```

---

### Quantum Simulation

```
Problem: Simulate a molecule (e.g., caffeine, protein)
Classical: exponential in number of electrons
Quantum: polynomial

Applications:
    • Drug design (COVID antivirals)
    • Battery materials (solid‑state batteries)
    • Fertilizer production (Haber process)
```

---

### Optimization (QAOA, VQE)

```
Combinatorial problems: 
    • Traveling salesman
    • Supply chain logistics
    • Portfolio optimisation

Quantum approximate speedup (not exponential, but useful)
```

---

## Summary 

| Topic | Key takeaway |
|-------|---------------|
| Non‑abelian HSP | No known quantum attack → possible crypto |
| Blockchain | Current ECC vulnerable; can migrate to PQC |
| NIST PQC | Kyber, Dilithium, SPHINCS+ – standards |
| Migration | Banks/governments: 2025‑2035 |
| Other QC | Simulation, optimisation, Grover |

---


```
Quantum computers WILL break RSA and ECC.

BUT:
    • We have post‑quantum crypto ready
    • Migration is a massive engineering challenge
    • Your career will span this transition

Learn the basics – you'll be ahead of 99% of developers.
```

---

## Next Lecture: Grover's Search Algorithm

- Unstructured search problem
- Quadratic speedup (√N vs N)
- Applications in databases, optimisation, AI

