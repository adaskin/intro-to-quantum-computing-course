import numpy as np
import pennylane as qml
from math import gcd
import matplotlib.pyplot as plt

# ---------- Helper: Continued Fractions ----------
def continued_fraction(x, y, max_denom):
    """
    Return a list of convergents (num, den) for the fraction x/y,
    with denominator <= max_denom.
    """
    a, b = int(x), int(y)          # ensure integers
    cf = []
    while b != 0:
        cf.append(a // b)
        a, b = b, a % b
    convs = []
    for i in range(len(cf)):
        if i == 0:
            num, den = cf[0], 1
        elif i == 1:
            num, den = cf[0]*cf[1] + 1, cf[1]
        else:
            num = cf[i]*convs[-1][0] + convs[-2][0]
            den = cf[i]*convs[-1][1] + convs[-2][1]
        if den <= max_denom:
            convs.append((num, den))
    return convs

# ---------- Periodic State Preparation (Reused) ----------
def periodic_state(r, Q, x0=0):
    """
    Create uniform superposition of |x0 + m·r mod Q⟩ for m=0..M-1,
    where M = Q / gcd(Q, r).
    Q is the Hilbert space dimension (power of 2).
    """
    state = np.zeros(Q, dtype=complex)
    x0 = x0 % r
    m = 0
    while True:
        idx = (x0 + m * r) % Q
        if state[idx] != 0:
            break
        state[idx] = 1.0
        m += 1
    state /= np.linalg.norm(state)
    return state

# ---------- QFT Circuit ----------
def qft_circuit(wires):
    n = len(wires)
    for i in range(n):
        qml.Hadamard(wires=wires[i])
        for j in range(i + 1, n):
            angle = np.pi / (2 ** (j - i))
            qml.ControlledPhaseShift(angle, wires=[wires[j], wires[i]])
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - i - 1]])

# ---------- Shor Period Finding Simulation ----------
def shor_period_finding(N=15, a=7, input_qubits=None, shots=1):
    """
    Simulate the period‑finding step for f(x) = a^x mod N.
    - N: number to factor
    - a: random coprime base
    - input_qubits: number of qubits for the input register (default: 2*ceil(log2(N)) )
    - shots: number of measurement repetitions (more shots improve success probability)
    """
    # 1. Determine the order r classically (for simulation only)
    r = None
    for cand in range(1, N):
        if pow(a, cand, N) == 1:
            r = cand
            break
    if r is None:
        raise ValueError(f"No order found for a={a} mod {N}")

    # 2. Set size of input register: Q = 2^m > N^2
    if input_qubits is None:
        input_qubits = int(np.ceil(2 * np.log2(N)))
    Q = 2**input_qubits

    print("=" * 60)
    print(f"Target: N = {N}, a = {a}  → order r = {r}")
    print(f"Input register: {input_qubits} qubits → Q = {Q}")
    print("=" * 60)

    # 3. Simulate the measurement of the output register.
    x0 = np.random.randint(0, r)
    print(f"Simulated output measurement gives offset x0 = {x0}")

    target_state = periodic_state(r, Q, x0)

    # 4. Apply QFT to the input register
    dev = qml.device('default.qubit', wires=input_qubits)

    @qml.qnode(dev)
    def apply_qft(state_vec):
        qml.StatePrep(state_vec, wires=range(input_qubits))
        qft_circuit(wires=range(input_qubits))
        return qml.probs(wires=range(input_qubits))

    probs = apply_qft(target_state)

    # 5. Sample from the QFT output
    measured_values = np.random.choice(Q, size=shots, p=probs)
    print(f"Measured input register value(s): {measured_values}")

    # 6. Attempt to recover r from each measurement via continued fractions
    candidates_found = []
    for c in measured_values:
        print(f"\nProcessing c = {c}:")
        frac = c / Q
        print(f"  Fraction c/Q = {c}/{Q} ≈ {frac:.4f}")
        convs = continued_fraction(c, Q, N-1)   # denominators up to N-1
        for num, den in convs:
            den_int = int(den)   
            print(f"    Convergent: {num}/{den_int}  → candidate r = {den_int}")
            if pow(a, den_int, N) == 1:
                print(f"      ✅ Valid period! r = {den_int}")
                candidates_found.append(den_int)
                break   # take the first valid candidate for this measurement
        else:
            print("      ❌ No valid period from this measurement.")

    # 7. Final result
    if candidates_found:
        final_r = candidates_found[0]   # in practice, use LCM of multiple candidates
        print(f"\nSuccess: Found period r = {final_r}")
        if final_r == r:
            print("  (Matches the true order.)")
        # Check if we can factor N
        if final_r % 2 == 0:
            x = pow(a, final_r//2, N)
            if x != N-1:
                f1 = gcd(x + 1, N)
                f2 = gcd(x - 1, N)
                print(f"  Factors of {N}: {f1} and {f2}")
            else:
                print("  a^(r/2) ≡ -1 mod N, try another a.")
        else:
            print("  r is odd, cannot factor with this a.")
    else:
        print("\nNo period recovered. Try increasing input_qubits or shots.")

    # Optional: plot the QFT probability distribution
    plt.figure(figsize=(8,3))
    plt.stem(range(Q), probs)
    plt.title(f'QFT output for N={N}, a={a}, r={r}')
    plt.xlabel('Measured value c')
    plt.ylabel('Probability')
    plt.show()

    return final_r if candidates_found else None

# ---------- Run the Demo ----------
np.random.seed(42)   # for reproducibility
shor_period_finding(N=15, a=7, input_qubits=8, shots=3)