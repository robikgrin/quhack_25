from graph_generator import *
from QUBO_transformers import *

from scipy.sparse.linalg import eigsh

from dimod import BinaryQuadraticModel
import neal

def probs2bit_str(probs: np.array) -> str:
    size = int(np.log2(probs.shape[0]))
    bit_s_num = int(np.argmax(probs))
    s = f"{bit_s_num:0{size}b}"
    return s

def basis_state_vector(bitstring: str) -> np.array:
    """Return computational basis column vector (dtype complex) for given bitstring.

    bitstring expects '0101' with length n; leftmost char is most-significant bit.
    """
    n = len(bitstring)
    dim = 2 ** n
    index = int(bitstring, 2)
    vec = np.zeros((dim,), dtype=np.complex64)
    vec[index] = 1.0
    return vec

if __name__ == "__main__":
    N = np.random.randint(5, 20)

    ### ВЫБИРАЕШЬ ГРАФ ###
    Qmat = generate_random_regular_graph_qubo(N, N-1, vis=True)
    Q = Qmat.toarray()
    ######################

    Q_dict = {}
    for i in range(N):
        start = Qmat.indptr[i]
        end = Qmat.indptr[i + 1]
        for idx in range(start, end):
            j = Qmat.indices[idx]
            val = Qmat.data[idx]
            if val != 0:
                Q_dict[(i, j)] =-val

    bqm = BinaryQuadraticModel.from_qubo(Q_dict)
    sampler = neal.SimulatedAnnealingSampler()

    energies = []
    solutions = []

    for i in range(10):
        sampleset = sampler.sample(bqm, num_reads=100)
        best = next(sampleset.data(['sample', 'energy']))
        solutions.append(best.sample)
        energies.append(best.energy)

    min_idx = energies.index(min(energies))
    best_solution = solutions[min_idx]

    ### РЕШЕНИЕ ОТЖИГОМ ###
    solution_x = np.array([best.sample[i] for i in range(N)], dtype=int)
    cut_value = float(solution_x @ Qmat @ solution_x)
    print("Classical cut value for solution:", cut_value)  # -> 5.0
    ######################

    ### СТРОИМ ГАМИЛТЬТОНИАН ИЗИНГА ###
    H = qubo_to_ising(-Q)

    print("Building Hamiltonian of shape:", H.shape)
    eigenvals, eigenvecs = eigsh(H, k=1, which='SR')
    energy = eigenvals[0]
    gs = eigenvecs[:, 0]
    probs = np.abs(gs) ** 2
    bitstr = probs2bit_str(probs)

    print("Quantum ground energy:", energy)
    print("Ground-state bitstring (most probable basis state):", bitstr)

    # compute expectation of H on the classical solution state
    bs = ''.join(str(b) for b in solution_x)
    v = basis_state_vector(bs)
    Ev = float(np.vdot(v.T, H.dot(v)).real)
    print(f"Energy of classical solution {bs} :", Ev)

    if energy <= Ev + 1e-8:
        print("OK: ground energy <= classical solution energy (as expected).")
    else:
        print("WARNING: ground energy > classical solution energy -- check transformation.")

    found_x = np.array([int(c) for c in bitstr], dtype=int)
    found_cut = float(found_x @ Qmat @ found_x)
    print("Cut value for found bitstring:", found_cut)


    if found_cut >= cut_value - 1e-8:
        print("Found cut is >= provided solution cut (OK).")
    else:
        print("Found cut is worse than provided solution (could still be due to degeneracy).")
