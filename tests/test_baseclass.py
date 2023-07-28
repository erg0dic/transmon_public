from src.baseclass import (
    QSys,
    expm,
    two_qubit_rotation_3_param_family,
    two_qubit_rotation_single_param_family,
)
import unittest
import numpy as np
from src.utilities import (
    random_density_matrix,
    dm2vec,
    vec2dm,
    get_pauli_basis_matrices,
)


class TestPaulis(unittest.TestCase):
    def setUp(self):
        self.qsys = QSys()

    def test_double_pauli_multiplication(self):
        sigma_1 = "XYZY1"
        sigma_2 = "ZY1XX"
        self.assertTrue(
            self.qsys.multiply_general_paulis(sigma_1, sigma_2) == "_-1_Y1ZZX"
        )
        sigma_1 = "_-1j_XYZY1"
        sigma_2 = "_-1j_ZY1XX"
        self.assertTrue(self.qsys.multiply_general_paulis(sigma_1, sigma_2) == "Y1ZZX")
        sigma_1 = "_-1j_XYZY1"
        sigma_2 = "ZY1XX"
        self.assertTrue(
            self.qsys.multiply_general_paulis(sigma_1, sigma_2) == "_1j_Y1ZZX"
        )
        sigma_1 = "XYZY1"
        sigma_2 = "_-1j_ZY1XX"
        self.assertTrue(
            self.qsys.multiply_general_paulis(sigma_1, sigma_2) == "_1j_Y1ZZX"
        )
        sigma_1 = "XYZY1"
        sigma_2 = "_-1_ZY1XX"
        self.assertTrue(self.qsys.multiply_general_paulis(sigma_1, sigma_2) == "Y1ZZX")

    def test_multiple_pauli_multiplication(self):
        sigma_1 = "XYZY1"
        sigma_2 = "ZY1XX"
        pauli_list = [sigma_1, sigma_1, sigma_2]
        self.assertTrue(self.qsys.multiply_many_general_paulis(pauli_list) == sigma_2)
        pauli_list = [sigma_1, sigma_1, sigma_2, sigma_2]
        self.assertTrue(self.qsys.multiply_many_general_paulis(pauli_list) == "11111")
        pauli_list = [sigma_1, sigma_1, sigma_2, sigma_2, sigma_1]
        self.assertTrue(self.qsys.multiply_many_general_paulis(pauli_list) == sigma_1)
        # slightly more non-trivial
        sigma_1 = "_-1j_" + sigma_1
        pauli_list = [sigma_1, sigma_1, sigma_2, sigma_2, sigma_1]
        self.assertTrue(
            self.qsys.multiply_many_general_paulis(pauli_list) == "_1j_XYZY1"
        )

        # big one
        pauli_list = [sigma_1, sigma_1, sigma_2, sigma_2, sigma_1, sigma_1] * 100
        self.assertTrue(self.qsys.multiply_many_general_paulis(pauli_list) == "11111")

        # check if base cases work
        sigma_1 = "XYZY1"
        sigma_2 = "ZY1XX"
        self.assertTrue(
            self.qsys.multiply_many_general_paulis([sigma_1, sigma_2]) == "_-1_Y1ZZX"
        )
        self.assertTrue(self.qsys.multiply_many_general_paulis(sigma_1) == sigma_1)

    def test_pauli_basis_generators(self):
        paulis_2qubits = list(self.qsys.basis_generator(2))
        self.assertTrue(len(paulis_2qubits) == pow(2, 4))
        A = set()
        for s1 in ["1", "X", "Y", "Z"]:
            for s2 in ["1", "X", "Y", "Z"]:
                A.add(s1 + s2)
        self.assertTrue(
            len(list(A.intersection(set(paulis_2qubits)))) == len(paulis_2qubits)
        )
        self.assertEqual(list(self.qsys.basis_generator(1)), ["1", "X", "Y", "Z"])
        paulis_5qubits = list(self.qsys.basis_generator(5))
        self.assertTrue(len(paulis_5qubits) == pow(2, 10))

    def test_pauli_commutator(self):

        self.assertTrue(
            np.allclose(
                self.qsys.pauli_commutator("X", "Y"), 2j * self.qsys.get_pauli["Z"]
            )
        )
        self.assertTrue(
            np.allclose(
                self.qsys.pauli_commutator("Y", "Z"), 2j * self.qsys.get_pauli["X"]
            )
        )

        self.assertTrue(
            np.allclose(
                self.qsys.pauli_commutator("Z", "X"), 2j * self.qsys.get_pauli["Y"]
            )
        )

    # @unittest.skip("a bit slow")
    def test_integration_1(self):
        XX = self.qsys.eval_pauli_string("XX")
        YY = self.qsys.eval_pauli_string("YY")
        xx_yy = self.qsys.commutator(XX, YY)
        self.assertTrue(np.allclose(self.qsys.pauli_commutator("XX", "YY"), xx_yy))

        randis = np.random.randint(low=0, high=4, size=(2, 10, 10))  # lol

        for i in range(len(randis[0])):
            p1 = "".join(self.qsys.paulis[j] for j in randis[0][i])
            p2 = "".join(self.qsys.paulis[j] for j in randis[1][i])
            XX = self.qsys.eval_pauli_string(p1)
            YY = self.qsys.eval_pauli_string(p2)
            xx_yy = self.qsys.commutator(XX, YY)
            self.assertTrue(np.allclose(self.qsys.pauli_commutator(p1, p2), xx_yy))
            xx_p_yy = self.qsys.anticommutator(XX, YY)
            self.assertTrue(
                np.allclose(self.qsys.pauli_anticommutator(p1, p2), xx_p_yy)
            )

    def test_density_mat_generation_single(self):
        for way in ["Bures", "HS"]:
            for dim in range(2, 17):
                mat1 = random_density_matrix(dim, way=way)
                mat2 = random_density_matrix(dim, way=way)
                self.assertTrue(np.allclose(mat1.conj().T - mat1, 0))
                self.assertTrue(np.allclose(mat2.conj().T - mat2, 0))
                self.assertTrue(np.allclose(mat1.trace(), 1))
                self.assertTrue(np.allclose(mat2.trace(), 1))

    def test_density_mat_generation_batch(self):
        for way in ["Bures", "HS"]:
            for dim in range(2, 17):
                mat1 = random_density_matrix(dim, way=way, size=100)
                mat2 = random_density_matrix(dim, way=way, size=100)
                self.assertTrue(
                    np.allclose(mat1.conj().transpose(-3, -1, -2) - mat1, 0)
                )
                self.assertTrue(
                    np.allclose(mat2.conj().transpose(-3, -1, -2) - mat2, 0)
                )
                self.assertTrue(np.allclose(np.einsum("...ii", mat1) - 1, 0))
                self.assertTrue(np.allclose(np.einsum("...ii", mat2) - 1, 0))

    def test_vec2dm2vec_batch(self):
        for size in [1, 100]:
            for dim in [2, 4, 8]:
                rhos = random_density_matrix(dim, size=size)
                pauli_exps = dm2vec(rhos)
                np.allclose(vec2dm(pauli_exps) - rhos, 0)

    def test_two_qubit_3_param_family(self):
        X = QSys.get_pauli["X"]
        Y = QSys.get_pauli["Y"]
        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        expected = expm(-1j * XX)
        expected2 = expm(-1j * (XX + YY))
        out = two_qubit_rotation_3_param_family([1, 0, 0])
        out2 = two_qubit_rotation_3_param_family([1, 1, 0])
        # single test
        self.assertTrue(np.allclose(expected - out, 0))
        self.assertTrue(np.allclose(expected2 - out2, 0))
        # double/batch test
        expected3 = np.array([expected, expected2])
        out3 = two_qubit_rotation_3_param_family([[1, 0, 0], [1, 1, 0]])
        self.assertTrue(np.allclose(expected3 - out3, 0))

    def test_correct_su_dn_recovery(self):
        # up to 2 qubits
        for n in range(1, 3):
            for trl in [2, 3, 5]:
                rhos = random_density_matrix(int(pow(trl, n)), size=100)
                paulis = get_pauli_basis_matrices(n, trl=trl)
                trhos = vec2dm(dm2vec(rhos, paulis=paulis), paulis)
                self.assertTrue(np.allclose(rhos, trhos))


if __name__ == "__main__":
    unittest.main()
