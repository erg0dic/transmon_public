from src.basic_transmon import Transmon
import unittest
import numpy as np
from src.utilities import fid_test, vec2dm, dm2vec


class TestTransmon(unittest.TestCase):
    def setUp(self):
        self.t = Transmon(trl=2, n=2)

    def test_ham_hermitian(self):
        self.assertTrue(
            np.allclose(
                self.t.two_level_hamiltonian(),
                self.t.two_level_hamiltonian().T.conjugate(),
            ),
            "2 lvl Ham not physical!",
        )
        self.assertTrue(
            np.allclose(
                self.t.hamiltonian_full(), self.t.hamiltonian_full().T.conjugate()
            ),
            "2 lvl Ham not physical!",
        )

    def test_fid_state(self):
        y = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2) * 1j])
        assert np.allclose(self.t.state_fid(y, y), 1), "fid function error"

    def test_lindblad_correctness(self):
        rhos = self.t.evolve_lindbladian()
        for rho in rhos:
            assert np.allclose(
                rho.T.conjugate(), rho
            ), "lindblad evolution is not hermiticty preserving"

    def test_ode_solver_for_lindblad_evo(self):
        # test ode solver
        self.t.dissipate = False
        self.t.test = True
        rhos = self.t.evolve_lindbladian()
        # rhos = self.normalize_rhos(rhos) # for safety but unnecessary
        fids1 = list(map(self.t.rho_fid, rhos.reshape(-1, self.t.rank, self.t.rank)))
        mask = fids1 != np.nan  # because of nan instability will need to drop these

        fid_test(fids1[mask])
        self.t.test = False
        fids2, _ = self.t.unitary_evolve_2lvl()

        # an iffy test...
        assert (
            np.real(fids1[mask]) - np.real(fids2[mask])
        ).sum() <= 0.5, "fids non matching approximately"

    def test_bloch_vec_integration_1(self):
        r = np.zeros((4, 4))
        r[0][0] = 1
        self.assertTrue(np.allclose(vec2dm(dm2vec(np.array([r]))), r))


if __name__ == "__main__":
    unittest.main()
