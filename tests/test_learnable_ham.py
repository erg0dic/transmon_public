import unittest
from src.ppo import PPO, CNOT
from src.utilities import super_op, get_rand_Us, dm2vec, ri_split, super_pre, super_post, vec2dm
from src.baseopt import QSys, Transmon, BaseOpt
from scipy.stats import unitary_group
from play.schro_torchsolver import get_init_states
from src.internal_ham_learning_model import LearnableHamiltonian
import numpy as np
import torch
from scipy.linalg import expm
from torch.optim import Adam

def test_sgd_of_learnable_ham(batch_size=10, heun_eps=1e-2, opt_lr=1e-3):
    "just a simple test to see if sgd works with the architecture => regress onto the inputs"

    learner = LearnableHamiltonian(4, qubits=2, num_timesteps=20, debug=False, 
                                                heun_eps=heun_eps, ansatz_A=True)
    optimizer = Adam(learner.parameters(), lr=opt_lr)
    randUs, randUts = get_rand_Us(batch_size, 2)
    batch_acts = (np.random.normal(size=(batch_size,2))*20-10)%10


    for t in range(10000):
        out_Us, _ = learner.predict_prop(torch.as_tensor(batch_acts, dtype=torch.float32),randUs)
        loss = ((out_Us-randUs)**2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t%50 == 0:
            print(f"loss {loss} step {t}" )
# test_sgd_of_learnable_ham(batch_size=100, heun_eps=1e-2, opt_lr=1e-2)
class TestUnitaryHamLearner(unittest.TestCase):

    learner = LearnableHamiltonian(1, qubits=2, num_timesteps=20, debug=True, 
                                                heun_eps=1e-4, dont_batch_inputs=True)
    hetero_learner = LearnableHamiltonian(4, qubits=2, num_timesteps=20, debug=False, 
                                                    heun_eps=1e-4, ansatz_A=True)

    tester = PPO(num_timesteps=20, target=CNOT)

    def test_single_propagation_step_unitary(self):
        self.tester.reset()
        test_starting_state,_ = get_init_states(int(pow(2,self.learner.qubits)))
        for _ in range(5):
            sample_action = (np.random.normal(size=2)*20-10)%10
            out_state, rew = self.tester.take_step(sample_action)
            # get complex unitary
            out_state = (out_state[:len(out_state)//2]+1j*out_state[len(out_state)//2:])
            out_state = out_state.reshape(int(np.sqrt(len(out_state))), -1)

            # check if the learner can get the same single step propagation
            out_target_state, rew_t = self.learner.predict_prop(torch.as_tensor(sample_action, dtype=torch.float32), 
                                                torch.as_tensor(test_starting_state, dtype=torch.float32))
            out_target_state, rew_t = out_target_state.detach().numpy(), rew_t.detach().numpy()
            out_target_state_c = out_target_state[:len(out_target_state)//2] + 1j*out_target_state[len(out_target_state)//2:]
            self.assertTrue(np.allclose(out_state,out_target_state_c,atol=1e-3))
            self.assertTrue(np.allclose(rew,rew_t,atol=5e-3))
            test_starting_state = out_target_state

    def test_uni_to_super_to_choi_conversion_for_multiple_qubits(self):
        for q in range(1,5):
            for _ in range(100):
                rand_U = unitary_group.rvs(int(pow(2,self.learner.qubits)))
                choi = QSys.super_to_choi(super_op(rand_U))
                assert np.allclose(choi.T.conj(),  choi, atol=1e-4), "choi conversion unsuccessful"

    def test_super_op_choi_to_bloch_torch(self):
        for _ in range(10):
            rand_U = unitary_group.rvs(int(pow(2,self.learner.qubits)))
            super_target = super_op(rand_U)
            rand_U_torch = torch.as_tensor(np.array([np.real(rand_U), np.imag(rand_U)]).reshape(2*rand_U.shape[-1], -1), dtype=torch.float32)
            super_op_torch_real, super_op_torch_imag = self.learner.super_op(rand_U_torch)
            super_op_torch_real, super_op_torch_imag = super_op_torch_real.detach().numpy(), super_op_torch_imag.detach().numpy()
            self.assertTrue(np.allclose(super_target, (super_op_torch_real + 1j*super_op_torch_imag), atol=1e-4))
            chR, chI = self.learner.convert_to_choi(rand_U_torch)
            chR, chI = chR.detach().numpy(), chI.detach().numpy()
            target_choi = QSys.super_to_choi(super_target)
            self.assertTrue(np.allclose(target_choi, chR + 1j*chI, atol=1e-4))
            bvec = self.learner.convert_to_bloch(rand_U_torch).detach().numpy()
            target_bvec = dm2vec(target_choi, self.learner.paulis_2n)
            self.assertTrue(np.allclose(target_bvec, bvec, atol=1e-4))
    
    def test_batch_fidelity_getter(self):
        rand_Us, rand_Us_t = get_rand_Us(100, self.learner.qubits)
        batch_fids = self.learner.unitary_fid(rand_Us, self.learner.target, self.learner.trl, self.learner.qubits).detach().numpy()
        batch_fids_target = np.array([self.tester.cost_function(rand_U, CNOT) for rand_U in rand_Us_t])
        self.assertTrue(np.allclose(batch_fids, batch_fids_target, atol=1e-4))

    def test_super_op_choi_to_bloch_torch_batch(self):
        rand_Us, rand_Us_t = get_rand_Us(100, self.learner.qubits)
        Us2choi = lambda rands: map(QSys.super_to_choi, map(super_op, rands))
        target_chois = np.array(list(Us2choi(rand_Us_t)))
        chRs, chIs = self.learner.convert_to_choi(rand_Us)
        self.assertTrue(np.allclose(target_chois, (chRs + 1j*chIs).detach().numpy(), atol=1e-4))
        target_bvecs = np.array(list(map(lambda x: dm2vec(x, self.learner.paulis_2n), Us2choi(rand_Us_t))))
        bvecs = self.learner.convert_to_bloch(rand_Us).detach().numpy()
        self.assertTrue(np.allclose(target_bvecs, bvecs, atol=1e-4))


    def test_homogeneous_batch_ensemble_single_propagation_step_unitary(self):
        "in/oout states are just tiles so everything is the same as the single propagation stage. Simple sanity check"
        learner = LearnableHamiltonian(5, qubits=2, num_timesteps=20, debug=True, 
                                                    heun_eps=1e-4)
        self.tester.reset()
        test_starting_state,_ = get_init_states(int(pow(2,self.learner.qubits)))
        test_starting_state = test_starting_state.numpy()
        for _ in range(5):
            sample_action = (np.random.normal(size=2)*20-10)%10
            out_state, rew = self.tester.take_step(sample_action)
            # get complex unitary
            out_state = (out_state[:len(out_state)//2]+1j*out_state[len(out_state)//2:])
            out_state = out_state.reshape(int(np.sqrt(len(out_state))), -1)

            # check if the learner can get the same single step propagation
            sample_action = np.tile(sample_action, reps=20).reshape(20, -1)
            
            test_starting_state = np.tile(test_starting_state.ravel(), reps=20).reshape(20, *test_starting_state.shape)
            out_target_state, rew_t = learner.predict_prop(torch.as_tensor(sample_action, dtype=torch.float32), 
                                                torch.as_tensor(test_starting_state, dtype=torch.float32))
            
            out_target_state, rew_t = out_target_state.detach().numpy(), rew_t.detach().numpy()
            
            out_target_state_c = out_target_state[:,:,:out_target_state.shape[-2]//2,:] + 1j*out_target_state[:,:,out_target_state.shape[-2]//2:,:]
            self.assertTrue(np.allclose(out_state,out_target_state_c.sum(axis=0).sum(axis=0)/(5*20),atol=1e-3))
            self.assertTrue(np.allclose(rew,rew_t.sum()/(5*20),atol=5e-3))
            test_starting_state = out_target_state[0][0]

    def get_hetero_learner(self):
        return self.hetero_learner

    def test_semiheterogeneous_batch_ensemble_single_propagation_step_unitary(self):
        "semi because all learning Hamiltonians get the same data to train on"
        learner = self.get_hetero_learner()
        hams = np.array(list(map(lambda x: x.transmon_sys_ham, learner.syses)))
        
        tbatch_init_states, batch_init_states = get_rand_Us(100, learner.qubits)
        batch_acts = (np.random.normal(size=(100,2))*20-10)%10
        
        unitary_ensemble_prop_batch, bloch_ensemble_prop_batch, rews = self.get_training_data(learner, batch_init_states, batch_acts)


        self.run_batch_tests(learner, tbatch_init_states, batch_acts, unitary_ensemble_prop_batch, bloch_ensemble_prop_batch, rews)

    def test_lindblad_diss_correctness(self):
        learner = self.get_hetero_learner()
        t = Transmon(trl=2)
        diss_params = np.random.uniform(size=(4,16))
        col_ops = vec2dm(diss_params)
        diss_params = torch.as_tensor(np.real(col_ops), dtype=torch.float32), torch.as_tensor(np.imag(col_ops), dtype=torch.float32)
        t_lind_term = t.get_static_lind_term( col_ops)
        t_lind_term = np.concatenate((np.real(t_lind_term), np.imag(t_lind_term)))
        my_lind_term = learner.get_learnt_accumulant(diss_params)
        self.assertTrue(np.allclose(t_lind_term, my_lind_term.detach().numpy(), atol=1e-6))


    def test_heterogeneous_batch_ensemble_single_propagation_step_unitary(self):
        "all learning Hamiltonians different data to train on"
        learner, batch_acts = self.get_init_learner_and_acts()
        tbatch_init_states, batch_init_states = get_rand_Us(100*4, learner.qubits)
        tbatch_init_states = tbatch_init_states.reshape(4, 100, -1, int(pow(2,learner.qubits)))
        batch_init_states = batch_init_states.reshape(4, 100, -1, int(pow(2,learner.qubits)))
        unitary_ensemble_prop_batch, bloch_ensemble_prop_batch, rews = self.get_training_data(learner, batch_init_states, batch_acts)
        self.run_batch_tests(learner, tbatch_init_states, batch_acts, unitary_ensemble_prop_batch, bloch_ensemble_prop_batch, rews)

    def test_choi_to_bloch_to_choi_batch(self):
        # test various correct conversion mapping for the batch ode solver
        learner = self.get_hetero_learner()
        out_target_states, _ = get_rand_Us(10, 2)
        supers_target = torch.cat(learner.super_op(out_target_states), dim=-2)
        chois_ri = learner.convert_to_choi(out_target_states)
        supers = torch.cat(learner.convert_to_choi(chois_ri, dont_convert_to_super=True), dim=-2)
        out_chois = torch.cat(chois_ri, dim=-2)
        out_blochs_target = learner.convert_to_bloch(out_target_states)
        out_chois_target = torch.cat(learner.bloch_to_choi(out_blochs_target), dim=-2)
        self.assertTrue(np.allclose(out_chois, out_chois_target, atol=1e-4))
        self.assertTrue(np.allclose(supers, supers_target, atol=1e-4))

    def test_heterogeneous_batch_ensemble_single_propagation_step_lind(self):
        learner, batch_acts = self.get_init_learner_and_acts(batch_size=10)
        tbatch_init_states, batch_init_states = get_rand_Us(10*4, learner.qubits)
        tbatch_init_states = tbatch_init_states.reshape(4, 10, -1, int(pow(2,learner.qubits)))
        batch_init_states = batch_init_states.reshape(4, 10, -1, int(pow(2,learner.qubits)))
        unitary_ensemble_prop_batch, _, _ = self.get_training_data(learner, batch_init_states, batch_acts, batch_size=10)
        tbatch_init_states = torch.cat(learner.super_op(tbatch_init_states), dim=-2)
        out_target_states, _ = learner.predict_prop(torch.as_tensor(batch_acts, dtype=torch.float32), tbatch_init_states, lind=True)  
        unitary_ensemble_prop_batch = np.concatenate([np.real(unitary_ensemble_prop_batch), np.imag(unitary_ensemble_prop_batch)], axis=-2)
        unitary_ensemble_prop_batch = torch.as_tensor(unitary_ensemble_prop_batch, dtype=torch.float32)
        unitary_ensemble_prop_batch = torch.cat(learner.super_op(unitary_ensemble_prop_batch), dim=-2)
        out_target_states, unitary_ensemble_prop_batch = out_target_states.detach().numpy(), unitary_ensemble_prop_batch.detach().numpy()
        self.assertTrue(np.allclose(out_target_states, unitary_ensemble_prop_batch, atol=1e-4))

    def test_identity_tiler(self):
        learner, batch_acts = self.get_init_learner_and_acts(full_traj=True)
        us = learner.get_init_Us(batch_size=batch_acts.shape[1])
        self.assertTrue(np.allclose(us.detach().numpy()-ri_split(np.eye(int(pow(2,learner.qubits)))),0))

    def test_heterogeneous_batch_ensemble_entire_trajectory_unitary(self):
        learner, batch_acts = self.get_init_learner_and_acts(full_traj=True, batch_size=2)
        U_t_i_s = learner.get_init_Us(batch_size=batch_acts.shape[1]).detach().numpy()
        U_t_i_s = U_t_i_s[:,:,:4, :] + 1j*U_t_i_s[:,:,4: ,:]
        final_Us = learner.predict_prop_entire_trajectory(torch.as_tensor(batch_acts, dtype=torch.float32)).detach().numpy()
        final_Us = final_Us[:,:,:4, :] + 1j*final_Us[:,:,4: ,:]
        for i_t in range(learner.num_timesteps):
            batch_acts_per_timestep = batch_acts[:,:,i_t,:]
            U_t_i_s, _, _ = self.get_training_data(learner, U_t_i_s, batch_acts_per_timestep, batch_size=2)
        self.assertTrue(np.allclose(U_t_i_s-final_Us, 0, atol=1e-3))

    def get_init_learner_and_acts(self, full_traj=False, batch_size=100, num_timesteps=20):

        learner = self.get_hetero_learner()
        if not full_traj:
            batch_acts = (np.random.normal(size=(4,batch_size,2))*20-10)%10
        else:
            batch_acts = (np.random.normal(size=(4,batch_size,num_timesteps,2))*20-10)%10
        return learner, batch_acts

    def get_training_data(self, learner: LearnableHamiltonian, batch_init_states, 
                          batch_acts, batch_size=100):
        
        unitary_ensemble_prop_batch = np.zeros((4,batch_size,batch_init_states.shape[-1], batch_init_states.shape[-1]), dtype=np.complex128)
        bloch_ensemble_prop_batch = np.zeros((4,batch_size,int(pow(2,learner.qubits*2*2))))
        rews = np.zeros((4,batch_size))
        final_time = 1.0
    
        def evolve_tester(sys, action, init_U):
            return expm(-1j*sys.hamiltonian_full(action)*final_time)@init_U
        for i, sys in enumerate(learner.syses):
            if len(batch_acts.shape)==3:
                acts, batch_inits = batch_acts[i], batch_init_states[i]
            else: 
                acts, batch_inits = batch_acts, batch_init_states
            unitary_ensemble_prop_batch[i] = np.array(list(map(lambda U, act: evolve_tester(sys, action=act,init_U=U), 
                                                        batch_inits, acts)))
            rews[i] = np.array(list(self.tester.cost_function(evo_U, CNOT) for evo_U in unitary_ensemble_prop_batch[i]))
            bloch_ensemble_prop_batch[i] = np.array(list(dm2vec(self.tester.qsys.super_to_choi(super_op(evo_U)), learner.paulis_2n) for evo_U in unitary_ensemble_prop_batch[i]))
        
        return unitary_ensemble_prop_batch, bloch_ensemble_prop_batch, rews

    
    def run_batch_tests(self, learner: LearnableHamiltonian, tbatch_init_states, batch_acts, 
                        unitary_ensemble_prop_batch, bloch_ensemble_prop_batch, rews):
        out_target_states, rew_ts = learner.predict_prop(torch.as_tensor(batch_acts, dtype=torch.float32), tbatch_init_states)  
        out_blochs_targets = learner.convert_to_bloch(out_target_states).detach().numpy()
        
        out_target_states, rew_ts = out_target_states.detach().numpy(), rew_ts.detach().numpy()
        out_target_states = out_target_states[:,:,:4, :] + 1j*out_target_states[:,:,4: ,:]
        # test correct predicted fidelity
        self.assertTrue(np.allclose(rew_ts, rews, atol=1e-5))
        # test correct propagation step
        self.assertTrue(np.allclose(out_target_states, unitary_ensemble_prop_batch, atol=1e-4))
        # test correct bloch form conversion
        self.assertTrue(np.allclose(out_blochs_targets, bloch_ensemble_prop_batch, atol=1e-4))

if __name__ == '__main__':
    unittest.main()