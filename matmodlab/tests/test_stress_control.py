import os
import numpy as np
from testconf import *
from matmodlab.utils.fileio import savefile
try:
    import matmodlab.lib.elastic as el
except ImportError:
    el = None


@pytest.mark.material
@pytest.mark.elastic
class TestStressControl(StandardMatmodlabTest):

    @pytest.mark.stresscontrol
    @pytest.mark.skipif(el is None, reason='elastic model not imported')
    @pytest.mark.parametrize('realization', range(0, 10))
    def test_stress_control(self, realization):

        runid = "stress_control_{0:04d}".format(realization)

        failtol = 5.0e-8
        Nsteps = np.random.randint(20, 101)  # upper limit exclusive
        print("===== Setup:")
        print("runid:  {0:s}".format(runid))
        print("failtol:{0:10.2e}".format(failtol))
        print("Nsteps: {0:d}".format(Nsteps))

        head = ('Time',
                'E.XX', 'E.YY', 'E.ZZ', 'E.XY', 'E.YZ', 'E.XZ',
                'S.XX', 'S.YY', 'S.ZZ', 'S.XY', 'S.YZ', 'S.XZ')
        data = np.zeros((Nsteps, len(head)))

        #
        # Parameters
        #
        E = 10.0 ** np.random.uniform(0.0, 12.0)
        NU = np.random.uniform(-0.95, 0.45)
        K = E / 3.0 / (1.0 - 2.0 * NU)
        G = E / 2.0 / (1.0 + NU)
        params = {"K": K, "G": G}
        print("===== Parameters:")
        print("E: {0:25.14e}".format(E))
        print("NU:{0:25.14e}".format(NU))
        print("K: {0:25.14e}".format(K))
        print("G: {0:25.14e}".format(G))

        #
        # Generate the path and analytical solution
        #
        eps = np.array([2. * (np.random.rand() - 0.5) * np.random.randint(0, 2)
                        for _ in range(6)])
        eps_iso = (np.sum(eps[:3]) / 3.0) * np.array([1., 1., 1., 0., 0., 0.])
        eps_dev = eps - eps_iso
        sig = 3.0 * K * eps_iso + 2.0 * G * eps_dev

        for idx, t in enumerate(np.linspace(0.0, 1.0, Nsteps)):
            curr_eps = t * eps
            curr_sig = t * sig
            data[idx, 0] = t
            data[idx, 1:7] = curr_eps
            data[idx, 7:] = curr_sig

        gold_f = runid + "_gold.txt.gz"
        savefile(gold_f, head, data)

        #
        # Run the strain-controlled version
        #
        mps_eps = MaterialPointSimulator(runid + "_eps", verbosity=0,
                                         d=this_directory,
                                         output_format="txt.gz")
        mps_eps.Material("elastic", params)

        mps_eps.DataSteps(gold_f, tc=0, columns=head[1:7], frames=1,
                          descriptors='EEEEEE')
        mps_eps.finish()
        assert mps_eps.ran

        #
        # Run the stress-controlled version
        #
        mps_sig = MaterialPointSimulator(runid + "_sig", verbosity=0,
                                         d=this_directory,
                                         output_format="txt.gz")
        mps_sig.Material("elastic", params)

        mps_sig.DataSteps(gold_f, tc=0, columns=head[7:], frames=1,
                          descriptors='SSSSSS')
        mps_sig.finish()
        assert mps_sig.ran

        #
        # Analysis
        #
        data_eps = mps_eps.get(*head, disp=-1)
        data_sig = mps_sig.get(*head, disp=-1)
        assert data.shape == data_eps.shape and data.shape == data_sig.shape


        F = open("time_errors.txt", 'a')
        has_passed = True
        print("{0:>10s}:{1:>25s}{2:>25s}".format("key", "err_eps", "err_sig"))
        dt = data[1, 0] - data[0, 0]
        for idx, key in enumerate(head):
            gold = data[:, idx]
            eps = data_eps[:, idx]
            sig = data_sig[:, idx]


            basis = max(1.0, np.trapz(np.abs(gold), dx=dt))
            err_eps = np.trapz(np.abs(gold-eps), dx=dt) / basis
            err_sig = np.trapz(np.abs(gold-sig), dx=dt) / basis
            if max(err_eps, err_sig) > failtol:
                has_passed = False

            if key == "Time":
                val = 0
            elif key.startswith("E."):
                val = 1
            else:
                val = 2
            F.write("{2:10d}{0:25.14e}{1:10d}\n".format(max(err_eps, err_sig), val, realization))

            print("{0:>8s}:{1:25.14e}{2:25.14e}".format(key, err_eps, err_sig))

        F.close()

        assert has_passed

        #
        # Passed! Clean up.
        #
        os.remove(gold_f)
        os.remove(mps_eps.filename)
        os.remove(mps_sig.filename)
