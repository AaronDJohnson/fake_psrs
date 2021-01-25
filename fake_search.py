import os, glob, json, pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
import itertools

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import enterprise_extensions
from enterprise_extensions import models, sampler, model_utils
from multiprocessing import Pool

import libstempo as libs
import libstempo.plot as LP, libstempo.toasim as LT



def make_fakes(amp, psrcut):
    datadir = './data/partim/'
    ## Get parameter noise dictionary
    noisedir = os.getcwd() + '/data/noisefiles/'

    noise_params = {}
    for filepath in glob.iglob(noisedir + '/*.json'):
        with open(filepath, 'r') as f:
            noise_params.update(json.load(f))
    # print(noise_params)

    # import par and tim files into parfiles and timfiles
    tempopsr = []
    print('Importing to libstempo....')
    for psr in psrcut:
        parfiles = sorted(glob.glob(datadir + psr.name + '*.par'))
        timfiles = sorted(glob.glob(datadir + psr.name + '*.tim'))
        psr = libs.tempopulsar(parfile = parfiles[0],
                               timfile = timfiles[0], maxobs=50000)
        tempopsr.append(psr)
    print('adding efac')
    for psr in tempopsr:
        LT.make_ideal(psr)
        LT.add_efac(psr, efac=1.0)
    print('adding red noise')
    for psr in tempopsr:
        # print(psr.name)
        rednoise_toadd = noise_params[str(psr.name) + '_log10_A']
        # print(psr.name)
        gamma = noise_params[str(psr.name) + '_gamma']
        LT.add_rednoise(psr, 10**rednoise_toadd, gamma, components=30)
        psr.fit()
    print('adding GWB')
    LT.createGWB(tempopsr, Amp=amp, gam=13./3.)
    for psr in tempopsr:
        psr.fit()
    # save the pulsars
    pardir = './fakes_gwb/par/'
    timdir = './fakes_gwb/tim/'
    if not os.path.exists(pardir):
        os.makedirs(pardir)
        os.makedirs(timdir)

    print('fixing tim files')
    for psr in tempopsr:
        psr.savepar(pardir + psr.name + '-sim.par')
        psr.savetim(timdir + psr.name + '-sim.tim')
        libs.purgetim(timdir + psr.name + '-sim.tim')  # fix the tim files


def load_fakes():
    # import the par and tim files
    datadir = './fakes_gwb'

    parfiles = sorted(glob.glob(datadir + '/par/' + '*.par'))
    timfiles = sorted(glob.glob(datadir + '/tim/' + '*.tim'))

    psrs = []
    for p, t in zip(parfiles, timfiles):
        # psrname = parfiles[0].split('/')[-1].split('_')[0]
        psr = Pulsar(p, t, ephem='DE438')
        psrs.append(psr)
    return psrs


def gwb_ul(psrs_cut, num_points):
    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs_cut]
    tmax = [p.toas.max() for p in psrs_cut]
    Tspan = np.max(tmax) - np.min(tmin)
    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)
    # white noise parameters
    # we set these ourselves so we know the most likely values!
    efac = parameter.Constant(1)
    # quad = parameter.Constant(0)
    # ecorr = parameter.Constant(0)

    # red noise parameters
    log10_A = parameter.LinearExp(-20, -11)
    gamma = parameter.Uniform(0, 7)

    # GW parameters (initialize with names here to use parameters in common across pulsars)
    log10_A_gw = parameter.LinearExp(-18,-12)('log10_A_gw')
    gamma_gw = parameter.Constant(4.33)('gamma_gw')
    # white noise
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    # eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
    # ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

    # red noise (powerlaw with 30 frequencies)
    pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
    rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

    # gwb (no spatial correlations)
    cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
    gw = gp_signals.FourierBasisGP(spectrum=cpl, components=30, Tspan=Tspan, name='gw')

    # timing model
    tm = gp_signals.TimingModel(use_svd=True) # stabilizing timing model design matrix with SVD
    s = ef + rn + gw + tm

    # intialize PTA
    models = []

    for p in psrs_cut:
        models.append(s(p))

    pta = signal_base.PTA(models)
    outDir = './chains/psrs/{0}'.format(psrs_cut[0].name)
    sample = sampler.setup_sampler(pta, outdir=outDir)
    x0 = np.hstack([p.sample() for p in pta.params])

    # sampler for N steps
    N = int(num_points)  # normally, we would use 5e6 samples (this will save time)
    sample.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, )

    chain = np.loadtxt(os.path.join(outDir, 'chain_1.txt'))
    pars = np.loadtxt(outDir + '/pars.txt', dtype=np.unicode_)
    ind = list(pars).index('log10_A_gw')

    UL, unc = model_utils.ul(chain[:,ind])
    return UL, unc


# def find_ul(psrs, best_list, ul_list, i):
#     # for i in range(start, len(psrs)):  # find the best pulsar each time
#     psrscut = []  # pulsars to be tested
#     for num in best_list:
#         psrscut.append(psrs[num])  # append best_list to psrscut
#         print(psrs[num].name)
#     psrscut.append(psrs[i])  # append the test case to psrscut
#     print(psrs[i].name)
#     if i in best_list:  # don't double up on pulsars
#         print('psr in list already... skipping....')
#         ul = 1
#     else:
#         print('number of pulsars =', len(psrscut))
#         ul = gwb_ul(psrscut)[0]
#     fname = 'ul_data_{0}_run_{1}.txt'.format(len(best_list) + 1, k)
#     with open(fname, 'ab') as f:
#         np.savetxt(f, np.c_[i, ul])  # save the upper limits to file
#     # best_list.append(np.argmin(ul_list)[0]) # append the best ul to best_list
#     return ul


def load_real():
    try:
        with open("psrs.p", "rb") as f:
            psrs_cut = pickle.load(f)
        return psrs_cut
    except:
        psrlist = None # define a list of pulsar name strings that can be used to filter.
        # set the data directory
        datadir = './data'
        if not os.path.isdir(datadir):
            datadir = '../data'
        # print(datadir)
        # for the entire pta
        parfiles = sorted(glob.glob(datadir + '/partim/*par'))
        timfiles = sorted(glob.glob(datadir + '/partim/*tim'))
        # print(parfiles)
        # filter
        if psrlist is not None:
            parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in psrlist]
            timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in psrlist]

        # Make sure you use the tempo2 parfile for J1713+0747!!
        # ...filtering out the tempo parfile... 
        parfiles = [x for x in parfiles if 'J1713+0747_NANOGrav_12yv3.gls.par' not in x]
        psrs = []
        ephemeris = 'DE436'
        for p, t in zip(parfiles, timfiles):
            psr = Pulsar(p, t, ephem=ephemeris)
            psrs.append(psr)
        psrs_cut = []
        for p in psrs:
            time_tot = (max(p.toas) - min(p.toas)) / 86400 / 365.25
            if time_tot > 6:
                psrs_cut.append(p)
        with open("psrs.p", "wb") as f:
            pickle.dump(psrs_cut, f)
        print(len(psrs_cut))
        return psrs_cut


# def multi_ul(psr_list):
#     i = psr_list[0]
#     psr = psr_list[1]
#     ul, unc = gwb_ul([psr], 100000)
#     return i, ul


# def multi_ul_all(psr_list):
#     i = psr_list[0]
#     psr = psr_list[1]
#     ul, unc = gwb_ul(psr, 1e6)
#     return i, ul


def sngl_ul_order(psrs):
    ul_list = []
    count = []
    for i in range(len(psrs)):
        ul, unc = gwb_ul(psrs, 100000)
        count.append(i)
        ul_list.append(ul)
    count = np.array(count)
    ul_list = np.array(ul_list)
    data = np.column_stack([count, ul_list])
    data_sort = data[data[:, 1].argsort()]
    ul_best = data_sort[:, 0]
    ul_list = data_sort[:, 1]
    return ul_best, ul_list


def all_ul_order(psrs, ul_list, k, amp):
    count = []
    ul_list = []
    for i in range(len(psrs)):
        ul, unc = gwb_ul(psrs, 1e6)
        with open('uls_run_{0}_amp_{1}'.format(k, amp), 'ab') as f:
            np.savetxt(f, np.c_[count, ul_list])
    return ul_list


def main():
    real_psrs = load_real()
    num_trials = 10  # number of realizations
    amp = 3e-15  # GWB injection amplitude
    for k in range(num_trials):
        print(k)
        make_fakes(amp, real_psrs)  # make fake pulsar set
        fakes = load_fakes()
        __, ul_list = sngl_ul_order(fakes)
        # ul_list = [21, 16, 11, 9, 18, 7, 8, 19, 14,  3,  6,  0,  2,  1, 13,  4, 20, 17,
        #            5, 12, 10, 15]
        all_ul_order(fakes, ul_list, k, amp)


if __name__ == '__main__':
    main()
