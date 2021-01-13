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
        psrname = parfiles[0].split('/')[-1].split('_')[0]
        psr = Pulsar(p, t, ephem='DE438')
        psrs.append(psr)


def gwb_ul(psrs_cut):

    # find the maximum time span to set GW frequency sampling
    tmin = [p.toas.min() for p in psrs_cut]
    tmax = [p.toas.max() for p in psrs_cut]
    Tspan = np.max(tmax) - np.min(tmin)
    # define selection by observing backend
    selection = selections.Selection(selections.by_backend)
    # white noise parameters
    # we set these ourselves so we know the most likely values!
    efac = parameter.Constant(1)
    equad = parameter.Constant(0)
    ecorr = parameter.Constant(0)

    # red noise parameters
    log10_A = parameter.LinearExp(-20, -11)
    gamma = parameter.LinearExp(0, 7)

    # GW parameters (initialize with names here to use parameters in common across pulsars)
    log10_A_gw = parameter.LinearExp(-18,-12)('log10_A_gw')
    gamma_gw = parameter.Constant(4.33)('gamma_gw')
    # white noise
    ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=selection)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

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
    outDir = './chains/{0}_psrs/{1}'.format(len(best_list) + 1, psrs_cut[-1].name)
    sample = sampler.setup_sampler(pta, outdir=outDir)
    x0 = np.hstack([p.sample() for p in pta.params])

    # sampler for N steps
    N = int(1e6)  # normally, we would use 5e6 samples (this will save time)
    sample.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50, )

    chain = np.loadtxt(os.path.join(outDir, 'chain_1.txt'))
    pars = np.loadtxt(outDir + '/pars.txt', dtype=np.unicode_)
    ind = list(pars).index('log10_A_gw')

    UL = model_utils.ul(chain[:,ind])
    return UL


def find_ul(psrs, best_list, ul_list, k, i):
    # for i in range(start, len(psrs)):  # find the best pulsar each time
    psrscut = []  # pulsars to be tested
    for num in best_list:
        psrscut.append(psrs[num])  # append best_list to psrscut
        print(psrs[num].name)
    psrscut.append(psrs[i])  # append the test case to psrscut
    print(psrs[i].name)
    if i in best_list:  # don't double up on pulsars
        print('psr in list already... skipping....')
        ul = 1
    else:
        print('number of pulsars =', len(psrscut))
        ul = gwb_ul(psrscut)[0]
    fname = 'ul_data_{0}_run_{1}.txt'.format(len(best_list) + 1, k)
    with open(fname, 'ab') as f:
        np.savetxt(f, np.c_[i, ul])  # save the upper limits to file
    # best_list.append(np.argmin(ul_list)[0]) # append the best ul to best_list
    return ul


def multi_ul(k, i):
    ul = find_ul(psrs, best_list, ul_list, k, i)
    return ul


# load real data
with open("psrs.p", "rb") as f:
    psrs = pickle.load(f)


num_trials = 10  # number of realizations
amp = 3e-15
n = 0  # number of times the trials go lower than GWB injection
for k in range(num_trials):

    make_fakes(amp, psrs)  # make fake pulsar set
    best_list = []  # to store pulsar order
    filename = 'best_list_run_{0}_amp_{1}.txt'.format(k, amp)
    for j in range(len(best_list), len(psrs)):  # cycle through once for each pulsar
        ul_list = []
        for i in range(len(psrs)):
            upper = multi_ul(k, i)
            ul_list.append(upper)
        best_ul = np.argmin(ul_list)
        best_list.append(best_ul)
        with open(filename, 'ab') as f:
            np.savetxt(f, np.c_[k, j, best_ul])
        if best_ul < amp:
            n += 1
            break
        elif j == 5 and best_ul > amp:
            print('With 5 pulsars, the ul is still above the GWB injection.')
            break






