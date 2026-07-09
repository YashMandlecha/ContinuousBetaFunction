from collections.abc import Callable
import typing
import pickle as _pickle
import gc as _gc
import warnings
import time

import gvar as _gvar
import numpy as _numpy
from scipy.special import zeta as _zeta

class BetaFunctionLog(object):
    def __init__(self, fn):
        if fn is not None:
            self.fn = fn
            self.file = open(fn, 'w+')
    def __call__(self, 
                 message, 
                 category, 
                 filename, 
                 lineno, 
                 file = None, 
                 line = None
        ): # Redirects warnings to log file
        msg = 25 * '-.' + '\n' + 'WARNING:\n'
        msg += warnings.formatwarning(message, category, filename, lineno)
        msg += 25 * '-.' + '\n'
        self.file.write(msg)
        self.file.close()
        self.file = open(self.fn, 'a')
    def write(self, *kargs):
        msg = 25 * '-.' + '\n'
        for arg in kargs: msg += arg + '\n'
        msg += 25 * '-.' + '\n'
        self.file.write(msg)
        self.file.close()
        self.file = open(self.fn, 'a')
    def close(self): 
        if self.fn is not None: self.file.close()

class BetaFunctionException(Exception):
    def __init__(self, *kargs): 
        self.msg = 25 * '-.' + '\n'
        for arg in kargs: self.msg += arg + '\n'
        self.msg = 25 * '-.' + '\n'
    def __str__(self): return self.msg

# Perturbative gradient flow beta-function from JHEP06(2019)121
class PerturbativeBetaFunction(object): 
    def __init__(self, 
                 nf: float | int,
                 nc: float | int
                 ):
        # Gauge in adjoint & fermions in fundamental
        tr = 0.5
        tf = tr*nf # differs from JHEP02(2017)090 by "nf"
        cf = 0.5*(nc*nc - 1.)/nc
        ca = nc

        # 1-3 loop MS-bar: PRL93B(1980)429, JHEP02(2017)090
        b0 = 11.*ca/3. - 4.*tf/3.
        b1 = 34.*ca*ca/3. - (4.*cf + 20.*ca/3.)*tf
        b2 = 2857.*ca*ca*ca/54. - 1415.*ca*ca*tf/27.
        b2 += (-205.*cf*ca/9. + 2.*cf*cf)*tf
        b2 += (44.*cf/9. + 158.*ca/27.)*tf*tf

        # 3-loop gradient flow: JHEP06(2019)121
        rho = 1./8.
        e10 = (52./9. + 22.*_numpy.log(2.)/3. - 3.*_numpy.log(3.))*ca - 8.*tf/9.
        e20 = 27.9786*ca*ca - 31.5652*tf*ca + (16.* _zeta(3.) - 43./3.)*tf*cf
        e20 += (8.*_numpy.pi*_numpy.pi/27. - 80./81.)*tf*tf
        L = _numpy.log(2.*rho) + _numpy.euler_gamma
        e1 = e10 + b0*L
        e2 = e20 + (2.*b0*e10 + b1)*L + (b0*L)*(b0*L)
        b2 = b2 - e1*b1 + (e2 - e1*e1)*b0

        # Conventional normalizations
        self.nrm = 4.*_numpy.pi
        self.b = [b/self.nrm**(n+1) for n,b in enumerate([b0,b1,b2])]

    def __call__(self, x: any, loops: int = 3) -> any:
        bfn = sum(b*(x/self.nrm)**(n+2) for n,b in enumerate(self.b) if n < loops)
        return -self.nrm*bfn

# Main class for setting calculation of beta-function up
class SetupBetaFunction(object):
    def __init__(
            self,
            nc: float | int = 3., 
            nf: float | int = None, 
            gauge_action: str = 's', # s = sym, w = wils, c = clov, h = hisq
            logfn: str = None
        ):
        if nf is None: raise BetaFunctionException('must specify nf')
        self.nf, self.nc = nf, nc
        
        self.perturbative_beta_function = PerturbativeBetaFunction(self.nf, self.nc)
        
        self._coupling_norm = 128.*_numpy.pi*_numpy.pi/3./(self.nc*self.nc-1.)
        
        self.os = ['p', 's', 'c']
        self.mc_observables = []
        for os in self.os: self.mc_observables.append('E' + os)
        self.mc_observables.append('Q')
        
        self._binsize = 1
        self.data = {}
        self.avg_data = {}

        self.log = BetaFunctionLog(logfn)
        if logfn is not None: warnings.showwarning = self.log

        self.gauge_action = gauge_action

    def _dxdlogt(self, x: any, t: any) -> any: # -dx/dlogt (5-point)
        x, t = _numpy.array(x), _numpy.array(t)
        dx = -x[4:] + 8.*x[3:-1] - 8.*x[1:-3] + x[:-4]
        dlogt = 6.*(t[4:] - t[:-4])/(t[4:] + t[:-4])
        return -dx/dlogt

    def _Q_filter(self, D):
        Q_in_keys = 'Q' in D.keys()
        not_inf = not (_numpy.isinf(self._min_Q) and _numpy.isinf(self._max_Q))
        if (Q_in_keys and not_inf): 
            Q = D['Q'][-1]
            filter = lambda d: [
                da for cf,da in enumerate(d) 
                if self._min_Q <= Q[cf] <= self._max_Q
            ]
            return filter
        else: return lambda d: d

    def _get(self, 
             coupling: str, 
             volume: str,
             mass: str, 
             flow: str, 
             path: str
        ) -> dict[str,list[float]]:
        fn = path + '_'.join([coupling,volume,mass,flow]) + '.bin'
        with open(fn,'rb') as in_file: data = _pickle.load(in_file)
        qf = self._Q_filter(data)
        return {
            '_'.join([o,t]): qf(data[o][n])
            for n,t in enumerate(data['flow_times']) 
            for o in data.keys() if o != 'flow_times'
            if self._min_fv_flt <= float(t) <= self._max_fv_flt
        }

    def _rearrange(
            self, 
            data: dict[str,dict[str,any]]
        ) -> dict[str,any]:
        return {
            '_'.join([flow,ot]): data[flow][ot] 
            for flow in data.keys()
            for ot in data[flow].keys()
        }

    def _preprocess(self, data: dict[str,any]) -> dict[str,any]:
        return _gvar.dataset.bin_data(data, binsize = self._binsize) 

    def _average(self, 
                 data: dict[str,_gvar.GVar], 
                 process: any = None
        ) -> dict[str,_gvar.GVar]:
        return _gvar.dataset.avg_data(data if process is None else process(data))

    def _sort_flow_times(self, key: str) -> float: return float(key)

    def _flow_times(self, data: dict[str,any]) -> list[float]:
        flow_times = list(set([o.split('_')[-1] for o in data.keys()]))
        flow_times.sort(key = self._sort_flow_times)
        return [str(t) for t in flow_times]

    def _undorearrange(
            self, 
            data: dict[str,_gvar.GVar], 
            flows: list[str],
            data_ref: dict[str,any]
        ) -> dict[str,dict[str,list[any]]]:
        return {
            flow: {
                o:
                    [
                        data['_'.join([flow,o,t])]
                        for t in self._flow_times(data_ref[flow])
                    ] for o in self.mc_observables
            } for flow in flows
        }
    
    def delta(self,
               flow_times: any,
               volume: str,
               flow: str,
               observable: str
        ) -> any:
        match self._correction:
            case 'finite-volume': # arXiv:1208.1051
                dims = volume.replace('t','l').split('l')[1:]
                da = -64.*_numpy.pi*_numpy.pi/3.
                de = 1.
                for dim in [*map(float, dims)]:
                    r = dim*dim/flow_times
                    da /= _numpy.sqrt(r)
                    de *= 1. + 2.*_numpy.exp(-0.125*r) + 2.*_numpy.exp(-0.5*r)
                return da + de - 1.
            case 'tree-level-normalization' | 'tln': # arXiv:1406.0827
                flow_translation = {
                    'symanzik': 's',
                    'C0p0': 'p',
                    'wilson': 'p',
                    'C13': 'a'
                }
                if flow not in list(flow_translation.keys()):
                    raise BetaFunctionException(flow+' not known flow for tln')
                dpath = self._tln_path + '/'
                dpath += self.gauge_action + flow_translation[flow] + observable
                dpath += volume + '.tln'
                t,d = [],[]
                with open(dpath,'r') as in_file:
                    for line in in_file.readlines():
                        tv,dv,_,_ = [*map(float,line.split())]
                        t.append(tv)
                        d.append(dv)
                spline = _gvar.cspline.CSpline(t,d)
                return _numpy.array([spline(t) for t in flow_times])-1.
            case _: return _numpy.array([0. for _ in flow_times])

    def _norm(self, 
              flow_times: any,
              volume: str,
              flow: str, 
              observable: str
        ) -> any:
        Ctlb = 1. + self.delta(flow_times, volume, flow, observable)
        return self._coupling_norm*flow_times*flow_times/Ctlb

    
    def get_g2GF_betaGF_and_Q(self, 
                     data, 
                     data_ref, 
                     flow, 
                     volume, 
                     coupling,
                     mass # not used
        ):
        flow_times = _numpy.array(
            [float(t) for t in self._flow_times(data_ref[flow])]
        )

        # Combining observables
        pdata = data
        if self.combine is not None: # TODO: fix issue w/ having multiple coupled entries in "combine"
            for o in self.combine.keys():
                pdata[flow]['E'+o] = sum(
                    self.combine[o][oo]*_numpy.array(pdata[flow]['E'+oo]) for oo in self.combine[o].keys())
            
        # Running coupling (g^2_O = norm * t^2<E_O(t)> / (1 + delta(L,beta,O)))
        result = {
            '_'.join(['g2',o[-1]]): 
            self._norm(flow_times, volume, flow, o[-1])*pdata[flow][o]    
            for o in pdata[flow].keys() if ('E' in o) and (o[-1] in self.os)
        }

        # Beta-function (beta = -t dg^2 / dt = -dg^2 / dlogt)
        olst = list(result.keys())
        for o in olst:
            result['_'.join(['beta',o[-1]])] = self._dxdlogt(result[o], flow_times)
            result[o] = result[o][2:-2]

        # Topological charge, flow times, return
        result['Q'] = pdata[flow]['Q'][2:-2]
        result['flow_times'] = [str(t) for t in flow_times][2:-2]

        # Convert array into dictionary & return
        for o in result.keys():
            if o != 'flow_times':
                result[o] = {
                    t: result[o][n] 
                    for n,t in enumerate(result['flow_times'])
                }
        return result

    def process_data(
            self, 
            data: dict[str,dict[str,dict[str,list[str]]]],
            path: str = '',
            get_data: any = None,
            average_data: any = None,
            preprocess_data: any = None,
            correction: str = 'finite-volume',
            tree_level_normalization_data_path: str = './',
            combine: any = None,
            mnt: float = 0.,
            mxt: float = _numpy.inf, 
            mnQ: float = -_numpy.inf,
            mxQ: float = _numpy.inf,
            verbosity: int = 0,
        ):
        average = self._average if average_data is None else average_data
        get = self._get if get_data is None else get_data
        process = self._preprocess if preprocess_data is None else preprocess_data
        self._correction = correction
        self._min_fv_flt, self._max_fv_flt = mnt, mxt
        self._min_Q, self._max_Q = mnQ, mxQ

        match self._correction:
            case 'tree-level-normalization' | 'tln': 
                self._tln_path = tree_level_normalization_data_path
            case _: pass

        del self.data, self.avg_data
        _gc.collect()

        self.data = {}
        self.avg_data = {}

        self.combine = combine

        for coupling in data.keys():
            self.data[coupling],self.avg_data[coupling] = {},{}
            for volume in data[coupling].keys():
                self.data[coupling][volume],self.avg_data[coupling][volume] = {},{}
                for mass in data[coupling][volume]:
                    if verbosity >= 1: 
                        self._start_timer()
                        msg = 25 * '-.' + '\nbeta_b = ' + coupling.replace('p','.')
                        msg += ', vol = ' + volume
                        msg += ', mass = ' + mass
                        print(msg)
                    
                    flows = data[coupling][volume][mass]
                    predata = {
                        flow: get(coupling,volume,mass,flow,path) for flow in flows
                    }
                    predata_rearranged = self._rearrange(predata)
                    self.data[coupling][volume][mass] = self._undorearrange(
                        predata_rearranged,flows,predata
                    )
                    for f in flows:
                        ts = self._flow_times(predata[f])
                        self.data[coupling][volume][mass][f]['flow_times'] = ts
                    avg_data = self._undorearrange(
                        average(predata_rearranged,process=process),flows,predata
                    )
                    self.avg_data[coupling][volume][mass] = {
                        flow: self.get_g2GF_betaGF_and_Q(
                            avg_data,predata,flow,volume,coupling,mass
                        ) for flow in flows
                    }
                    if verbosity >= 1:
                        print('dt =', self._stop_timer(), '(secs)\n' + 25 * '-.')

    def _start_timer(self): self._ti = time.time()

    def _stop_timer(self): return round(time.time() - self._ti, 2)

    def _iv_xtrp_fcn(self, x: any, p: dict[str,any]) -> any:
        return p['k1(t;beta)'][0] + p['k2(t;beta)'][0]*x
    
    def _ch_xtrp_fcn(self, x: any, p: dict[str,any]) -> any:
        return p['k1(t;beta,L)'][0] + p['k2(t;beta,L)'][0]*x

    def set_binsize(self, binsize: int): self._binsize = binsize
    
    def load(self, info: str, fn: str, **kwargs):
        match info:
            case 'fv': self.avg_data = _gvar.load(fn, **kwargs)
            case 'iv': self.iv_fits = _gvar.load(fn, **kwargs)
            case _: BetaFunctionException(info + ' is not a valid option.')

    def load_iv(
            self,
            fn: str,
            fcn: Callable[[any,dict[str,any]],any] = None,
            exclude: dict[str,list[str]] = None,
            **kwargs
        ):
        del self.iv_fits
        _gc.collect()
        self.load('iv',fn,**kwargs)
        if fcn is None: self.iv_fcn = self._iv_xtrp_fcn
        if exclude is None:
            self._iv_exclude = {c: [] for c in self.avg_data.keys()}
        else: self._iv_exclude = exclude

    def save(self, info: str, fn: str, **kwargs):
        match info:
            case 'fv': _gvar.dump(self.avg_data, fn, **kwargs)
            case 'iv': _gvar.dump(self.iv_fits, fn, **kwargs)
            case _: BetaFunctionException(info + ' is not a valid option.')

    def get(self, info: str):
        match info:
            case 'fv': return self.avg_data
            case 'iv': return self.iv_fits
            case _: BetaFunctionException(info + ' is not a valid option.')

if __name__ == '__main__': pass
