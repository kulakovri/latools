import warnings
import numpy as np
import uncertainties.unumpy as un
from ..helpers.helpers import un_interp1d, Bunch
from ..helpers.stat_fns import gauss_weighted_stats
from latools import analyse


class BackgroundCalculator(analyse):
    def __init__(self, autoranged, analytes=None, weight_fwhm=None,
                 n_min=20, n_max=None, cstep=None, errtype='stderr',
                 bkg_filter=False, f_win=7, f_n_lim=3, focus_stage='despiked'):
        self.move_attributes_from_previous_stage(autoranged)
        if analytes is None:
            analytes = self.analytes
            self.bkg = Bunch()
        elif isinstance(analytes, str):
            analytes = [analytes]

        if weight_fwhm is None:
            weight_fwhm = 600  # 10 minute default window

        self.get_background(n_min=n_min, n_max=n_max,
                            bkg_filter=bkg_filter,
                            f_win=f_win, f_n_lim=f_n_lim, focus_stage=focus_stage)

        # Gaussian - weighted average
        if 'calc' not in self.bkg.keys():
            # create time points to calculate background
            if cstep is None:
                cstep = weight_fwhm / 20
            elif cstep > weight_fwhm:
                warnings.warn("\ncstep should be less than weight_fwhm. Your backgrounds\n" +
                              "might not behave as expected.\n")
            bkg_t = np.linspace(0,
                                self.max_time,
                                int(self.max_time // cstep))
            self.bkg['calc'] = Bunch()
            self.bkg['calc']['uTime'] = bkg_t

        # TODO : calculation then dict assignment is clumsy...
        mean, std, stderr = gauss_weighted_stats(self.bkg['raw'].uTime,
                                                 self.bkg['raw'].loc[:, analytes].values,
                                                 self.bkg['calc']['uTime'],
                                                 fwhm=weight_fwhm)
        self.bkg_interps = {}

        for i, a in enumerate(analytes):
            self.bkg['calc'][a] = {'mean': mean[i],
                                    'std': std[i],
                                    'stderr': stderr[i]}
            self.bkg_interps[a] = un_interp1d(x=self.bkg['calc']['uTime'],
                                              y=un.uarray(self.bkg['calc'][a]['mean'],
                                                          self.bkg['calc'][a][errtype]))

    # functions for background correction
    def get_background(self, n_min=10, n_max=None, focus_stage='despiked', bkg_filter=False, f_win=5, f_n_lim=3):
        """
        Extract all background data from all samples on universal time scale.
        Used by both 'polynomial' and 'weightedmean' methods.

        Parameters
        ----------
        n_min : int
            The minimum number of points a background region must
            have to be included in calculation.
        n_max : int
            The maximum number of points a background region must
            have to be included in calculation.
        filter : bool
            If true, apply a rolling filter to the isolated background regions
            to exclude regions with anomalously high values. If True, two parameters
            alter the filter's behaviour:
        f_win : int
            The size of the rolling window
        f_n_lim : float
            The number of standard deviations above the rolling mean
            to set the threshold.
        focus_stage : str
            Which stage of analysis to apply processing to.
            Defaults to 'despiked' if present, or 'rawdata' if not.
            Can be one of:
            * 'rawdata': raw data, loaded from csv file.
            * 'despiked': despiked data.
            * 'signal'/'background': isolated signal and background data.
              Created by self.separate, after signal and background
              regions have been identified by self.autorange.
            * 'bkgsub': background subtracted data, created by
              self.bkg_correct
            * 'ratios': element ratio data, created by self.ratio.
            * 'calibrated': ratio data calibrated to standards, created by self.calibrate.

        Returns
        -------
        pandas.DataFrame object containing background data.
        """
        allbkgs = {'uTime': [],
                   'ns': []}

        if focus_stage == 'despiked':
            if 'despiked' not in self.stages_complete:
                focus_stage = 'rawdata'

        for a in self.analytes:
            allbkgs[a] = []

        n0 = 0
        for s in self.data.values():
            if sum(s.bkg) > 0:
                allbkgs['uTime'].append(s.uTime[s.bkg])
                allbkgs['ns'].append(enumerate_bool(s.bkg, n0)[s.bkg])
                n0 = allbkgs['ns'][-1][-1]
                for a in self.analytes:
                    allbkgs[a].append(s.data[focus_stage][a][s.bkg])

        allbkgs.update((k, np.concatenate(v)) for k, v in allbkgs.items())
        bkgs = pd.DataFrame(allbkgs)  # using pandas here because it's much more efficient than loops.

        self.bkg = Bunch()
        # extract background data from whole dataset
        if n_max is None:
            self.bkg['raw'] = bkgs.groupby('ns').filter(lambda x: len(x) > n_min)
        else:
            self.bkg['raw'] = bkgs.groupby('ns').filter(lambda x: (len(x) > n_min) & (len(x) < n_max))
        # calculate per - background region stats
        self.bkg['summary'] = self.bkg['raw'].groupby('ns').aggregate([np.mean, np.std, stderr])
        # sort summary by uTime
        self.bkg['summary'].sort_values(('uTime', 'mean'), inplace=True)
        # self.bkg['summary'].index = np.arange(self.bkg['summary'].shape[0])
        # self.bkg['summary'].index.name = 'ns'

        if bkg_filter:
            # calculate rolling mean and std from summary
            t = self.bkg['summary'].loc[:, idx[:, 'mean']]
            r = t.rolling(f_win).aggregate([np.nanmean, np.nanstd])
            # calculate upper threshold
            upper = r.loc[:, idx[:, :, 'nanmean']] + f_n_lim * r.loc[:, idx[:, :, 'nanstd']].values
            # calculate which are over upper threshold
            over = r.loc[:, idx[:, :, 'nanmean']] > np.roll(upper.values, 1, 0)
            # identify them
            ns_drop = over.loc[over.apply(any, 1), :].index.values
            # drop them from summary
            self.bkg['summary'].drop(ns_drop, inplace=True)
            # remove them from raw
            ind = np.ones(self.bkg['raw'].shape[0], dtype=bool)
            for ns in ns_drop:
                ind = ind & (self.bkg['raw'].loc[:, 'ns'] != ns)
            self.bkg['raw'] = self.bkg['raw'].loc[ind, :]
        return

class BackgroundSubtractor(analyse):
    def __init__(self, autoranged, analytes=None, errtype='stderr', focus_stage='despiked'):
        self.move_attributes_from_previous_stage(autoranged)
        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        if focus_stage == 'despiked':
            if 'despiked' not in self.stages_complete:
                focus_stage = 'rawdata'

        # make uncertainty-aware background interpolators
        # bkg_interps = {}
        # for a in analytes:
        #     bkg_interps[a] = un_interp1d(x=self.bkg['calc']['uTime'],
        #                                  y=un.uarray(self.bkg['calc'][a]['mean'],
        #                                              self.bkg['calc'][a][errtype]))
        # self.bkg_interps = bkg_interps

        # apply background corrections
        with self.pbar.set(total=len(self.data), desc='Background Subtraction') as prog:
            for d in self.data.values():
                # [d.bkg_subtract(a, bkg_interps[a].new(d.uTime), None, focus_stage=focus_stage) for a in analytes]
                [d.bkg_subtract(a, self.bkg_interps[a].new(d.uTime), ~d.sig, focus_stage=focus_stage) for a in analytes]
                d.setfocus('bkgsub')

                prog.update()

        self.stages_complete.update(['bkgsub'])
        self.focus_stage = 'bkgsub'
        return
