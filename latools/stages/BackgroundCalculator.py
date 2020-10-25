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
