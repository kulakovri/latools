import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from scipy.optimize import curve_fit
from ..helpers.helpers import rolling_window
from ..helpers.stat_fns import *
from latools import analyse


class Despiker(analyse):
    def __init__(self, raw, expdecay_despiker=False, exponent=None,
                noise_despiker=True, win=3, nlim=12., exponentplot=False,
                maxiter=4, autorange_kwargs={}, focus_stage='rawdata'):
        if focus_stage != self.focus_stage:
            self.set_focus(focus_stage)

        self._move_attributes_from_previous_stage(raw)
        if expdecay_despiker and exponent is None:
            if not hasattr(self, 'expdecay_coef'):
                self.find_expcoef(plot=exponentplot,
                                  autorange_kwargs=autorange_kwargs)
            exponent = self.expdecay_coef
            time.sleep(0.1)

        with self.pbar.set(total=len(self.data), desc='Despiking') as prog:
            for d in self.data.values():
                d.despike(expdecay_despiker, exponent,
                          noise_despiker, win, nlim, maxiter)
                prog.update()

        self.stages_complete.update(['despiked'])
        self.focus_stage = 'despiked'
        return

    def _move_attributes_from_previous_stage(self, rawloader):
        self.__dict__ = rawloader.__dict__

    def find_expcoef(self, nsd_below=0., plot=False,
                     trimlim=None, autorange_kwargs={}):
        """
        Determines exponential decay coefficient for despike filter.

        Fits an exponential decay function to the washout phase of standards
        to determine the washout time of your laser cell. The exponential
        coefficient reported is `nsd_below` standard deviations below the
        fitted exponent, to ensure that no real data is removed.

        Total counts are used in fitting, rather than a specific analyte.

        Parameters
        ----------
        nsd_below : float
            The number of standard deviations to subtract from the fitted
            coefficient when calculating the filter exponent.
        plot : bool or str
            If True, creates a plot of the fit, if str the plot is to the
            location specified in str.
        trimlim : float
            A threshold limit used in determining the start of the
            exponential decay region of the washout. Defaults to half
            the increase in signal over background. If the data in
            the plot don't fall on an exponential decay line, change
            this number. Normally you'll need to increase it.

        Returns
        -------
        None
        """
        print('Calculating exponential decay coefficient\nfrom SRM washouts...')

        def findtrim(tr, lim=None):
            trr = np.roll(tr, -1)
            trr[-1] = 0
            if lim is None:
                lim = 0.5 * np.nanmax(tr - trr)
            ind = (tr - trr) >= lim
            return np.arange(len(ind))[ind ^ np.roll(ind, -1)][0]

        if not hasattr(self.stds[0], 'trnrng'):
            for s in self.stds:
                s.autorange(**autorange_kwargs, ploterrs=False)

        trans = []
        times = []
        for v in self.stds:
            for trnrng in v.trnrng[-1::-2]:
                tr = minmax_scale(v.data['total_counts'][(v.Time > trnrng[0]) & (v.Time < trnrng[1])])
                sm = np.apply_along_axis(np.nanmean, 1,
                                         rolling_window(tr, 3, pad=0))
                sm[0] = sm[1]
                trim = findtrim(sm, trimlim) + 2
                trans.append(minmax_scale(tr[trim:]))
                times.append(np.arange(tr[trim:].size) *
                             np.diff(v.Time[1:3]))

        times = np.concatenate(times)
        times = np.round(times, 2)
        trans = np.concatenate(trans)

        ti = []
        tr = []
        for t in np.unique(times):
            ti.append(t)
            tr.append(np.nanmin(trans[times == t]))

        def expfit(x, e):
            """
            Exponential decay function.
            """
            return np.exp(e * x)

        ep, ecov = curve_fit(expfit, ti, tr, p0=(-1.))

        eeR2 = R2calc(trans, expfit(times, ep))

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=[6, 4])

            ax.scatter(times, trans, alpha=0.2, color='k', marker='x', zorder=-2)
            ax.scatter(ti, tr, alpha=1, color='k', marker='o')
            fitx = np.linspace(0, max(ti))
            ax.plot(fitx, expfit(fitx, ep), color='r', label='Fit')
            ax.plot(fitx, expfit(fitx, ep - nsd_below * np.diag(ecov)**.5, ),
                    color='b', label='Used')
            ax.text(0.95, 0.75,
                    ('y = $e^{%.2f \pm %.2f * x}$\n$R^2$= %.2f \nCoefficient: '
                     '%.2f') % (ep,
                                np.diag(ecov)**.5,
                                eeR2,
                                ep - nsd_below * np.diag(ecov)**.5),
                    transform=ax.transAxes, ha='right', va='top', size=12)
            ax.set_xlim(0, ax.get_xlim()[-1])
            ax.set_xlabel('Time (s)')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Proportion of Signal')
            plt.legend()
            if isinstance(plot, str):
                fig.savefig(plot)

        self.expdecay_coef = ep - nsd_below * np.diag(ecov)**.5

        print('  {:0.2f}'.format(self.expdecay_coef[0]))

        return
