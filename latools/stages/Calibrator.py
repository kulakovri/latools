import pandas as pd
import numpy as np
import uncertainties as unc

from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from ..helpers.helpers import un_interp1d, Bunch
from ..helpers.stat_fns import un, nominal_values
from ..helpers import srm as srms
from ..helpers.chemistry import decompose_molecule
from ..helpers.analyte_names import get_analyte_name, analyte_2_massname
from latools import analyse

idx = pd.IndexSlice  # multi-index slicing!


class Calibrator(analyse):
    def __init__(self, ratio_calculated, analytes=None, drift_correct=True,
                 srms_used=['NIST610', 'NIST612', 'NIST614'],
                 zero_intercept=True, n_min=10, reload_srm_database=False):
        if analytes is None:
            analytes = self.analytes_sorted(self.analytes.difference([self.internal_standard]))

        elif isinstance(analytes, str):
            analytes = [analytes]

        if isinstance(srms_used, str):
            srms_used = [srms_used]

        if not hasattr(self, 'srmtabs'):
            self.srm_id_auto(srms_used=srms_used, n_min=n_min, reload_srm_database=reload_srm_database)

        # make container for calibration params
        gTime = np.asanyarray(self.caltab.index.levels[0])
        if not hasattr(self, 'calib_params'):
            self.calib_params = pd.DataFrame(columns=pd.MultiIndex.from_product([analytes, ['m']]),
                                             index=gTime)

        if zero_intercept:
            fn = lambda x, m: x * m
        else:
            fn = lambda x, m, c: x * m + c

        for a in analytes:
            if zero_intercept:
                if (a, 'c') in self.calib_params:
                    self.calib_params.drop((a, 'c'), 1, inplace=True)
            if drift_correct:
                for g in gTime:
                    if self.caltab.loc[g].size == 0:
                        continue
                    meas = self.caltab.loc[g, (a, 'meas_mean')].values
                    meas_err = self.caltab.loc[g, (a, 'meas_err')].values
                    srm = self.caltab.loc[g, (a, 'srm_mean')].values
                    srm_err = self.caltab.loc[g, (a, 'srm_err')].values
                    # TODO: replace curve_fit with Sambridge's 2D likelihood function for better uncertainty incorporation?
                    sigma = np.sqrt(meas_err ** 2 + srm_err ** 2)
                    if len(meas) > 1:
                        # multiple SRMs - do a regression
                        p, cov = curve_fit(fn, meas, srm, sigma=sigma)
                        pe = unc.correlated_values(p, cov)
                        self.calib_params.loc[g, (a, 'm')] = pe[0]
                        if not zero_intercept:
                            self.calib_params.loc[g, (a, 'c')] = pe[1]
                    else:
                        # deal with case where there's only one datum
                        self.calib_params.loc[g, (a, 'm')] = (un.uarray(srm, srm_err) /
                                                              un.uarray(meas, meas_err))[0]
                        if not zero_intercept:
                            self.calib_params.loc[g, (a, 'c')] = 0
            else:
                meas = self.caltab.loc[:, (a, 'meas_mean')].values
                meas_err = self.caltab.loc[:, (a, 'meas_err')].values
                srm = self.caltab.loc[:, (a, 'srm_mean')].values
                srm_err = self.caltab.loc[:, (a, 'srm_err')].values
                # TODO: replace curve_fit with Sambridge's 2D likelihood function for better uncertainty incorporation?
                sigma = np.sqrt(meas_err ** 2 + srm_err ** 2)

                if len(meas) > 1:
                    p, cov = curve_fit(fn, meas, srm, sigma=sigma)
                    pe = unc.correlated_values(p, cov)
                    self.calib_params.loc[:, (a, 'm')] = pe[0]
                    if not zero_intercept:
                        self.calib_params.loc[:, (a, 'c')] = pe[1]
                else:
                    self.calib_params.loc[:, (a, 'm')] = (un.uarray(srm, srm_err) /
                                                          un.uarray(meas, meas_err))[0]
                    if not zero_intercept:
                        self.calib_params.loc[:, (a, 'c')] = 0

        if self.calib_params.index.min() == 0:
            self.calib_params.drop(0, inplace=True)
            self.calib_params.drop(self.calib_params.index.max(), inplace=True)
        self.calib_params.loc[0, :] = self.calib_params.loc[self.calib_params.index.min(), :]
        maxuT = np.max([d.uTime.max() for d in self.data.values()])  # calculate max uTime
        self.calib_params.loc[maxuT, :] = self.calib_params.loc[self.calib_params.index.max(), :]
        # sort indices for slice access
        self.calib_params.sort_index(1, inplace=True)
        self.calib_params.sort_index(0, inplace=True)

        # calculcate interpolators for applying calibrations
        self.calib_ps = Bunch()
        for a in analytes:
            # TODO: revisit un_interp1d to see whether it plays well with correlated values.
            # Possible re-write to deal with covariance matrices?
            self.calib_ps[a] = {'m': un_interp1d(self.calib_params.index.values,
                                                 self.calib_params.loc[:, (a, 'm')].values)}
            if not zero_intercept:
                self.calib_ps[a]['c'] = un_interp1d(self.calib_params.index.values,
                                                    self.calib_params.loc[:, (a, 'c')].values)

        with self.pbar.set(total=len(self.data), desc='Applying Calibrations') as prog:
            for d in self.data.values():
                d.calibrate(self.calib_ps, analytes)
                prog.update()

        # record SRMs used for plotting
        markers = 'osDsv<>PX'  # for future implementation of SRM-specific markers.
        if not hasattr(self, 'srms_used'):
            self.srms_used = set(srms_used)
        else:
            self.srms_used.update(srms_used)
        self.srm_mdict = {k: markers[i] for i, k in enumerate(self.srms_used)}

        self.stages_complete.update(['calibrated'])
        self.focus_stage = 'calibrated'

        return

        # def calibrate(self, analytes=None, drift_correct=True,
        #               srms_used=['NIST610', 'NIST612', 'NIST614'],
        #               zero_intercept=True, n_min=10, reload_srm_database=False):
        #     """
        #     Calibrates the data to measured SRM values.

        #     Assumes that y intercept is zero.

        #     Parameters
        #     ----------
        #     analytes : str or iterable
        #         Which analytes you'd like to calibrate. Defaults to all.
        #     drift_correct : bool
        #         Whether to pool all SRM measurements into a single calibration,
        #         or vary the calibration through the run, interpolating
        #         coefficients between measured SRMs.
        #     srms_used : str or iterable
        #         Which SRMs have been measured. Must match names given in
        #         SRM data file *exactly*.
        #     n_min : int
        #         The minimum number of data points an SRM measurement
        #         must have to be included.

        #     Returns
        #     -------
        #     None
        #     """
        #     if analytes is None:
        #         analytes = self.analytes.difference(self.internal_standard)
        #     elif isinstance(analytes, str):
        #         analytes = [analytes]

        #     if isinstance(srms_used, str):
        #         srms_used = [srms_used]

        #     if not hasattr(self, 'srmtabs'):
        #         self.srm_id_auto(srms_used=srms_used, n_min=n_min, reload_srm_database=reload_srm_database)

        #     # make container for calibration params
        #     if not hasattr(self, 'calib_params'):
        #         gTime = self.stdtab.gTime.unique()
        #         self.calib_params = pd.DataFrame(columns=pd.MultiIndex.from_product([analytes, ['m']]),
        #                                         index=gTime)

        #     calib_analytes = self.srmtabs.index.get_level_values(0).unique()

        #     if zero_intercept:
        #         fn  = lambda x, m: x * m
        #     else:
        #         fn = lambda x, m, c: x * m + c

        #     for a in calib_analytes:
        #         if zero_intercept:
        #             if (a, 'c') in self.calib_params:
        #                 self.calib_params.drop((a, 'c'), 1, inplace=True)
        #         if drift_correct:
        #             for g in self.stdtab.gTime.unique():
        #                 ind = idx[a, :, :, g]
        #                 if self.srmtabs.loc[ind].size == 0:
        #                     continue
        #                 # try:
        #                 meas = self.srmtabs.loc[ind, 'meas_mean']
        #                 srm = self.srmtabs.loc[ind, 'srm_mean']
        #                 # TODO: replace curve_fit with Sambridge's 2D likelihood function for better uncertainty incorporation.
        #                 merr = self.srmtabs.loc[ind, 'meas_err']
        #                 serr = self.srmtabs.loc[ind, 'srm_err']
        #                 sigma = np.sqrt(merr**2 + serr**2)

        #                 if len(meas) > 1:
        #                     # multiple SRMs - do a regression
        #                     p, cov = curve_fit(fn, meas, srm, sigma=sigma)
        #                     pe = unc.correlated_values(p, cov)
        #                     self.calib_params.loc[g, (a, 'm')] = pe[0]
        #                     if not zero_intercept:
        #                         self.calib_params.loc[g, (a, 'c')] = pe[1]
        #                 else:
        #                     # deal with case where there's only one datum
        #                     self.calib_params.loc[g, (a, 'm')] = (un.uarray(srm, serr) /
        #                                                           un.uarray(meas, merr))[0]
        #                     if not zero_intercept:
        #                         self.calib_params.loc[g, (a, 'c')] = 0

        #                 # This should be obsolete, because no-longer sourcing locator from calib_params index.
        #                 # except KeyError:
        #                 #     # If the calibration is being recalculated, calib_params
        #                 #     # will have t=0 and t=max(uTime) values that are outside
        #                 #     # the srmtabs index.
        #                 #     # If this happens, drop them, and re-fill them at the end.
        #                 #     self.calib_params.drop(g, inplace=True)
        #         else:
        #             ind = idx[a, :, :, :]
        #             meas = self.srmtabs.loc[ind, 'meas_mean']
        #             srm = self.srmtabs.loc[ind, 'srm_mean']
        #             merr = self.srmtabs.loc[ind, 'meas_err']
        #             serr = self.srmtabs.loc[ind, 'srm_err']
        #             sigma = np.sqrt(merr**2 + serr**2)

        #             if len(meas) > 1:
        #                 p, cov = curve_fit(fn, meas, srm, sigma=sigma)
        #                 pe = unc.correlated_values(p, cov)
        #                 self.calib_params.loc[:, (a, 'm')] = pe[0]
        #                 if not zero_intercept:
        #                     self.calib_params.loc[:, (a, 'c')] = pe[1]
        #             else:
        #                 self.calib_params.loc[:, (a, 'm')] = (un.uarray(srm, serr) /
        #                                                       un.uarray(meas, merr))[0]
        #                 if not zero_intercept:
        #                     self.calib_params.loc[:, (a, 'c')] = 0

        #     # if fill:
        #     # fill in uTime=0 and uTime = max cases for interpolation
        #     if self.calib_params.index.min() == 0:
        #         self.calib_params.drop(0, inplace=True)
        #         self.calib_params.drop(self.calib_params.index.max(), inplace=True)
        #     self.calib_params.loc[0, :] = self.calib_params.loc[self.calib_params.index.min(), :]
        #     maxuT = np.max([d.uTime.max() for d in self.data.values()])  # calculate max uTime
        #     self.calib_params.loc[maxuT, :] = self.calib_params.loc[self.calib_params.index.max(), :]
        #     # sort indices for slice access
        #     self.calib_params.sort_index(1, inplace=True)
        #     self.calib_params.sort_index(0, inplace=True)

        #     # calculcate interpolators for applying calibrations
        #     self.calib_ps = Bunch()
        #     for a in analytes:
        #         # TODO: revisit un_interp1d to see whether it plays well with correlated values.
        #         # Possible re-write to deal with covariance matrices?
        #         self.calib_ps[a] = {'m': un_interp1d(self.calib_params.index.values,
        #                                             self.calib_params.loc[:, (a, 'm')].values)}
        #         if not zero_intercept:
        #             self.calib_ps[a]['c'] = un_interp1d(self.calib_params.index.values,
        #                                                 self.calib_params.loc[:, (a, 'c')].values)

        #     with self.pbar.set(total=len(self.data), desc='Applying Calibrations') as prog:
        #         for d in self.data.values():
        #             d.calibrate(self.calib_ps, analytes)
        #             prog.update()

        #     # record SRMs used for plotting
        #     markers = 'osDsv<>PX'  # for future implementation of SRM-specific markers.
        #     if not hasattr(self, 'srms_used'):
        #         self.srms_used = set(srms_used)
        #     else:
        #         self.srms_used.update(srms_used)
        #     self.srm_mdict = {k: markers[i] for i, k in enumerate(self.srms_used)}

        #     self.stages_complete.update(['calibrated'])
        #     self.focus_stage = 'calibrated'

        #     return

    def srm_id_auto(self, srms_used=['NIST610', 'NIST612', 'NIST614'], analytes=None, n_min=10,
                    reload_srm_database=False):
        """
        Function for automarically identifying SRMs using KMeans clustering.

        KMeans is performed on the log of SRM composition, which aids separation
        of relatively similar SRMs within a large compositional range.

        Parameters
        ----------
        srms_used : iterable
            Which SRMs have been used. Must match SRM names
            in SRM database *exactly* (case sensitive!).
        analytes : array_like
            Which analytes to base the identification on. If None,
            all analytes are used (default).
        n_min : int
            The minimum number of data points a SRM measurement
            must contain to be included.
        reload_srm_database : bool
            Whether or not to re-load the SRM database before running the function.
        """
        # TODO: srm_id_plot!
        if isinstance(srms_used, str):
            srms_used = [srms_used]

        # compile measured SRM data
        self.srm_compile_measured(n_min)

        # load SRM database
        self.srm_load_database(srms_used, reload_srm_database)

        if analytes is None:
            analytes = self._srm_id_analytes
        else:
            analytes = [a for a in analytes if a in self._srm_id_analytes]

        # get and scale mean srm values for all analytes
        srmid = self.srmtab.loc[:, idx[analytes, 'mean']]
        _srmid = scale(np.log(srmid))
        srm_labels = srmid.index.values

        # get and scale measured srm values for all analytes
        stdid = self.stdtab.loc[:, idx[analytes, 'mean']]
        _stdid = scale(np.log(stdid))

        # fit KMeans classifier to srm database
        classifier = KMeans(len(srms_used)).fit(_srmid)
        # apply classifier to measured data
        std_classes = classifier.predict(_stdid)

        # get srm names from classes
        std_srm_labels = np.array([srm_labels[np.argwhere(classifier.labels_ == i)][0][0] for i in std_classes])

        self.stdtab.loc[:, 'SRM'] = std_srm_labels
        self.srms_ided = True

        self.srm_build_calib_table()

    def srm_build_calib_table(self):
        """
        Combine SRM database values and identified measured values into a calibration database.
        """
        caltab = self.stdtab.reset_index()
        caltab.set_index(['gTime', 'uTime'], inplace=True)
        levels = ['meas_' + c if c != '' else c for c in caltab.columns.levels[1]]
        caltab.columns.set_levels(levels, 1, inplace=True)

        for a in self.analytes:
            if a == self.internal_standard:
                continue

            caltab.loc[:, (a, 'srm_mean')] = self.srmtab.loc[caltab.SRM, (a, 'mean')].values
            caltab.loc[:, (a, 'srm_err')] = self.srmtab.loc[caltab.SRM, (a, 'err')].values

        self.caltab = caltab.reindex(self.stdtab.columns.levels[0], axis=1, level=0)

    # def srm_id_auto(self, srms_used=['NIST610', 'NIST612', 'NIST614'], n_min=10, reload_srm_database=False):
    #     """
    #     Function for automarically identifying SRMs

    #     Parameters
    #     ----------
    #     srms_used : iterable
    #         Which SRMs have been used. Must match SRM names
    #         in SRM database *exactly* (case sensitive!).
    #     n_min : int
    #         The minimum number of data points a SRM measurement
    #         must contain to be included.
    #     """
    #     if isinstance(srms_used, str):
    #         srms_used = [srms_used]

    #     # get mean and standard deviations of measured standards
    #     self.srm_compile_measured(n_min)
    #     stdtab = self.stdtab.copy()
    #     stdtab.loc[:, 'SRM'] = ''

    #     # load corresponding SRM database
    #     self.srm_load_database(srms_used, reload_srm_database)

    #     # create blank srm table
    #     srm_tab = self.srmdat.loc[:, ['mol_ratio', 'element']].reset_index().pivot(index='SRM', columns='element', values='mol_ratio')

    #     # Auto - ID STDs
    #     # 1. identify elements in measured SRMS with biggest range of values
    #     meas_tab = stdtab.loc[:, (slice(None), 'mean')]  # isolate means of standards
    #     meas_tab.columns = meas_tab.columns.droplevel(1)  # drop 'mean' column names
    #     meas_tab.columns = [re.findall('[A-Za-z]+', a)[0] for a in meas_tab.columns]  # rename to element names
    #     meas_tab = meas_tab.T.groupby(level=0).first().T  # remove duplicate columns

    #     ranges = nominal_values(meas_tab.apply(lambda a: np.ptp(a) / np.nanmean(a), 0))  # calculate relative ranges of all elements
    #     # (used as weights later)

    #     # 2. Work out which standard is which
    #     # normalise all elements between 0-1
    #     def normalise(a):
    #         a = nominal_values(a)
    #         if np.nanmin(a) < np.nanmax(a):
    #             return (a - np.nanmin(a)) / np.nanmax(a - np.nanmin(a))
    #         else:
    #             return np.ones(a.shape)

    #     nmeas = meas_tab.apply(normalise, 0)
    #     nmeas.dropna(1, inplace=True)  # remove elements with NaN values
    #     # nmeas.replace(np.nan, 1, inplace=True)
    #     nsrm_tab = srm_tab.apply(normalise, 0)
    #     nsrm_tab.dropna(1, inplace=True)
    #     # nsrm_tab.replace(np.nan, 1, inplace=True)

    #     for uT, r in nmeas.iterrows():  # for each standard...
    #         idx = np.nansum(((nsrm_tab - r) * ranges)**2, 1)
    #         idx = abs((nsrm_tab - r) * ranges).sum(1)
    #         # calculate the absolute difference between the normalised elemental
    #         # values for each measured SRM and the SRM table. Each element is
    #         # multiplied by the relative range seen in that element (i.e. range / mean
    #         # measuerd value), so that elements with a large difference are given
    #         # more importance in identifying the SRM.
    #         # This produces a table, where wach row contains the difference between
    #         # a known vs. measured SRM. The measured SRM is identified as the SRM that
    #         # has the smallest weighted sum value.
    #         stdtab.loc[uT, 'SRM'] = srm_tab.index[idx == min(idx)].values[0]

    #     # calculate mean time for each SRM
    #     # reset index and sort
    #     stdtab.reset_index(inplace=True)
    #     stdtab.sort_index(1, inplace=True)
    #     # isolate STD and uTime
    #     uT = stdtab.loc[:, ['gTime', 'STD']].set_index('STD')
    #     uT.sort_index(inplace=True)
    #     uTm = uT.groupby(level=0).mean()  # mean uTime for each SRM
    #     # replace uTime values with means
    #     stdtab.set_index(['STD'], inplace=True)
    #     stdtab.loc[:, 'gTime'] = uTm
    #     # reset index
    #     stdtab.reset_index(inplace=True)
    #     stdtab.set_index(['STD', 'SRM', 'gTime'], inplace=True)

    #     # combine to make SRM reference tables
    #     srmtabs = Bunch()
    #     for a in self.analytes:
    #         el = re.findall('[A-Za-z]+', a)[0]

    #         sub = stdtab.loc[:, a]

    #         srmsub = self.srmdat.loc[self.srmdat.element == el, ['mol_ratio', 'mol_ratio_err']]

    #         srmtab = sub.join(srmsub)
    #         srmtab.columns = ['meas_err', 'meas_mean', 'srm_mean', 'srm_err']

    #         srmtabs[a] = srmtab

    #     self.srmtabs = pd.concat(srmtabs).apply(nominal_values).sort_index()
    #     self.srmtabs.dropna(subset=['srm_mean'], inplace=True)
    #     # replace any nan error values with zeros - nans cause problems later.
    #     self.srmtabs.loc[:, ['meas_err', 'srm_err']] = self.srmtabs.loc[:, ['meas_err', 'srm_err']].replace(np.nan, 0)

    #     # remove internal standard from calibration elements
    #     self.srmtabs.drop(self.internal_standard, level=0, inplace=True)

    #     self.srms_ided = True
    #     return

    def srm_compile_measured(self, n_min=10, focus_stage='ratios'):
        """
        Compile mean and standard errors of measured SRMs

        Parameters
        ----------
        n_min : int
            The minimum number of points to consider as a valid measurement.
            Default = 10.
        """
        warns = []
        # compile mean and standard errors of samples
        for s in self.stds:
            s_stdtab = pd.DataFrame(columns=pd.MultiIndex.from_product([s.analytes, ['err', 'mean']]))
            s_stdtab.index.name = 'uTime'

            if not s.n > 0:
                s.stdtab = s_stdtab
                continue

            for n in range(1, s.n + 1):
                ind = s.ns == n
                if sum(ind) >= n_min:
                    for a in s.analytes:
                        aind = ind & ~np.isnan(nominal_values(s.data[focus_stage][a]))
                        s_stdtab.loc[np.nanmean(s.uTime[s.ns == n]),
                                     (a, 'mean')] = np.nanmean(nominal_values(s.data[focus_stage][a][aind]))
                        s_stdtab.loc[np.nanmean(s.uTime[s.ns == n]),
                                     (a, 'err')] = np.nanstd(nominal_values(s.data[focus_stage][a][aind])) / np.sqrt(
                            sum(aind))
                else:
                    warns.append('   Ablation {:} of SRM measurement {:} ({:} points)'.format(n, s.sample, sum(ind)))

            # sort column multiindex
            s_stdtab = s_stdtab.loc[:, s_stdtab.columns.sort_values()]
            # sort row index
            s_stdtab.sort_index(inplace=True)

            # create 'SRM' column for naming SRM
            s_stdtab.loc[:, 'STD'] = s.sample

            s.stdtab = s_stdtab

        if len(warns) > 0:
            print('WARNING: Some SRM ablations have been excluded because they do not contain enough data:')
            print('\n'.join(warns))
            print("To *include* these ablations, reduce the value of n_min (currently {:})".format(n_min))

        # compile them into a table
        stdtab = pd.concat([s.stdtab for s in self.stds]).apply(pd.to_numeric, 1, errors='ignore')
        stdtab = stdtab.reindex(self.analytes_sorted() + ['STD'], level=0, axis=1)

        # identify groups of consecutive SRMs
        ts = stdtab.index.values
        start_times = [s.uTime[0] for s in self.data.values()]

        lastpos = sum(ts[0] > start_times)
        group = [1]
        for t in ts[1:]:
            pos = sum(t > start_times)
            rpos = pos - lastpos
            if rpos <= 1:
                group.append(group[-1])
            else:
                group.append(group[-1] + 1)
            lastpos = pos

        stdtab.loc[:, 'group'] = group
        # calculate centre time for the groups
        stdtab.loc[:, 'gTime'] = np.nan

        for g, d in stdtab.groupby('group'):
            ind = stdtab.group == g
            stdtab.loc[ind, 'gTime'] = stdtab.loc[ind].index.values.mean()

        self.stdtab = stdtab

    def srm_load_database(self, srms_used=None, reload=False):
        if not hasattr(self, 'srmdat') or reload:
            # load SRM info
            srmdat = srms.read_table(self.srmfile)
            srmdat = srmdat.loc[srms_used]
            srmdat.reset_index(inplace=True)
            srmdat.set_index(['SRM', 'Item'], inplace=True)
            # empty columns for mol_ratio and mol_ratio_err
            srmdat.loc[:, 'mol_ratio'] = np.nan
            srmdat.loc[:, 'mol_ratio_err'] = np.nan

            # get element name
            internal_el = get_analyte_name(self.internal_standard)
            # calculate ratios to internal_standard for all elements

            analyte_srm_link = {}
            warns = []
            srmsubs = []

            for srm in srms_used:
                srmsub = srmdat.loc[srm]

                # determine analyte - Item pairs in table
                ad = {}
                for a in self.analytes:
                    # check ig there's an exact match of form [Mass][Element] in srmdat
                    mna = analyte_2_massname(a)
                    if mna in srmsub.index:
                        ad[a] = mna
                    else:
                        # if not, match by element name.
                        item = srmsub.index[srmsub.index.str.contains(get_analyte_name(a))].values
                        if len(item) > 1:
                            item = item[item == get_analyte_name(a)]
                        if len(item) == 1:
                            ad[a] = item[0]
                        else:
                            warns.append(f'   No {a} value for {srm}.')

                analyte_srm_link[srm] = ad

                # find denominator
                denom = srmsub.loc[ad[self.internal_standard]]
                # calculate denominator composition (multiplier to account for stoichiometry,
                # e.g. if internal standard is Na, N will be 2 if measured in SRM as Na2O)
                N = float(decompose_molecule(ad[self.internal_standard])[internal_el])

                # calculate molar ratio
                ind = (srm, list(ad.values()))
                srmdat.loc[ind, 'mol_ratio'] = srmdat.loc[ind, 'mol/g'] / (denom['mol/g'] * N)
                srmdat.loc[ind, 'mol_ratio_err'] = (((srmdat.loc[ind, 'mol/g_err'] / srmdat.loc[ind, 'mol/g']) ** 2 +
                                                     (denom['mol/g_err'] / denom['mol/g'])) ** 0.5 *
                                                    srmdat.loc[ind, 'mol_ratio'])  # propagate uncertainty

            srmdat.dropna(subset=['mol_ratio'], inplace=True)

            # where uncertainties are missing, replace with zeros
            srmdat.loc[srmdat.mol_ratio_err.isnull(), 'mol_ratio_err'] = 0

            # compile stand-alone table of SRM values
            srmtab = pd.DataFrame(index=srms_used, columns=pd.MultiIndex.from_product([self.analytes, ['mean', 'err']]))
            for srm, ad in analyte_srm_link.items():
                for a, k in ad.items():
                    srmtab.loc[srm, (a, 'mean')] = srmdat.loc[(srm, k), 'mol_ratio']
                    srmtab.loc[srm, (a, 'err')] = srmdat.loc[(srm, k), 'mol_ratio_err']

            # record outputs
            self.srmdat = srmdat  # the full SRM table
            self._analyte_srmdat_link = analyte_srm_link  # dict linking analyte names to rows in srmdat
            self.srmtab = srmtab.reindex(self.analytes_sorted(), level=0, axis=1).astype(
                float)  # a summary of relevant mol/mol values only

            # record which analytes have missing CRM data
            means = self.srmtab.loc[:, idx[:, 'mean']]
            means.columns = means.columns.droplevel(1)
            self._analytes_missing_srm = means.columns.values[means.isnull().any()]
            self._srm_id_analytes = means.columns.values[~means.isnull().any()]
            self._calib_analytes = means.columns.values[~means.isnull().all()]

            # Print any warnings
            if len(warns) > 0:
                print('WARNING: Some analytes are not present in the SRM database:')
                print('\n'.join(warns))