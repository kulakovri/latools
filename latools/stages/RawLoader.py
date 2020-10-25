import json
import os
import re
import time
import warnings
import textwrap

import pkg_resources as pkgrs
from ..D_obj import D
from ..helpers.helpers import (get_date, Bunch)
from ..helpers.config import read_configuration, config_locator
from ..helpers.stat_fns import *
from ..helpers.progressbars import progressbar
from latools import analyse


class RawLoader(analyse):

    def __init__(self, data_folder, errorhunt=False, config='DEFAULT',
                 dataformat=None, extension='.csv', srm_identifier='STD',
                 cmap=None, time_format=None, internal_standard='Ca43',
                 names='file_names', srm_file=None, pbar=None):
        # initialise log
        params = {k: v for k, v in locals().items() if k not in ['self', 'pbar']}
        self.log = ['__init__ :: args=() kwargs={}'.format(str(params))]

        # assign file paths
        self.folder = os.path.realpath(data_folder)
        self.parent_folder = os.path.dirname(self.folder)
        self.files = np.array([f for f in os.listdir(self.folder)
                               if extension in f])

        # set line length for outputs
        self._line_width = 80

        # make output directories
        self.report_dir = re.sub('//', '/',
                                 os.path.join(self.parent_folder,
                                              os.path.basename(self.folder) + '_reports/'))
        if not os.path.isdir(self.report_dir):
            os.mkdir(self.report_dir)
        self.export_dir = re.sub('//', '/',
                                 os.path.join(self.parent_folder,
                                              os.path.basename(self.folder) + '_export/'))
        if not os.path.isdir(self.export_dir):
            os.mkdir(self.export_dir)

        # load configuration parameters
        self.config = read_configuration(config)

        # print some info about the analysis and setup.
        startmsg = self._fill_line('-') + 'Starting analysis:'
        if srm_file is None or dataformat is None:
            startmsg += '\n  Using {} configuration'.format(self.config['config'])
            if config == 'DEFAULT':
                startmsg += ' (default).'
            else:
                startmsg += '.'
            pretext = '  with'
        else:
            pretext = 'Using'

        if srm_file is not None:
            startmsg += '\n  ' + pretext + ' custom srm_file ({})'.format(srm_file)
        if isinstance(dataformat, str):
            startmsg += '\n  ' + pretext + ' custom dataformat file ({})'.format(dataformat)
        elif isinstance(dataformat, dict):
            startmsg += '\n  ' + pretext + ' custom dataformat dict'
        print(startmsg)

        self._load_srmfile(srm_file)

        self._load_dataformat(dataformat)

        # link up progress bars
        if pbar is None:
            self.pbar = progressbar()
        else:
            self.pbar = pbar

        # load data into list (initialise D objects)
        with self.pbar.set(total=len(self.files), desc='Loading Data') as prog:
            data = [None] * len(self.files)
            for i, f in enumerate(self.files):
                data[i] = (D(os.path.join(self.folder, f),
                             dataformat=self.dataformat,
                             errorhunt=errorhunt,
                             cmap=cmap,
                             internal_standard=internal_standard,
                             name=names))
                prog.update()

        # create universal time scale
        if 'date' in data[0].meta.keys():
            if (time_format is None) and ('time_format' in self.dataformat.keys()):
                time_format = self.dataformat['time_format']

            start_times = []
            for d in data:
                start_times.append(get_date(d.meta['date'], time_format))
            min_time = min(start_times)

            for d, st in zip(data, start_times):
                d.uTime = d.Time + (st - min_time).seconds
        else:
            ts = 0
            for d in data:
                d.uTime = d.Time + ts
                ts += d.Time[-1]
            msg = self._wrap_text(
                "Time not determined from dataformat. Universal time scale " +
                "approximated as continuously measured samples. " +
                "Samples might not be in the right order. "
                "Background correction and calibration may not behave " +
                "as expected.")
            warnings.warn(self._wrap_msg(msg, '*'))

        self.max_time = max([d.uTime.max() for d in data])

        # sort data by uTime
        data.sort(key=lambda d: d.uTime[0])

        # process sample names
        if (names == 'file_names') | (names == 'metadata_names'):
            samples = np.array([s.sample for s in data], dtype=object)  # get all sample names
            # if duplicates, rename them
            usamples, ucounts = np.unique(samples, return_counts=True)
            if usamples.size != samples.size:
                dups = usamples[ucounts > 1]  # identify duplicates
                nreps = ucounts[ucounts > 1]  # identify how many times they repeat
                for d, n in zip(dups, nreps):  # cycle through duplicates
                    new = [d + '_{}'.format(i) for i in range(n)]  # append number to duplicate names
                    ind = samples == d
                    samples[ind] = new  # rename in samples
                    for s, ns in zip([data[i] for i in np.where(ind)[0]], new):
                        s.sample = ns  # rename in D objects
        else:
            samples = np.arange(len(data))  # assign a range of numbers
            for i, s in enumerate(samples):
                data[i].sample = s
        self.samples = samples

        # copy colour map to top level
        self.cmaps = data[0].cmap

        # get analytes
        # TODO: does this preserve the *order* of the analytes?
        all_analytes = set()
        extras = set()
        for d in data:
            all_analytes.update(d.analytes)
            extras.update(all_analytes.symmetric_difference(d.analytes))
        self.analytes = all_analytes.difference(extras)
        mismatch = []
        if self.analytes != all_analytes:
            smax = 0
            for d in data:
                if d.analytes != self.analytes:
                    mismatch.append((d.sample, d.analytes.difference(self.analytes)))
                    if len(d.sample) > smax:
                        smax = len(d.sample)
            msg = (self._fill_line('*') +
                   'All data files do not contain the same analytes.\n' +
                   'Only analytes present in all files will be processed.\n' +
                   'In the following files, these analytes will be excluded:\n')
            for s, a in mismatch:
                msg += ('  {0: <' + '{:}'.format(smax + 2) + '}:  ').format(s) + str(a) + '\n'
            msg += self._fill_line('*')
            warnings.warn(msg)

        if len(self.analytes) == 0:
            raise ValueError(
                'No analyte names identified. Please check the \ncolumn_id > pattern ReGeX in your dataformat file.')

        if internal_standard in self.analytes:
            self.internal_standard = internal_standard
        else:
            raise ValueError('The internal standard ({}) is not amongst the '.format(internal_standard) +
                             'analytes in\nyour data files. Please make sure it is specified correctly.')
        self.minimal_analytes = set([internal_standard])

        # keep record of which stages of processing have been performed
        self.stages_complete = set(['rawdata'])

        # From this point on, data stored in dicts
        self.data = Bunch(zip(self.samples, data))

        # remove mismatch analytes - QUICK-FIX - SHOULD BE DONE HIGHER UP?
        for s, a in mismatch:
            self.data[s].analytes = self.data[s].analytes.difference(a)

        # get SRM info
        self.srm_identifier = srm_identifier
        self.stds = []  # make this a dict
        _ = [self.stds.append(s) for s in self.data.values()
             if self.srm_identifier in s.sample]
        self.srms_ided = False

        # set up focus_stage recording
        self.focus_stage = 'rawdata'
        self.focus = Bunch()

        # set up subsets
        self._has_subsets = False
        self._subset_names = []
        self.subsets = Bunch()
        self.subsets['All_Analyses'] = self.samples
        self.subsets[self.srm_identifier] = [s for s in self.samples if self.srm_identifier in s]
        self.subsets['All_Samples'] = [s for s in self.samples if self.srm_identifier not in s]
        self.subsets['not_in_set'] = self.subsets['All_Samples'].copy()

        # remove any analytes for which all counts are zero
        # self.get_focus()
        # for a in self.analytes:
        #     if np.nanmean(self.focus[a] == 0):
        #         self.analytes.remove(a)
        #         warnings.warn('{} contains no data - removed from analytes')

        # initialise classifiers
        self.classifiers = Bunch()

        # report
        print(('Loading Data:\n  {:d} Data Files Loaded: {:d} standards, {:d} '
               'samples').format(len(self.data),
                                 len(self.stds),
                                 len(self.data) - len(self.stds)))
        astr = self._wrap_text('Analytes: ' + ' '.join(self.analytes_sorted()))
        print(astr)
        print('  Internal Standard: {}'.format(self.internal_standard))

    def _fill_line(self, char, newline=True):
        """Generate a full line of given character"""
        if newline:
            return char * self._line_width + '\n'
        else:
            return char * self._line_width

    def _wrap_text(self, text):
        """Splits text over multiple lines to fit within self._line_width"""
        return '\n'.join(textwrap.wrap(text, width=self._line_width,
                                       break_long_words=False))

    def _wrap_msg(self, msg, char):
        return self._fill_line(char) + msg + '\n' + self._fill_line(char, False)

    def _load_dataformat(self, dataformat):
        """
        Load in dataformat.

        Check dataformat file exists, and store it in a class attribute.
        If dataformat is not provided during initialisation, assign it
        fom configuration file
        """
        if dataformat is None:
            if os.path.exists(self.config['dataformat']):
                dataformat = self.config['dataformat']
            elif os.path.exists(pkgrs.resource_filename('latools',
                                                        self.config['dataformat'])):
                dataformat = pkgrs.resource_filename('latools',
                                                     self.config['dataformat'])
            else:
                config_file = config_locator()
                raise ValueError(('The dataformat file specified in the ' +
                                  self.config['config'] + ' configuration cannot be found.\n'
                                                          'Please make sure the file exists, and that'
                                                          'the path in the config file is correct.\n'
                                                          'Your configurations can be found here:'
                                                          '    {}\n'.format(config_file)))
            self.dataformat_file = dataformat
        else:
            self.dataformat_file = 'None: dict provided'

        # if it's a string, check the file exists and import it.
        if isinstance(dataformat, str):
            if os.path.exists(dataformat):
                # self.dataformat = eval(open(dataformat).read())
                self.dataformat = json.load(open(dataformat))
            else:
                warnings.warn(("The dataformat file (" + dataformat +
                               ") cannot be found.\nPlease make sure the file "
                               "exists, and that the path is correct.\n\nFile "
                               "Path: " + dataformat))

        # if it's a dict, just assign it straight away.
        elif isinstance(dataformat, dict):
            self.dataformat = dataformat

    def _load_srmfile(self, srm_file):
        """
        Check srmfile exists, and store it in a class attribute.
        """
        if srm_file is not None:
            if os.path.exists(srm_file):
                self.srmfile = srm_file
            else:
                raise ValueError(('Cannot find the specified SRM file:\n   ' +
                                  srm_file +
                                  'Please check that the file location is correct.'))
        else:
            if os.path.exists(self.config['srmfile']):
                self.srmfile = self.config['srmfile']
            elif os.path.exists(pkgrs.resource_filename('latools',
                                                        self.config['srmfile'])):
                self.srmfile = pkgrs.resource_filename('latools',
                                                       self.config['srmfile'])
            else:
                config_file = config_locator()
                raise ValueError(('The SRM file specified in the ' + self.config['config'] +
                                  ' configuration cannot be found.\n'
                                  'Please make sure the file exists, and that the '
                                  'path in the config file is correct.\n'
                                  'Your configurations can be found here:'
                                  '    {}\n'.format(config_file)))

    def _get_samples(self, subset=None):
        """
        Helper function to get sample names from subset.

        Parameters
        ----------
        subset : str
            Subset name. If None, returns all samples.

        Returns
        -------
        List of sample names
        """
        if subset is None:
            samples = self.subsets['All_Samples']
        else:
            try:
                samples = self.subsets[subset]
            except KeyError:
                raise KeyError(("Subset '{:s}' does not ".format(subset) +
                                "exist.\nUse 'make_subset' to create a" +
                                "subset."))
        return samples

    def _log_header(self):
        return ['# LATOOLS analysis log saved at {}'.format(time.strftime('%Y:%m:%d %H:%M:%S')),
                'data_folder :: {}'.format(self.folder),
                '# Analysis Log Start: \n'
                ]
