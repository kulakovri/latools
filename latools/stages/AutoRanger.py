import warnings
from latools import analyse


class AutoRanger(analyse):
    def __init__(self, despiked, analyte='total_counts', gwin=5, swin=3, win=20,
                  on_mult=[1., 1.5], off_mult=[1.5, 1],
                  transform='log', ploterrs=True, focus_stage='despiked'):
        self._move_attributes_from_previous_stage(despiked)
        if focus_stage == 'despiked':
            if 'despiked' not in self.stages_complete:
                focus_stage = 'rawdata'

        if analyte is None:
            analyte = self.internal_standard
        elif analyte in self.analytes:
            self.minimal_analytes.update([analyte])

        fails = {}  # list for catching failures.
        with self.pbar.set(total=len(self.data), desc='AutoRange') as prog:
            for s, d in self.data.items():
                f = d.autorange(analyte=analyte, gwin=gwin, swin=swin, win=win,
                                on_mult=on_mult, off_mult=off_mult,
                                ploterrs=ploterrs, transform=transform)
                if f is not None:
                    fails[s] = f
                prog.update()  # advance progress bar
        # handle failures
        if len(fails) > 0:
            wstr = ('\n\n' + '*' * 41 + '\n' +
                    '                 WARNING\n' + '*' * 41 + '\n' +
                    'Autorange failed for some samples:\n')

            kwidth = max([len(k) for k in fails.keys()]) + 1
            fstr = '  {:' + '{}'.format(kwidth) + 's}: '
            for k in sorted(fails.keys()):
                wstr += fstr.format(k) + ', '.join(['{:.1f}'.format(f) for f in fails[k][-1]]) + '\n'

            wstr += ('\n*** THIS IS NOT NECESSARILY A PROBLEM ***\n' +
                     'But please check the plots below to make\n' +
                     'sure they look OK. Failures are marked by\n' +
                     'dashed vertical red lines.\n\n' +
                     'To examine an autorange failure in more\n' +
                     'detail, use the `autorange_plot` method\n' +
                     'of the failing data object, e.g.:\n' +
                     "dat.data['Sample'].autorange_plot(params)\n" +
                     '*' * 41 + '\n')
            warnings.warn(wstr)

        self.stages_complete.update(['autorange'])
        return
