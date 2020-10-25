from latools import analyse


class RatioCalculator(analyse):
    def __init__(self, bkg_sub, internal_standard=None, analytes=None):
        self.move_attributes_from_previous_stage(bkg_sub)
        if 'bkgsub' not in self.stages_complete:
            raise RuntimeError('Cannot calculate ratios before background subtraction.')

        if analytes is None:
            analytes = self.analytes
        elif isinstance(analytes, str):
            analytes = [analytes]

        if internal_standard is not None:
            self.internal_standard = internal_standard
            self.minimal_analytes.update([internal_standard])

        with self.pbar.set(total=len(self.data), desc='Ratio Calculation') as prog:
            for s in self.data.values():
                s.ratio(internal_standard=self.internal_standard, analytes=analytes)
                prog.update()

        self.stages_complete.update(['ratios'])
        self.focus_stage = 'ratios'
        return
