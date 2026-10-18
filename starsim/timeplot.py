"""
Plotting dates and durations with Matplotlib: Starsim's dates (`ss.date`, `ss.DateArray`) and durations (`ss.dur`) as axis values, with date ticks
"""
import collections
import datetime as dt
import matplotlib
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

__all__ = ['DateConverter']


class MockAxis():
    """
    Emulator for Axis.get_view_internal() to return converted values

    Matplotlib's date locators (AutoDateLocator, YearLocator, MonthLocator etc.)
    call Axis.get_view_internal() to get the current axis limits. These locators all
    expect the values to be returned in matplotlib dates (i.e., days since 1970-01-01).
    This helper class is bound to the actual axis, and emulates `Axis.get_view_interval()`
    performing the conversion to return the values in the expected units. That way, the
    Locator instances can be used directly without needing any other changes.

    When this class is instantiated, it is bound to an actual Axis instance. The axis
    instance would have our custom date units and values in floating point years. Then
    `MockAxis.get_view_interval()` converts these years to standard matplotlib date values
    before they get passed to the Matplotlib locators.
    """

    def __init__(self, axis):
        self.axis = axis

    def get_view_interval(self):
        """
        Return converted axis limits

        :return: Tuple with (min, max) axis limits in matplotlib date units (days since 1970-01-01)
        """
        ymin, ymax = self.axis.get_view_interval()
        vmin, vmax = matplotlib.dates.date2num(ss.date(ymin)), matplotlib.dates.date2num(ss.date(ymax))
        return vmin, vmax


class FloatYearLocator(matplotlib.ticker.Locator):
    """
    Locator to calculate tick positions for float year axes

    Matplotlib's AutoDateLocator is the standard way to calculating date-based tick
    positions, incorporating a lot of functionality around sensibly setting the tick intervals
    e.g., to whole months. This class wraps AutoDateLocator to allow it to be used
    with float year axis values, by converting the float year values to matplotlib date values
    before calling the locator, and then converting the calculated tick positions back to
    float years before returning them.
    """

    def __init__(self, unit):
        super().__init__()
        self._convert_dates = isinstance(unit, type) and issubclass(unit, ss.date)

        if self._convert_dates:
            self._date_locator = matplotlib.dates.AutoDateLocator(minticks=3) # Store an AutoDateLocator instance to calculate the tick locations
            self._float_locator = None
        else:
            self._date_locator = None
            self._float_locator = matplotlib.ticker.AutoLocator() # Store an AutoLocator to handle DateArrays that contain dur values rather than dates

    def set_axis(self, axis):
        # When binding this locator to an axis, we also need to bind the internal date locator
        # to a MockAxis instance that performs a conversion from our years to matplotlib values.
        # This instance is internally passed from AutoDateLocator to whichever child class instance
        # gets used to calculate the tick locations. The float_locator (for dur values) does not need
        # to be bound to a MockAxis because the axis is already in the raw units (from `to_human()`)
        # and therefore we can bind it to the actual axis directly.
        super().set_axis(axis)
        if self._convert_dates:
            self._date_locator.set_axis(MockAxis(axis))
        else:
            self._float_locator.set_axis(axis)

    def __call__(self):

        # If not converting dates, just directly call the underlying float locator
        if not self._convert_dates:
            return self.tick_values()

        # Get the axis view limits in matplotlib date units
        vmin, vmax = self._date_locator.axis.get_view_interval()

        # Calculate approximate number of ticks that can fit
        if self.axis is not None:
            ax_width = self.axis.axes.bbox.width  # Width in points
            char_width = 10  # Approximate points per character
            label_chars = 6  # Approximate characters per label
            label_width = char_width * label_chars
            self._date_locator.maxticks = collections.defaultdict(lambda: max(self._date_locator.minticks, int(ax_width / label_width)))

        return self.tick_values(vmin, vmax)


    def tick_values(self, *args, **kwargs):
        if self._convert_dates:
            vticks = self._date_locator() # The internal date locator will return the tick positions in matplotlib units
            yticks = [ss.date(matplotlib.dates.num2date(v)).years for v in vticks] # Convert them to float years before returning
            return np.array(yticks)
        else:
            return self._float_locator()


class FloatYearFormatter(sc.ScirisDateFormatter):
    """
    Formatter that renders float years as calendar dates

    This class provides the same formatting as sc.ScirisDateFormatter, but works with
    axis values that are float years, rather than matplotlib date values (days since 1970-01-01)
    """
    def format_data_short(self, value):
        return super().format_data_short(matplotlib.dates.date2num(ss.date(value)))

    def format_ticks(self, values, *args, **kwargs):
        values = np.asarray(values)
        valid = values >= 1  # Year 0 doesn't exist in Python's datetime
        safe_values = np.where(valid, values, 1)  # Replace invalid years to avoid crash
        labels = super().format_ticks(matplotlib.dates.date2num([ss.date(v) for v in safe_values])) # Convert to matplotlib dates, as for format_data_short()
        return ['' if not v else l for v, l in zip(valid, labels)]


class DateConverter(matplotlib.units.ConversionInterface):
    """
    Starsim date conversion interface for matplotlib

    This class gets registered with matplotlib as the converter for Starsim
    date and DateArray types, allowing them to be used directly as axis values. This achieves
    two outcomes

    1. When Starsim date or DateArray values are plotted, the x-axis values are automatically converted
       to float years for plotting. If this class is not used, matplotlib will instead convert them to
       days since 1970-01-01, which is the default behaviour for pd.Timestamp.
    2. If the first plotting command used on an axis has date/DateArray as the x-axis, our standard
       custom date locators and formatters will be used, so that the float years will automatically be
       labelled with dates.

    Together, this means that users can freely plot both Starsim date/DateArray values and
    plain float years on the same axis without needing any conversions, and they'll also automatically
    get date-based ticks without needing to set the locator, even if they are using custom plotting
    code rather than ss.Result.plot().
    """

    @staticmethod
    def axisinfo(unit, axis):
        """
        Specify which locators and formatters should be used if an instance of a registered class is the first one plotted
        """

        # If the unit is a date, use our custom locators and formatters to set the tick locations/labels
        # Otherwise, set them to `None` to use matplotlib's defaults
        if issubclass(unit, ss.date):
            loc = FloatYearLocator(unit)
            fmt = FloatYearFormatter(loc)
        else:
            loc = None
            fmt = None

        axis.set_converter(DateConverter()) # All future quantities plotted on this axis should use our converter
        axis._set_converter = lambda converter: None
        return matplotlib.units.AxisInfo(majloc=loc, majfmt=fmt, label=None)

    @staticmethod
    def default_units(x, axis):
        if isinstance(x, ss.DateArray):
            return ss.dur if (len(x) and x.is_dur) else ss.date
        elif sc.isiterable(x) and not isinstance(x, ss.TimePar): # e.g. a list or pandas Series of durations, so check the first entry
            x = next(iter(x), None)
        if isinstance(x, ss.dur):
            return ss.dur
        else:
            return ss.date

    @staticmethod
    def _convert_single(v):
        """ Convert a single date-like or duration value to a number; pass through numeric values unchanged """
        if isinstance(v, ss.date):
            return v.years
        elif isinstance(v, ss.datedur): # Has no magnitude of its own, so use years
            return v.years
        elif isinstance(v, ss.dur): # Use the magnitude in the duration's own units, e.g. days(100) -> 100
            return v.value
        elif isinstance(v, (pd.Timestamp, dt.date, dt.datetime)):
            return ss.date(v).years
        else:
            return v

    @staticmethod
    def convert(value, unit, axis):
        # Handle the conversion of registered class instances to numerical values
        # that will appear as the x-values for any plotted data - this is the crucial
        # part so that plotting float year data is compatible with date data. Timepars
        # are converted here rather than by float(), which is deliberately disallowed.
        if isinstance(value, np.ndarray) and value.dtype != object: # Already numeric, e.g. float years
            return value
        elif sc.isiterable(value):
            return [DateConverter._convert_single(v) for v in value]
        else:
            return DateConverter._convert_single(value)

# Register the converter for the Starsim date classes
matplotlib.units.registry[ss.date] = DateConverter()
matplotlib.units.registry[ss.DateArray] = DateConverter()
matplotlib.units.registry[ss.dur] = DateConverter() # Durations are converted here rather than by float(), which is deliberately disallowed
