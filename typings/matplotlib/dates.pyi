from collections.abc import Sequence
from typing import Any, Literal

from matplotlib import ticker, units
from numpy.typing import NDArray

"""
Matplotlib provides sophisticated date plotting capabilities, standing on the
shoulders of python :mod:`datetime` and the add-on module dateutil_.

By default, Matplotlib uses the units machinery described in
`~matplotlib.units` to convert `datetime.datetime`, and `numpy.datetime64`
objects when plotted on an x- or y-axis. The user does not
need to do anything for dates to be formatted, but dates often have strict
formatting needs, so this module provides many tick locators and formatters.
A basic example using `numpy.datetime64` is::

    import numpy as np

    times = np.arange(np.datetime64('2001-01-02'),
                      np.datetime64('2002-02-03'), np.timedelta64(75, 'm'))
    y = np.random.randn(len(times))

    fig, ax = plt.subplots()
    ax.plot(times, y)

.. seealso::

    - :doc:`/gallery/text_labels_and_annotations/date`
    - :doc:`/gallery/ticks/date_concise_formatter`
    - :doc:`/gallery/ticks/date_demo_convert`

.. _date-format:

Matplotlib date format
----------------------

Matplotlib represents dates using floating point numbers specifying the number
of days since a default epoch of 1970-01-01 UTC; for example,
1970-01-01, 06:00 is the floating point number 0.25. The formatters and
locators require the use of `datetime.datetime` objects, so only dates between
year 0001 and 9999 can be represented.  Microsecond precision
is achievable for (approximately) 70 years on either side of the epoch, and
20 microseconds for the rest of the allowable range of dates (year 0001 to
9999). The epoch can be changed at import time via `.dates.set_epoch` or
:rc:`date.epoch` to other dates if necessary; see
:doc:`/gallery/ticks/date_precision_and_epochs` for a discussion.

.. note::

   Before Matplotlib 3.3, the epoch was 0000-12-31 which lost modern
   microsecond precision and also made the default axis limit of 0 an invalid
   datetime.  In 3.3 the epoch was changed as above.  To convert old
   ordinal floats to the new epoch, users can do::

     new_ordinal = old_ordinal + mdates.date2num(np.datetime64('0000-12-31'))


There are a number of helper functions to convert between :mod:`datetime`
objects and Matplotlib dates:

.. currentmodule:: matplotlib.dates

.. autosummary::
   :nosignatures:

   datestr2num
   date2num
   num2date
   num2timedelta
   drange
   set_epoch
   get_epoch

.. note::

   Like Python's `datetime.datetime`, Matplotlib uses the Gregorian calendar
   for all conversions between dates and floating point numbers. This practice
   is not universal, and calendar differences can cause confusing
   differences between what Python and Matplotlib give as the number of days
   since 0001-01-01 and what other software and databases yield.  For
   example, the US Naval Observatory uses a calendar that switches
   from Julian to Gregorian in October, 1582.  Hence, using their
   calculator, the number of days between 0001-01-01 and 2006-04-01 is
   732403, whereas using the Gregorian calendar via the datetime
   module we find::

     In [1]: date(2006, 4, 1).toordinal() - date(1, 1, 1).toordinal()
     Out[1]: 732401

All the Matplotlib date converters, locators and formatters are timezone aware.
If no explicit timezone is provided, :rc:`timezone` is assumed, provided as a
string.  If you want to use a different timezone, pass the *tz* keyword
argument of `num2date` to any date tick locators or formatters you create. This
can be either a `datetime.tzinfo` instance or a string with the timezone name
that can be parsed by `~dateutil.tz.gettz`.

A wide range of specific and general purpose date tick locators and
formatters are provided in this module.  See
:mod:`matplotlib.ticker` for general information on tick locators
and formatters.  These are described below.

The dateutil_ module provides additional code to handle date ticking, making it
easy to place ticks on any kinds of dates.  See examples below.

.. _dateutil: https://dateutil.readthedocs.io

.. _date-locators:

Date tick locators
------------------

Most of the date tick locators can locate single or multiple ticks. For example::

    # import constants for the days of the week
    from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU

    # tick on Mondays every week
    loc = WeekdayLocator(byweekday=MO, tz=tz)

    # tick on Mondays and Saturdays
    loc = WeekdayLocator(byweekday=(MO, SA))

In addition, most of the constructors take an interval argument::

    # tick on Mondays every second week
    loc = WeekdayLocator(byweekday=MO, interval=2)

The rrule locator allows completely general date ticking::

    # tick every 5th easter
    rule = rrulewrapper(YEARLY, byeaster=1, interval=5)
    loc = RRuleLocator(rule)

The available date tick locators are:

* `MicrosecondLocator`: Locate microseconds.

* `SecondLocator`: Locate seconds.

* `MinuteLocator`: Locate minutes.

* `HourLocator`: Locate hours.

* `DayLocator`: Locate specified days of the month.

* `WeekdayLocator`: Locate days of the week, e.g., MO, TU.

* `MonthLocator`: Locate months, e.g., 7 for July.

* `YearLocator`: Locate years that are multiples of base.

* `RRuleLocator`: Locate using a `rrulewrapper`.
  `rrulewrapper` is a simple wrapper around dateutil_'s `dateutil.rrule`
  which allow almost arbitrary date tick specifications.
  See :doc:`rrule example </gallery/ticks/date_demo_rrule>`.

* `AutoDateLocator`: On autoscale, this class picks the best `DateLocator`
  (e.g., `RRuleLocator`) to set the view limits and the tick locations.  If
  called with ``interval_multiples=True`` it will make ticks line up with
  sensible multiples of the tick intervals.  For example, if the interval is
  4 hours, it will pick hours 0, 4, 8, etc. as ticks.  This behaviour is not
  guaranteed by default.

.. _date-formatters:

Date formatters
---------------

The available date formatters are:

* `AutoDateFormatter`: attempts to figure out the best format to use.  This is
  most useful when used with the `AutoDateLocator`.

* `ConciseDateFormatter`: also attempts to figure out the best format to use,
  and to make the format as compact as possible while still having complete
  date information.  This is most useful when used with the `AutoDateLocator`.

* `DateFormatter`: use `~datetime.datetime.strftime` format strings.
"""
__all__ = (
    "DAILY",
    "FR",
    "HOURLY",
    "MICROSECONDLY",
    "MINUTELY",
    "MO",
    "MONTHLY",
    "SA",
    "SECONDLY",
    "SU",
    "TH",
    "TU",
    "WE",
    "WEEKLY",
    "YEARLY",
    "AutoDateFormatter",
    "AutoDateLocator",
    "ConciseDateConverter",
    "ConciseDateFormatter",
    "DateConverter",
    "DateFormatter",
    "DateLocator",
    "DayLocator",
    "HourLocator",
    "MicrosecondLocator",
    "MinuteLocator",
    "MonthLocator",
    "RRuleLocator",
    "SecondLocator",
    "WeekdayLocator",
    "YearLocator",
    "date2num",
    "datestr2num",
    "drange",
    "get_epoch",
    "num2date",
    "num2timedelta",
    "relativedelta",
    "rrule",
    "rrulewrapper",
    "set_epoch",
)
_log = ...
UTC = ...
EPOCH_OFFSET = ...
MICROSECONDLY = ...
HOURS_PER_DAY = ...
MIN_PER_HOUR = ...
SEC_PER_MIN = ...
MONTHS_PER_YEAR = ...
DAYS_PER_WEEK = ...
DAYS_PER_MONTH = ...
DAYS_PER_YEAR = ...
MINUTES_PER_DAY = ...
SEC_PER_HOUR = ...
SEC_PER_DAY = ...
SEC_PER_WEEK = ...
MUSECONDS_PER_DAY = ...
WEEKDAYS = ...
_epoch = ...

def set_epoch(epoch) -> None: ...
def get_epoch(): ...

_from_ordinalf_np_vectorized = ...
_dateutil_parser_parse_np_vectorized = ...

def datestr2num(d, default=...) -> NDArray[Any] | NDArray[float64] | Any: ...
def date2num(d) -> NDArray[Any] | NDArray[float64] | Any: ...
def num2date(x, tz=...) -> Incomplete: ...

_ordinalf_to_timedelta_np_vectorized = ...

def num2timedelta(x) -> Incomplete: ...
def drange(dstart, dend, delta) -> NDArray[float64]: ...

class DateFormatter(ticker.Formatter):
    def __init__(self, fmt, tz=..., *, usetex=...) -> None: ...
    def __call__(self, x, pos=...) -> str | Incomplete: ...
    def set_tzinfo(self, tz) -> None: ...

class ConciseDateFormatter(ticker.Formatter):
    def __init__(
        self,
        locator,
        tz=...,
        formats=...,
        offset_formats=...,
        zero_formats=...,
        show_offset=...,
        *,
        usetex=...,
    ) -> None: ...
    def __call__(self, x, pos=...) -> str | Incomplete: ...
    def format_ticks(self, values) -> list[str]: ...
    def get_offset(self) -> str | Incomplete: ...
    def format_data_short(self, value) -> Incomplete: ...

class AutoDateFormatter(ticker.Formatter):
    def __init__(self, locator, tz=..., defaultfmt=..., *, usetex=...) -> None: ...
    def __call__(self, x, pos=...) -> str | Incomplete | object: ...

class rrulewrapper:
    def __init__(self, freq, tzinfo=..., **kwargs: Any) -> None: ...
    def set(self, **kwargs: Any) -> None: ...
    def __getattr__(self, name) -> _Wrapped[..., Any, ..., list[Any]]: ...
    def __setstate__(self, state) -> None: ...

class DateLocator(ticker.Locator):
    hms0d = ...
    def __init__(self, tz=...) -> None: ...
    def set_tzinfo(self, tz) -> None: ...
    def datalim_to_dt(self) -> tuple[Incomplete, Incomplete]: ...
    def viewlim_to_dt(self) -> tuple[Incomplete, Incomplete]: ...
    def nonsingular(
        self,
        vmin,
        vmax,
    ) -> (
        tuple[
            NDArray[Any] | NDArray[float64] | Any,
            NDArray[Any] | NDArray[float64] | Any,
        ]
        | tuple[Any, Any]
    ): ...

class RRuleLocator(DateLocator):
    def __init__(self, o, tz=...) -> None: ...
    def __call__(
        self,
    ) -> list[Any] | NDArray[Any] | NDArray[float64] | Any | Sequence[float]: ...
    def tick_values(
        self,
        vmin,
        vmax,
    ) -> NDArray[Any] | NDArray[float64] | Any | Sequence[float]: ...
    @staticmethod
    def get_unit_generic(freq) -> float | Literal[-1]: ...

class AutoDateLocator(DateLocator):
    def __init__(
        self,
        tz=...,
        minticks=...,
        maxticks=...,
        interval_multiples=...,
    ) -> None: ...
    def __call__(
        self,
    ) -> list[Any] | NDArray[Any] | NDArray[float64] | Any | Sequence[float]: ...
    def tick_values(
        self,
        vmin,
        vmax,
    ) -> NDArray[Any] | NDArray[float64] | Any | Sequence[float]: ...
    def nonsingular(
        self,
        vmin,
        vmax,
    ) -> (
        tuple[
            NDArray[Any] | NDArray[float64] | Any,
            NDArray[Any] | NDArray[float64] | Any,
        ]
        | tuple[Any, Any]
    ): ...
    def get_locator(
        self,
        dmin,
        dmax,
    ) -> YearLocator | RRuleLocator | MicrosecondLocator: ...

class YearLocator(RRuleLocator):
    def __init__(self, base=..., month=..., day=..., tz=...) -> None: ...

class MonthLocator(RRuleLocator):
    def __init__(self, bymonth=..., bymonthday=..., interval=..., tz=...) -> None: ...

class WeekdayLocator(RRuleLocator):
    def __init__(self, byweekday=..., interval=..., tz=...) -> None: ...

class DayLocator(RRuleLocator):
    def __init__(self, bymonthday=..., interval=..., tz=...) -> None: ...

class HourLocator(RRuleLocator):
    def __init__(self, byhour=..., interval=..., tz=...) -> None: ...

class MinuteLocator(RRuleLocator):
    def __init__(self, byminute=..., interval=..., tz=...) -> None: ...

class SecondLocator(RRuleLocator):
    def __init__(self, bysecond=..., interval=..., tz=...) -> None: ...

class MicrosecondLocator(DateLocator):
    def __init__(self, interval=..., tz=...) -> None: ...
    def set_axis(self, axis) -> None: ...
    def __call__(self) -> list[Any]: ...
    def tick_values(self, vmin, vmax): ...

class DateConverter(units.ConversionInterface):
    def __init__(self, *, interval_multiples=...) -> None: ...
    def axisinfo(self, unit, axis) -> AxisInfo: ...
    @staticmethod
    def convert(value, unit, axis) -> NDArray[Any] | NDArray[float64] | Any: ...
    @staticmethod
    def default_units(x, axis) -> None: ...

class ConciseDateConverter(DateConverter):
    def __init__(
        self,
        formats=...,
        zero_formats=...,
        offset_formats=...,
        show_offset=...,
        *,
        interval_multiples=...,
    ) -> None: ...
    def axisinfo(self, unit, axis) -> AxisInfo: ...

class _SwitchableDateConverter:
    def axisinfo(self, *args, **kwargs: Any) -> AxisInfo: ...
    def default_units(self, *args, **kwargs: Any) -> None: ...
    def convert(
        self,
        *args,
        **kwargs: Any,
    ) -> NDArray[Any] | NDArray[float64] | Any: ...
