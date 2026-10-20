Whole inputs and filtered outputs from unchanged apply_minimal_filtering.
Pinned reference definitions SHA256:
173eef407c6dd75d96e721c1d781dd7618d05ba26331ab8eb87a6a49f4b6c4ef
Source execution: CPU x86_64, NumPy 2.5.3, SciPy 1.16.3.
Two complete float64 cases: a 100 by 3 ramp and the ingestion witness's
100 by 3 large-offset signal. No native outputs enter the record.
The offset output reproduces every value of the unchanged ingestion witness
after the original standardization and window extraction.
