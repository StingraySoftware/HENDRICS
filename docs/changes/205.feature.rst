The Numba functions of HENDRICS are now cached on disk (``cache=True``), so
command-line scripts no longer compile them again at every run. From the second
run on, ``HENzsearch --fast`` is 5 to 10 s faster on the CPU, and about 1.3 s
faster with ``--use-gpu``. Importing ``hendrics.efsearch`` no longer compiles the
FFA helper ``sum_arrays``, saving about 0.2 s for every command that imports it.
