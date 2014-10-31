# MaLTPyNT #

These tools are a stripped-down version of a huge and messy library of codes I've developed to do timing with NuSTAR and other satellites.

## Rough way to use them ##

Starting from event files A.evt and B.evt, we will first save them in MaLTPyNT format (a pickle file basically)
```
python mp_read_events.py *.evt
```
This will create new files with a _ev.p extension

TBC