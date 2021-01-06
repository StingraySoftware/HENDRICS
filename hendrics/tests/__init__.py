def _dummy_par(par, pb=1e20, a1=0.0, f0=1.0):
    with open(par, "w") as fobj:
        print("PSRJ     FAKE_X-1", file=fobj)
        print("RAJ      00:55:01", file=fobj)
        print("DECJ     12:00:40.2", file=fobj)
        print("PEPOCH   560000.0", file=fobj)
        print(f"F0       {f0}", file=fobj)
        print("BINARY   BT", file=fobj)
        print("DM       0", file=fobj)
        print(f"PB       {pb}", file=fobj)
        print(f"A1       {a1}", file=fobj)
        print(f"OM       0.0", file=fobj)
        print("ECC      0.0", file=fobj)
        print("T0       56000", file=fobj)
        print("EPHEM    DE421", file=fobj)
        print("CLK      TT(TAI)", file=fobj)

    return par
