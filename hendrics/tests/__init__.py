def _dummy_par(par):
    with open(par, "a") as fobj:
        print("PEPOCH 560000", file=fobj)
        print("F0 1", file=fobj)
        print("BINARY BT", file=fobj)
        print("PB  1e20", file=fobj)
        print("A1  0", file=fobj)
        print("T0  56000", file=fobj)
        print("EPHEM  DE200", file=fobj)
        print("RAJ  00:55:01", file=fobj)
        print("DECJ 12:00:40.2", file=fobj)
    return par
