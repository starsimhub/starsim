# CK feedback 2

## Tests

The Starsim test suite used to take 30 seconds to run and now it takes 8 minutes. I also saw the number of tests has ballooned from like 150 to 650. Get runtime back down to <1 min, number of tests down to <300, and move everything else to tests/devtests, and do not run the devtests by default.

## Compilation

I notice first runs are often still slow. I assume this is because of functions needing to be compiled. This is fine, but please print "Compiling <X>, estimated time: <min>-<max> s"
