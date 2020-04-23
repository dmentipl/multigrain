Compiling Phantom for dustyshock
================================

First jump into tmux.

```bash
tmux
```

Checkout, patch, compile Phantom.

```bash
bash code/initial-conditions/dustyshock-compile-phantom.sh
```

Set up and run calculations.

```bash
# N=1, i.e. one dust species
bash code/initial-conditions/dustyshock-compile-phantom.sh 'N=1'
```

```bash
# N=3, i.e. one dust species
bash code/initial-conditions/dustyshock-compile-phantom.sh 'N=3'
```
