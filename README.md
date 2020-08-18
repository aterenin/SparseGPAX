# SparseGPX

┌───────────────────┐\
│⢰⢦⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡆⠀⠀⢀⣴⣶⢄⠀⠀⠀⠀⠀⠀⠀⠀│\
│⢨⣿⣷⣻⣵⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⣠⣾⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀│\
│⠨⡿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⣼⣿⣿⠿⣿⣾⣿⡀⠀⠀⠀⠀⠀⠀│\
│⠸⠋⠑⢿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⡟⠉⠺⣻⣿⣧⠀⠀⠀⠀⠀⠀│\
│⠀⠀⠀⠀⠹⣿⣿⡀⠀⠀⠀⠀⠀⠀⣼⣿⣿⠝⠀⠀⠀⠈⢿⣿⡆⠀⠀⠀⠀⠀│\
│⠉⠉⠉⠉⠉⢻⣿⣯⠉⠉⠉⠉⠉⣹⣿⣿⠏⠉⠉⠉⠉⠉⠉⣿⣿⣍⡉⠉⠉⠉│\
│⠀⠀⠀⠀⠀⠀⢿⣿⡆⠀⠀⢀⣮⣿⣿⡏⠀⠀⠀⠀⠀⠀⠀⠸⣿⣏⢟⡄⡜⠀│\
│⠀⠀⠀⠀⠀⠀⠈⢿⣿⣦⣶⣫⣿⣿⡏⡇⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⣯⣫⣦⠂│\
│⠀⠀⠀⠀⠀⠀⠀⠘⢿⣿⣿⣿⡿⡟⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⡄│\
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠾⠿⠏⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⢿⠿⠀│\
│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⠀│\
└───────────────────┘

An *ultra-lightweight* JAX/Haiku implementation of sparse Gaussian processes trained via sparse variational inference.

It supports models of the form

```
(f | u)(.) = f(.) + K_{(.)z} (K_{zz} + \Lambda)^{-1} (\mu - f(z) - \epsilon)
```