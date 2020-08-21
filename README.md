# SparseGPAX

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

An *ultra-lightweight* JAX/Haiku implementation of sparse Gaussian processes.

It supports models of the form
```
(f | u)(.) = f(.) + K_{(.)z} (K_{zz} + \Lambda)^{-1} (\mu - f(z) - \epsilon)
```
trained using doubly stochastic sparse variational inference via pathwise sampling.