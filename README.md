# ParallelComputing-MergeSort-OpenMP

**Polimi Class-Parallel Computing Challenge1:**
`MergeSort Acceleration By OpenMP` from professor `Serena Curze`. Here are some goals:

- Follow the hints in the code
- Use tasks when required, add synchronization
- Create a parallel region with a single creator, multiple executors
-Add a cut-off mechanism
- Test different configurations:
  - Here is the link to the [Challenge1Report](Challenge1-YanlongWang.pdf) about some testing.

**Achieved:**
Use 16 threads to sort about 4GB data, with parameters like depth and cutoffs to control parallel scale. The speedup is 3x compared to the serial version.
![alt text](<Screenshot Of Resource Monitor.png>)
