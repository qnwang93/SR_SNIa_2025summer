To-do list

Start working on a few best targets/simulated sample and test:
- Is it robust against:
  - Noise
    - Can SR pick out the noise term?
  - Data gap
- Which approach works better:
  - Fitting the whole light curve to peak, or
  - Do a cutoff at certain time/flux?
    - Can SR find the optimal cutoff?
- optional: If we add flux error as an extra input, can SR automatically develops error cut? 
For next steps:
- Can we find single analytical function for multiple dataset simultaneously
  - Achievable by PhySO. Unclear whether PySR can do it.
  - Can SR serve as a classifier between SNe Ia with single- and double-component rise?
