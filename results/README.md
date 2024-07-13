<h1> Experimental results and Statistical figures </h1>

<hr>

Experimental results and statistical figures are provided in the current directory.

- In the ```Experiment``` folder, there are CSV files that show the results of the performance degradation experiments based on models obtained from multiple training sessions. The CSV files are also divided according to the percentage of perturbations applied..
- In the ```Statistic``` folder, there are statistical figures for the performance degradation experiments of the multiple models, with CSV files divided according to the percentage of perturbations applied.
- Due to resource constraints, for methods that exceed the GPU’s VRAM when applying XAI techniques, we assumed no performance degradation and used the original accuracy to calculate the statistical metrics.