Experimental design
---------------------
Designing a deep mutational scanning experiment to be analyzed by ``Polyclonal`` requires making a variety of choices, such as the library size, mutation rate, and set of concentrations to use.

Here are some simulations that test how the overall accuracy of the inferences depend on these experimental choices.
Note that the conclusions of these simulations are conditioned on the parameters used in the simulation: three epitopes with the chosen activities, and a relatively short protein.
For instance, a larger library size could be helpful for a larger protein.

Nonetheless, we come away with the following high-level suggestions:

 - For a library of 1,923 possible single amino-acid mutations, *at least* 10,000 variants in the library are needed, and performance is better when it's in the range of 20,000 to 30,000 variants. Larger libraries might be useful when there are more possible mutations.

 - The average per-variant mutation rate should be *at least* two mutations (assuming a Poisson distribution of mutations per variant), and performance may be slightly better if the mutation rate is closer to three.

 - If you just use one concentration of serum, it should be around the IC99 to IC99.9. However, you will get better results if you use three sera concentrations, with one around the IC99 to IC99.9, one ~4-fold higher, and one ~4-fold lower.

Here are notebooks that perform the analyses supporting these general recommendations:

.. toctree::
   :maxdepth: 1

   library_size
   mutation_rate
   concentration_set
