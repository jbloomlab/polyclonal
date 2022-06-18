---
title: 'polyclonal: A Python package for modeling viral escape from polyclonal antibodies'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
bibliography: paper.bib
---

# Summary

Viral evolution is driven by mutations that escape antibody recognition. However, multiple mutations are often required for viruses to fully escape antibodies in sera, which are polyclonal and can target several distinct epitopes on a viral protein. Understanding the mechanistic relationship between viral protein sequence and polyclonal antibody escape is critical for interpreting viral adaptation to host immunity. Here we introduce [polyclonal](https://github.com/jbloomlab/polyclonal), a Python package for fitting biophysical models of polyclonal antibody escape to data from deep mutational scanning of multiply-mutated viral protein variants. This model can reveal the locations of epitopes that are recognized by antibodies and infer the specific mutations that impede their recognition, providing a detailed picture on how antibody immunity is eroded.

# Statement of need

Deep mutational scanning (DMS) is a method that leverages next-generation sequencing to assay the effects of large numbers of protein variants in multiplex ([Fowler et al. 2014](https://www.nature.com/articles/nmeth.3027)). Importantly, DMS has been applied to measure the effects of thousands of single mutations to viral proteins on antibody escape ([Lee et al. 2020](https://elifesciences.org/articles/49324)). However, the next frontier of these experiments revolves around variants with multiple mutations. Unlike in single-mutant DMS studies, the presence of multiple mutations presents an analytical challenge. This is because mutations exhibit epistasis, and there is a need for computational methods that can be fit on multi-mutant DMS data to accurately predict the consequence of multiple mutations on antibody escape.

In addition, researchers in the field often desire models that provide biologically interpretable parameters explaining how individual mutations contribute to antibody escape. One example is the *global epistasis model* ([Otwinowski et al. 2018](https://www.pnas.org/doi/10.1073/pnas.1804015115)). This model assumes that single mutations act additively on an underlying latent phenotype—and this latent phenotype is non-linearly related to an observed phenotype. For example, mutations contribute additively to the binding free energy of an antibody-epitope interaction, but this binding free energy is related to the measured fraction of escape variants in a non-linear fashion. While global epistasis models are useful for deconvolving the effects of individual mutations on escape from monoclonal antibodies, it becomes less practical in the context of sera. In the latter case, it is less appropriate to model the effects of mutations on a single latent phenotype. Instead, it is more insightful to deconvolve the effects of mutations on multiple latent phenotypes, which correspond to the epitopes at which polyclonal antibodies bind. 

[polyclonal](https://github.com/jbloomlab/polyclonal) is a Python package that fits a biophysical model with multiple latent phenotypes. These latent phenotypes represent the particular epitopes targeted by antibodies in sera, and the model parameters describe the contributions of individual mutations to antibody escape at each epitope. Users can also use the model to predict the antibody escape probability of unseen, multiply-mutated variants. Thus, this package fulfills the need for a predictive and interpretable model of polyclonal antibody escape. 

# Implementation

## Biophysical model

Consider a viral protein bound by polyclonal antibodies, such as might be found in sera.
We want to determine the contribution of each mutation to escaping these polyclonal antibodies, being cognizant of the fact that different antibodies target different epitopes.

The actual experimental measurable is as follows: at each concentration $c$ of the antibody mixture, we measure $p_v\left(c\right)$, which is the fraction of all variants $v$ of the viral protein that escape binding or neutralization by all antibodies in the mix.

We assume that antibodies in the mix can bind to one of $E$ epitopes on the protein.
Let $U_e\left(v,c\right)$ be the fraction of the time that epitope $e$ is not bound on variant $v$ when the mix is at concentration $c$.
Then assuming antibodies bind independently without competition, the overall experimentally measured fraction of variants that escape binding at concentration $c$ is:
$$
p_v\left(c\right) = \prod_{e=1}^E U_e\left(v, c\right),
\label{pv} \tag{1}
$$
where $e$ ranges over the $E$ epitopes.

Next, we can decompose $U_e\left(v,c\right)$ in terms of underlying physical properties like the relative concentrations of antibodies targeting different epitopes, and the affinities of these antibodies ([Einav et al. 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007830)). If we assume that there is no competition among antibodies binding to different epitopes, that all antibodies targeting a given epitope have same affinity, and that there is no cooperativity in antibody binding (Hill coefficient of antibody binding is one), then the fraction of all variants $v$ that are not bound by an antibody targeting epitope $e$ at concentration $c$ is given by a Hill equation:
$$
\begin{eqnarray}
U_e\left(v, c\right) &=& \frac{1}{1 + \frac{c f_e}{K_{d,e}\left(v\right)}} \\
&=& \frac{1}{1 + c f_e \exp \left(-\frac{\Delta G_e\left(v\right)}{RT}\right)} \\
&=& \frac{1}{1 + c \exp \left(-\phi_e\left(v\right)\right)}, \\
\label{Ue} \tag{2}
\end{eqnarray}
$$
where $\phi_e\left(v\right)$ represents the total binding activity of antibodies to epitope $e$ against variant $v$, and is related to the free energy of binding $\Delta G_e\left(v\right)$ and the fraction of antibodies $f_e$ targeting epitope $e$ by $\phi_e\left(v\right) = \frac{\Delta G_e\left(v\right)}{RT} - \ln f_e$; note that $RT$ is the product of the molar gas constant and the temperature and $K_{d,e}= \exp\left(\frac{\Delta G_e\left(v\right)}{RT}\right)$ is the dissociation constant.
The value of $\phi_e\left(v\right)$ depends both on the affinity of antibodies targeting epitope $e$ (via $\Delta G_e\left(v\right)$) and on the abundance of antibodies with this specificity in the overall mix (via $f_e$), and so is a measure of the overall importance of antibodies with this specificity in the polyclonal mix.
Smaller (more negative) values of $\phi_e\left(v\right)$ correspond to a higher overall contribution of antibodies with specificity for epitope $e$ to the activity against variant $v$.

Finally, we can frame $\phi_e\left(v\right)$ in terms of the actual quantities of biological interest.
There are two quantities of biological interest:

1. The activity of antibodies binding epitope $e$ in the unmutated ("wildtype") protein background, which will be denoted as $a_{\rm{wt}, e}$.
2. The extent of escape mediated by each amino-acid mutation $m$ on binding of antibodies targeting epitope $e$, which will be denoted as $\beta_{m,e}$.

In order to infer these quantities, we assume that mutations have additive effects on the free energy of binding ([Otwinowski et al. 2018](https://academic.oup.com/mbe/article/35/10/2345/5063899?login=true)) for antibodies targeting any given epitope $e$. Specifically, let $a_{\rm{wt}, e}$ be the total activity against the "wildtype" protein of antibodies targeting epitope $e$, with larger values of $a_{\rm{wt}, e}$ indicating stronger antibody binding (or neutralization) at this epitope. Let $\beta_{m,e}$ be the extent to which mutation $m$ (where $1 \le m \le M$) reduces binding by antibodies targeting epitope $e$, with larger values of $\beta_{m,e}$ corresponding to more escape from binding (a value of 0 means the mutation has no effect on antibodies targeting this epitope).
We can then write:
$$
\phi_e\left(v\right) = -a_{\rm{wt}, e} + \sum_{m=1}^M \beta_{m,e} b\left(v\right)_m
\label{phie} \tag{3}
$$
where $b\left(v\right)_m$ is one if variant $v$ has mutation $m$ and 0 otherwise.

These equations relate the quantities of biological interest ($a_{\rm{wt}, e}$ and $\beta_{m,e}$) to the experimental measurables ($p_v\left(c\right)$).

## Optimization
By default, [polyclonal](https://github.com/jbloomlab/polyclonal) uses the gradient-based L-BFGS-B method in `scipy.optimize.minimize` to minimize a Pseudo-Huber loss function on the difference between predicted and measured escape fractions for each variant ($p_v(c)$). We recommend using the default option of first fitting a "site-level" model, in which all mutations are lumped together so there are just two characters (wildtype and mutant). The escape values from this initial site-level fitting are then used to initialize the full mutation-level escape values which are then further optimized. This option is implemented via the `fit_site_level_first` parameter in `Polyclonal.fit`. 

If there are multiple variants with the same mutations, users can choose whether to treat each as independent measurements or "collapse" them to a single variant that is then given a weight proportional to the number of constituent variants that is used when calculating the loss. Collapsing is implemented by default via `collapse_identical_variants` during the initialization of a `Polyclonal` object. In most cases, collapsing speeds up fitting without substantially changing fitting results. However, users should not collapse if performing bootstrapping. 

Lastly, [polyclonal](https://github.com/jbloomlab/polyclonal) models are regularized based on three biologically motivated assumptions:

1. Most mutations should not mediate escape.
2. When a site is involved in escape for a given epitope, most mutations at a site will have similar effects.
3. Epitope activities should be small (or negative) except when clear evidence to the contrary.

We regularize the escape values ($\beta_{m,e}$), the variance of escape values at each site, and the epitope activities ($a_{wt,e}$). Users can tune the strengths of regularization for each by modifying the parameters in `Polyclonal.fit`.


## Bootstrapping

Leave this section for Zorian.

# Simulated data

We [simulated data](https://jbloomlab.github.io/polyclonal/simulate_RBD.html) from a DMS experiment using the SARS-CoV-2 receptor binding domain (RBD) antibody mix in order to provide hypothetical data on which to fit [polyclonal](https://github.com/jbloomlab/polyclonal) models. We use this simulation to demonstrate that fitting a model to the hypothetical data results in accurate inference of the ground-truth mutation escape values and activities for each epitope. Additionally, we use this simulation to identify the [experimental conditions](https://jbloomlab.github.io/polyclonal/expt_design.html) that yield the best model performance.

Maybe add some figures here, alongside static figures that Will can generate?

# Code availability
The `polyclonal` source code is on GitHub at https://github.com/jbloomlab/polyclonal and the documentation is at https://jbloomlab.github.io/polyclonal.

# Acknowledgements


# References