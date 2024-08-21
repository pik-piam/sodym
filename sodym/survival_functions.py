from abc import abstractmethod
import numpy as np
import scipy.stats


class SurvivalModel():
    def __init__(
        self,
        shape,
        **kwargs
    ):
        self.shape = tuple(shape)
        self.n_t = list(shape)[0]
        self.shape_cohort = (self.n_t,) + self.shape
        self.shape_no_t = tuple(list(self.shape)[1:])
        self.sf = self.survival_function(**kwargs)

    @property
    def t_diag_indices(self):
        return np.diag_indices(self.n_t) + (slice(None),) * len(self.shape_no_t)

    def tile(self, a: np.ndarray) -> np.ndarray:
        index = (slice(None),) * a.ndim + (np.newaxis,) * len(self.shape_no_t)
        out = a[index]
        return np.tile(out, self.shape_no_t)

    def remaining_ages(self, m):
        return self.tile(np.arange(0, self.n_t - m))

    @abstractmethod
    def survival_function(self):
        """Survival table self.sf(m,n) denotes the share of an inflow in year n (age-cohort) still
        present at the end of year m (after m-n years).
        The computation is self.sf(m,n) = ProbDist.sf(m-n), where ProbDist is the appropriate
        scipy function for the lifetime model chosen.
        For lifetimes 0 the sf is also 0, meaning that the age-cohort leaves during the same year
        of the inflow.
        The method compute outflow_sf returns an array year-by-cohort of the surviving fraction of
        a flow added to stock in year m (aka cohort m) in in year n. This value equals sf(n,m).
        This is the only method for the inflow-driven model where the lifetime distribution directly
        enters the computation.
        All other stock variables are determined by mass balance.
        The shape of the output sf array is NoofYears * NoofYears,
        and the meaning is years by age-cohorts.
        The method does nothing if the sf alreay exists.
        For example, sf could be assigned to the dynamic stock model from an exogenous computation
        to save time.
        """
        pass

    def compute_outflow_pdf(self):
        """Lifetime model. The method compute outflow_pdf returns an array year-by-cohort of the probability of a item
        added to stock in year m (aka cohort m) leaves in in year n. This value equals pdf(n,m).

        The pdf is computed from the survival table sf, where the type of the lifetime distribution enters. The shape of
        the output pdf array is n_t * n_t, but the meaning is years by age-cohorts. The method does nothing if the pdf
        already exists.
        """
        self.sf = self.survival_function()
        self.pdf = np.zeros(self.shape_cohort)
        self.pdf[self.t_diag_indices] = 1.0 - np.moveaxis(self.sf.diagonal(0, 0, 1), -1, 0)
        for m in range(0, self.n_t):
            self.pdf[np.arange(m + 1, self.n_t), m, ...] = -1 * np.diff(self.sf[np.arange(m, self.n_t), m, ...], axis=0)
        return self.pdf


class FixedSurvival(SurvivalModel):
    """Fixed lifetime, age-cohort leaves the stock in the model year when the age specified as 'Mean' is reached."""

    def survival_function(self, lifetime_mean, **kwargs):
        sf = np.zeros(self.shape_cohort)
        # Perform specific computations and checks for each lifetime distribution:            
        for m in range(0, self.n_t):  # cohort index
            sf[m::, m, ...] = (self.remaining_ages(m) < lifetime_mean[m, ...]).astype(int)
        # Example: if lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years,
        # gone after 4 years.
        return sf


class NormalSurvival(SurvivalModel):
    """Normally distributed lifetime with mean and standard deviation.
    Watch out for nonzero values, for negative ages, no correction or truncation done here.
    NOTE: As normal distributions have nonzero pdf for negative ages,
    which are physically impossible, these outflow contributions can either be ignored (
    violates the mass balance) or allocated to the zeroth year of residence,
    the latter being implemented in the method compute compute_o_c_from_s_c.
    As alternative, use lognormal or folded normal distribution options.
    """
    def survival_function(self, lifetime_mean, lifetime_std, **kwargs):
        if np.min(lifetime_mean) < 0:
            raise ValueError('lifetime_mean must be greater than zero.')

        sf = np.zeros(self.shape_cohort)
        for m in range(0, self.n_t):  # cohort index
            sf[m::, m, ...] = scipy.stats.norm.sf(
                self.remaining_ages(m),
                loc=lifetime_mean[m, ...],
                scale=lifetime_std[m, ...],
            )
        return sf


class FoldedNormalSurvival(SurvivalModel):
    """Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
    NOTE: call this with the parameters of the normal distribution mu and sigma of curve
    BEFORE folding, curve after folding will have different mu and sigma.
    """
    def survival_function(self, lifetime_mean, lifetime_std, **kwargs):
        if np.min(lifetime_mean) < 0:
            raise ValueError('lifetime_mean must be greater than zero.')

        sf = np.zeros(self.shape_cohort)
        for m in range(0, self.n_t):  # cohort index
            sf[m::, m, ...] = scipy.stats.foldnorm.sf(
                self.remaining_ages(m),
                lifetime_mean[m, ...] / lifetime_std[m, ...],
                0,
                scale=lifetime_std[m, ...],
            )
        return sf


class LogNormalSurvival(SurvivalModel):
    """Lognormal distribution
    Here, the mean and stddev of the lognormal curve, not those of the underlying normal
    distribution, need to be specified!
    Values chosen according to description on
    https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
    Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
    """
    def survival_function(self, lifetime_mean, lifetime_std, **kwargs):
        sf = np.zeros(self.shape_cohort)
        for m in range(0, self.n_t):  # cohort index
            # calculate parameter mu of underlying normal distribution:
            lt_ln = np.log(
                lifetime_mean[m, ...]
                / np.sqrt(
                    1 + lifetime_mean[m, ...] * lifetime_mean[m, ...]
                    / (lifetime_std[m, ...] * lifetime_std[m, ...])
                )
            )
            # calculate parameter sigma of underlying normal distribution
            sg_ln = np.sqrt(
                np.log(
                    1 + lifetime_mean[m, ...] * lifetime_mean[m, ...]
                    / (lifetime_std[m, ...] * lifetime_std[m, ...])
                )
            )
            # compute survial function
            sf[m::, m, ...] = scipy.stats.lognorm.sf(
                self.remaining_ages(m), s=sg_ln, loc=0, scale=np.exp(lt_ln)
            )
        return sf


class WeibullSurvival(SurvivalModel):
    """Weibull distribution with standard definition of scale and shape parameters."""
    def survival_function(self, lifetime_shape, lifetime_scale, **kwargs):
        if np.min(lifetime_shape) < 0:
            raise ValueError("Lifetime shape must be positive for Weibull distribution.")

        sf = np.zeros(self.shape_cohort)
        for m in range(0, self.n_t):  # cohort index
            sf[m::, m, ...] = scipy.stats.weibull_min.sf(
                self.remaining_ages(m),
                c=lifetime_shape[m, ...],
                loc=0,
                scale=lifetime_scale[m, ...],
            )
        return sf
