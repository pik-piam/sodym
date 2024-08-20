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


class DynamicStockModel():
    def __init__(self, shape, survival_model: SurvivalModel) -> None:
        self.shape = tuple(shape)
        self.n_t = list(shape)[0]
        self.shape_cohort = (self.n_t,) + self.shape
        self.shape_no_t = tuple(list(self.shape)[1:])
        self.survival_model = survival_model

    @property
    def t_diag_indices(self):
        return np.diag_indices(self.n_t) + (slice(None),) * len(self.shape_no_t)

    @abstractmethod
    def compute():
        pass


class InflowDrivenDSM(DynamicStockModel):
    """Inflow driven model
    Given: inflow, lifetime dist.
    Default order of methods:
    1) determine stock by cohort
    2) determine total stock
    2) determine outflow by cohort
    3) determine total outflow
    4) check mass balance.
    """
    def __init__(self, shape, survival_model: SurvivalModel, inflow):
        super().__init__(shape, survival_model)
        self.inflow = inflow

    def compute(self):
        stock_by_cohort = self.compute_i_lt__2__sc()
        outflow_by_cohort = self.compute_sc__2__oc(stock_by_cohort)
        self.stock = stock_by_cohort.sum(axis=1)
        self.outflow = outflow_by_cohort.sum(axis=1)

    def compute_i_lt__2__sc(self):
        """With given inflow and lifetime distribution, the method builds the stock by cohort."""
        stock_by_cohort = np.einsum("c...,tc...->tc...", self.inflow, self.survival_model.sf)
        # This command means: s_c[t,c] = i[c] * sf[t,c] for all t, c
        # from the perspective of the stock the inflow has the dimension age-cohort,
        # as each inflow(t) is added to the age-cohort c = t
        return stock_by_cohort

    def compute_sc__2__oc(self, stock_by_cohort):
        """Compute outflow by cohort from stock by cohort."""
        outflow_by_cohort = np.zeros(self.shape_cohort)
        outflow_by_cohort[1:, :, ...] = -np.diff(stock_by_cohort, axis=0)
        outflow_by_cohort[self.t_diag_indices] = self.inflow - np.moveaxis(
            stock_by_cohort.diagonal(0, 0, 1), -1, 0
        )  # allow for outflow in year 0 already
        return outflow_by_cohort

    def compute_s_is__2__i(self, initial_stock: np.ndarray):
        """Given a stock at t0 broken down by different cohorts tx ...  t0, an "initial stock",
        This method calculates the original inflow that generated this stock.
        """
        assert initial_stock.shape[0] == self.n_t
        self.inflow = np.zeros(self.shape)
        # construct the sf of a product of cohort tc surviving year t
        # using the lifetime distributions of the past age-cohorts
        for cohort in range(0, self.n_t):
            self.inflow[cohort, ...] = np.where(
                self.survival_model.sf[-1, cohort, ...] != 0,
                initial_stock[cohort, ...] / self.survival_model.sf[-1, cohort, ...],
                0.0,
            )
        return self.inflow


class StockDrivenDSM(DynamicStockModel):
    """Stock driven model
    Given: total stock, lifetime dist.
    Default order of methods:
    1) determine inflow, outflow by cohort, and stock by cohort
    2) determine total outflow
    3) determine stock change
    4) check mass balance.
    """
    def __init__(self, shape, survival_model: SurvivalModel, stock):
        super().__init__(shape, survival_model)
        self.stock = stock

    def compute(self):
        self.inflow, outflow_by_cohort, stock_by_cohort = self.compute_s_lt__2__sc_oc_i()
        self.outflow = outflow_by_cohort.sum(axis=1)

    def compute_s_lt__2__sc_oc_i(self, do_correct_negative_inflow=False):
        """With given total stock and lifetime distribution, the method builds the stock by cohort and the inflow."""
        stock_by_cohort = np.zeros(self.shape_cohort)
        outflow_by_cohort = np.zeros(self.shape_cohort)
        inflow = np.zeros(self.shape)
        sf = self.survival_model.sf
        # construct the sf of a product of cohort tc remaining in the stock in year t
        # First year:
        inflow[0, ...] = np.where(sf[0, 0, ...] != 0.0, self.stock[0] / sf[0, 0], 0.0)
        stock_by_cohort[:, 0, ...] = (
            inflow[0, ...] * sf[:, 0, ...]
        )  # Future decay of age-cohort of year 0.
        outflow_by_cohort[0, 0, ...] = inflow[0, ...] - stock_by_cohort[0, 0, ...]
        # all other years:
        for m in range(1, self.n_t):  # for all years m, starting in second year
            # 1) Compute outflow from previous age-cohorts up to m-1
            outflow_by_cohort[m, 0:m, ...] = (
                stock_by_cohort[m - 1, 0:m, ...] - stock_by_cohort[m, 0:m, ...]
            )  # outflow table is filled row-wise, for each year m.
            # 2) Determine inflow from mass balance:
            if not do_correct_negative_inflow:  # if no correction for negative inflows is made
                inflow[m, ...] = np.where(
                    sf[m, m, ...] != 0.0,
                    (self.stock[m, ...] - stock_by_cohort[m, :, ...].sum(axis=0)) / sf[m, m, ...],
                    0.0,
                )  # allow for outflow during first year by rescaling with 1/sf[m,m]
                # 3) Add new inflow to stock and determine future decay of new age-cohort
                stock_by_cohort[m::, m, ...] = inflow[m, ...] * sf[m::, m, ...]
                outflow_by_cohort[m, m, ...] = inflow[m, ...] * (1 - sf[m, m, ...])
            # 2a) Correct remaining stock in cases where inflow would be negative:
            else:
                # if the stock declines faster than according to the lifetime model, this option allows to extract
                # additional stock items.
                # The negative inflow correction implemented here was developed in a joined effort by Sebastiaan Deetman
                # and Stefan Pauliuk.
                inflow_test = self.stock[m, ...] - stock_by_cohort[m, :, ...].sum(axis=0)
                if inflow_test < 0:  # if stock-driven model would yield negative inflow
                    delta = -1 * inflow_test  # Delta > 0!
                    inflow[m, ...] = 0  # Set inflow to 0 and distribute mass balance gap onto remaining cohorts:
                    delta_percent = np.where(
                        stock_by_cohort[m, :, ...].sum(axis=0) != 0,
                        delta / stock_by_cohort[m, :, ...].sum(axis=0),
                        0.0,
                    )
                    # - Distribute gap equally across all cohorts (each cohort is adjusted by the same %, based on
                    #   surplus with regards to the prescribed stock)
                    # - delta_percent is a % value <= 100%
                    # - correct for outflow and stock in current and future years
                    # - adjust the entire stock AFTER year m as well, stock is lowered in year m, so future cohort
                    #   survival also needs to decrease.

                    # increase outflow according to the lost fraction of the stock, based on Delta_c
                    outflow_by_cohort[m, :, ...] = outflow_by_cohort[m, :, ...] + (
                        stock_by_cohort[m, :, ...] * delta_percent
                    )
                    # shrink future description of stock from previous age-cohorts by factor Delta_percent in current
                    # AND future years.
                    stock_by_cohort[m::, 0:m, ...] = (
                        stock_by_cohort[m::, 0:m, ...] * (1 - delta_percent)
                    )
                else:  # If no negative inflow would occur
                    inflow[m, ...] = np.where(
                        sf[m, m, ...] != 0,  # Else, inflow is 0.
                        (self.stock[m, ...] - stock_by_cohort[m, :, ...].sum(axis=0))
                        / sf[m, m, ...],  # allow for outflow during first year by rescaling with 1/sf[m,m]
                        0.0,
                    )
                    # Add new inflow to stock and determine future decay of new age-cohort
                    stock_by_cohort[m::, m, ...] = inflow[m, ...] * sf[m::, m, ...]
                    outflow_by_cohort[m, m, ...] = inflow[m, ...] * (1 - sf[m, m, ...])
                # NOTE: This method of negative inflow correction is only of of many plausible methods of increasing the
                # outflow to keep matching stock levels. It assumes that the surplus stock is removed in the year that
                # it becomes obsolete. Each cohort loses the same fraction. Modellers need to try out whether this
                # method leads to justifiable results. In some situations it is better to change the lifetime assumption
                # than using the NegativeInflowCorrect option.

        return inflow, outflow_by_cohort, stock_by_cohort
