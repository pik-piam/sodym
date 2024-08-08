"""
Adapted from:

ODYM
Copyright (c) 2018 Industrial Ecology
author: Stefan Pauliuk, Uni Freiburg, Germany
https://github.com/IndEcol/ODYM

Adapted for use in the simson project
"""

import numpy as np
import scipy.stats


class DynamicStockModel(object):
    def __init__(
        self,
        shape,
        inflow=None,
        outflow=None,
        stock=None,
        ldf_type=None,
        lifetime_mean=None,
        lifetime_std=None,
        lifetime_shape=None,
        lifetime_scale=None,
        stock_by_cohort=None,
        outflow_by_cohort=None,
        sf=None,
    ):
        self.shape = tuple(shape)
        self.n_t = list(shape)[0]
        self.shape_cohort = (self.n_t,) + self.shape
        self.shape_no_t = tuple(list(self.shape)[1:])

        self.inflow = inflow
        self.stock = stock
        self.dsdt = None
        self.stock_by_cohort = stock_by_cohort
        self.outflow = outflow
        self.outflow_by_cohort = outflow_by_cohort

        self.ldf_type = ldf_type
        self.lifetime_mean = lifetime_mean
        self.lifetime_std = lifetime_std
        self.lifetime_shape = lifetime_shape
        self.lifetime_scale = lifetime_scale

        self.sf = sf
        return

    def tile(self, a: np.ndarray) -> np.ndarray:
        index = (slice(None),) * a.ndim + (np.newaxis,) * len(self.shape_no_t)
        out = a[index]
        return np.tile(out, self.shape_no_t)

    @property
    def t_diag_indices(self):
        return np.diag_indices(self.n_t) + (slice(None),) * len(self.shape_no_t)

    # THEORY:
    # six quantities: i, o, s, sc, oc, lt

    def compute_stock_driven(self):
        assert self.stock is not None, "Stock must be specified."
        self.compute_s_lt__2__sc_oc_i()
        self.compute_oc__2__o()
        self.check_stock_balance()

    def compute_inflow_driven(self):
        assert self.inflow is not None, "Inflow must be specified."
        self.compute_i_lt__2__sc()
        self.compute_sc__2__oc()
        self.compute_sc__2__s()
        self.compute_oc__2__o()
        self.check_stock_balance()

    def compute_simple_stock(self):
        self.compute_i_o__2__s()

    def compute_i_o__2__s(self):
        """Compute stock from inflow and outflow via mass balance."""
        self.stock = np.cumsum(self.inflow - self.outflow, axis=0)
        return self.stock

    def compute_s__2__dsdt(self):
        """Determine stock change from time series for stock.
        Formula: stock_change(t) = stock(t) - stock(t-1).
        """
        if self.dsdt is None:
            self.dsdt = np.diff(self.stock, axis=0, prepend=0)
        return self.dsdt

    def compute_sc__2__s(self):
        """Determine total stock as row sum of cohort-specific stock."""
        self.stock = self.stock_by_cohort.sum(axis=1)
        return self.stock

    def compute_oc__2__o(self):
        """Determine total outflow as row sum of cohort-specific outflow."""
        self.outflow = self.outflow_by_cohort.sum(axis=1)
        return self.outflow

    def compute_i_s__2__o(self):
        """Compute outflow from process via mass balance.
        Needed in cases where lifetime is zero.
        """
        self.compute_s__2__dsdt()
        self.outflow = self.inflow - self.dsdt
        return self.outflow

    def compute_o_s__2__i(self):
        self.compute_s__2__dsdt()
        self.inflow = self.dsdt + self.outflow
        return self.outflow

    """ Part 2: Lifetime model. """

    def compute_outflow_pdf(self):
        """Lifetime model. The method compute outflow_pdf returns an array year-by-cohort of the probability of a item
        added to stock in year m (aka cohort m) leaves in in year n. This value equals pdf(n,m).

        The pdf is computed from the survival table sf, where the type of the lifetime distribution enters. The shape of
        the output pdf array is n_t * n_t, but the meaning is years by age-cohorts. The method does nothing if the pdf
        already exists.
        """
        self.compute_lt__2__sf()
        self.pdf = np.zeros(self.shape_cohort)
        self.pdf[self.t_diag_indices] = 1.0 - np.moveaxis(self.sf.diagonal(0, 0, 1), -1, 0)
        for m in range(0, self.n_t):
            self.pdf[np.arange(m + 1, self.n_t), m, ...] = -1 * np.diff(self.sf[np.arange(m, self.n_t), m, ...], axis=0)
        return self.pdf

    def compute_lt__2__sf(self):  # survival functions
        """Survival table self.sf(m,n) denotes the share of an inflow in year n (age-cohort) still present at the end of
        year m (after m-n years). The computation is self.sf(m,n) = ProbDist.sf(m-n), where ProbDist is the appropriate
        scipy function for the lifetime model chosen. For lifetimes 0 the sf is also 0, meaning that the age-cohort
        leaves during the same year of the inflow. The method compute outflow_sf returns an array year-by-cohort of the
        surviving fraction of a flow added to stock in year m (aka cohort m) in in year n. This value equals sf(n,m).
        This is the only method for the inflow-driven model where the lifetime distribution directly enters the.

        computation. All other stock variables are determined by mass balance. The shape of the output sf array is
        NoofYears * NoofYears, and the meaning is years by age-cohorts. The method does nothing if the sf alreay exists.
        For example, sf could be assigned to the dynamic stock model from an exogenous computation to save time.
        """

        if self.sf is not None:
            return  # if sf is already computed, do nothing.

        self.check_lifetime_consistency()
        self.sf = np.zeros(self.shape_cohort)
        # Perform specific computations and checks for each lifetime distribution:

        def remaining_ages(m):
            return self.tile(np.arange(0, self.n_t - m))

        if (
            self.ldf_type == "Fixed"
        ):  # fixed lifetime, age-cohort leaves the stock in the model year when the age specified as 'Mean' is reached.
            for m in range(0, self.n_t):  # cohort index
                self.sf[m::, m, ...] = (remaining_ages(m) < self.lifetime_mean[m, ...]).astype(
                    int
                )  # converts bool to 0/1
            # Example: if lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years, gone after 4
            # years.

        if (
            self.ldf_type == "Normal"
        ):  # normally distributed lifetime with mean and standard deviation. Watch out for nonzero values
            # for negative ages, no correction or truncation done here. Cf. note below.
            for m in range(0, self.n_t):  # cohort index
                self.sf[m::, m, ...] = scipy.stats.norm.sf(
                    remaining_ages(m),
                    loc=self.lifetime_mean[m, ...],
                    scale=self.lifetime_std[m, ...],
                )
                # NOTE: As normal distributions have nonzero pdf for negative ages, which are physically impossible,
                # these outflow contributions can either be ignored (violates the mass balance) or allocated to the
                # zeroth year of residence, the latter being implemented in the method compute compute_o_c_from_s_c.
                # As alternative, use lognormal or folded normal distribution options.

        if (
            self.ldf_type == "FoldedNormal"
        ):  # Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
            for m in range(0, self.n_t):  # cohort index
                self.sf[m::, m, ...] = scipy.stats.foldnorm.sf(
                    remaining_ages(m),
                    self.lifetime_mean[m, ...] / self.lifetime_std[m, ...],
                    0,
                    scale=self.lifetime_std[m, ...],
                )
                # NOTE: call this option with the parameters of the normal distribution mu and sigma of curve BEFORE
                # folding, curve after folding will have different mu and sigma.

        if self.ldf_type == "LogNormal":  # lognormal distribution
            # Here, the mean and stddev of the lognormal curve,
            # not those of the underlying normal distribution, need to be specified! conversion of parameters done here:
            for m in range(0, self.n_t):  # cohort index
                # calculate parameter mu    of underlying normal distribution:
                lt_ln = np.log(
                    self.lifetime_mean[m, ...]
                    / np.sqrt(
                        1
                        + self.lifetime_mean[m, ...]
                        * self.lifetime_mean[m, ...]
                        / (self.lifetime_std[m, ...] * self.lifetime_std[m, ...])
                    )
                )
                # calculate parameter sigma of underlying normal distribution:
                sg_ln = np.sqrt(
                    np.log(
                        1
                        + self.lifetime_mean[m, ...]
                        * self.lifetime_mean[m, ...]
                        / (self.lifetime_std[m, ...] * self.lifetime_std[m, ...])
                    )
                )
                # compute survial function
                self.sf[m::, m, ...] = scipy.stats.lognorm.sf(remaining_ages(m), s=sg_ln, loc=0, scale=np.exp(lt_ln))
                # values chosen according to description on
                # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
                # Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"

        if self.ldf_type == "Weibull":  # Weibull distribution with standard definition of scale and shape parameters
            for m in range(0, self.n_t):  # cohort index
                self.sf[m::, m, ...] = scipy.stats.weibull_min.sf(
                    remaining_ages(m),
                    c=self.lifetime_shape[m, ...],
                    loc=0,
                    scale=self.lifetime_scale[m, ...],
                )

    """
    Part 3: Inflow driven model
    Given: inflow, lifetime dist.
    Default order of methods:
    1) determine stock by cohort
    2) determine total stock
    2) determine outflow by cohort
    3) determine total outflow
    4) check mass balance.
    """

    def compute_i_lt__2__sc(self):
        """With given inflow and lifetime distribution, the method builds the stock by cohort."""
        self.compute_lt__2__sf()
        self.stock_by_cohort = np.einsum("c...,tc...->tc...", self.inflow, self.sf)
        # This command means: s_c[t,c] = i[c] * sf[t,c] for all t, c
        # from the perspective of the stock the inflow has the dimension age-cohort,
        # as each inflow(t) is added to the age-cohort c = t
        return self.stock_by_cohort

    def compute_sc__2__oc(self):
        """Compute outflow by cohort from stock by cohort."""
        self.outflow_by_cohort = np.zeros(self.shape_cohort)
        self.outflow_by_cohort[1:, :, ...] = -np.diff(self.stock_by_cohort, axis=0)
        self.outflow_by_cohort[self.t_diag_indices] = self.inflow - np.moveaxis(
            self.stock_by_cohort.diagonal(0, 0, 1), -1, 0
        )  # allow for outflow in year 0 already
        return self.outflow_by_cohort

    def compute_s_is__2__i(self, initial_stock: np.ndarray):
        """Given a stock at t0 broken down by different cohorts tx ...  t0, an "initial stock",
        This method calculates the original inflow that generated this stock.
        """
        assert initial_stock.shape[0] == self.n_t
        self.inflow = np.zeros(self.shape)
        # construct the sf of a product of cohort tc surviving year t
        # using the lifetime distributions of the past age-cohorts
        self.compute_lt__2__sf()
        for cohort in range(0, self.n_t):
            self.inflow[cohort, ...] = np.where(
                self.sf[-1, cohort, ...] != 0,
                initial_stock[cohort, ...] / self.sf[-1, cohort, ...],
                0.0,
            )
        return self.inflow

    """
    Part 4: Stock driven model
    Given: total stock, lifetime dist.
    Default order of methods:
    1) determine inflow, outflow by cohort, and stock by cohort
    2) determine total outflow
    3) determine stock change
    4) check mass balance.
    """

    def compute_s_lt__2__sc_oc_i(self, do_correct_negative_inflow=False):
        """With given total stock and lifetime distribution, the method builds the stock by cohort and the inflow."""
        self.stock_by_cohort = np.zeros(self.shape_cohort)
        self.outflow_by_cohort = np.zeros(self.shape_cohort)
        self.inflow = np.zeros(self.shape)
        # construct the sf of a product of cohort tc remaining in the stock in year t
        self.compute_lt__2__sf()  # Computes sf if not present already.
        # First year:
        self.inflow[0, ...] = np.where(self.sf[0, 0, ...] != 0.0, self.stock[0] / self.sf[0, 0], 0.0)
        self.stock_by_cohort[:, 0, ...] = (
            self.inflow[0, ...] * self.sf[:, 0, ...]
        )  # Future decay of age-cohort of year 0.
        self.outflow_by_cohort[0, 0, ...] = self.inflow[0, ...] - self.stock_by_cohort[0, 0, ...]
        # all other years:
        for m in range(1, self.n_t):  # for all years m, starting in second year
            # 1) Compute outflow from previous age-cohorts up to m-1
            self.outflow_by_cohort[m, 0:m, ...] = (
                self.stock_by_cohort[m - 1, 0:m, ...] - self.stock_by_cohort[m, 0:m, ...]
            )  # outflow table is filled row-wise, for each year m.
            # 2) Determine inflow from mass balance:
            if not do_correct_negative_inflow:  # if no correction for negative inflows is made
                self.inflow[m, ...] = np.where(
                    self.sf[m, m, ...] != 0.0,
                    (self.stock[m, ...] - self.stock_by_cohort[m, :, ...].sum(axis=0)) / self.sf[m, m, ...],
                    0.0,
                )  # allow for outflow during first year by rescaling with 1/sf[m,m]
                # 3) Add new inflow to stock and determine future decay of new age-cohort
                self.stock_by_cohort[m::, m, ...] = self.inflow[m, ...] * self.sf[m::, m, ...]
                self.outflow_by_cohort[m, m, ...] = self.inflow[m, ...] * (1 - self.sf[m, m, ...])
            # 2a) Correct remaining stock in cases where inflow would be negative:
            else:
                # if the stock declines faster than according to the lifetime model, this option allows to extract
                # additional stock items.
                # The negative inflow correction implemented here was developed in a joined effort by Sebastiaan Deetman
                # and Stefan Pauliuk.
                inflow_test = self.stock[m, ...] - self.stock_by_cohort[m, :, ...].sum(axis=0)
                if inflow_test < 0:  # if stock-driven model would yield negative inflow
                    delta = -1 * inflow_test  # Delta > 0!
                    self.inflow[m, ...] = 0  # Set inflow to 0 and distribute mass balance gap onto remaining cohorts:
                    delta_percent = np.where(
                        self.stock_by_cohort[m, :, ...].sum(axis=0) != 0,
                        delta / self.stock_by_cohort[m, :, ...].sum(axis=0),
                        0.0,
                    )
                    # - Distribute gap equally across all cohorts (each cohort is adjusted by the same %, based on
                    #   surplus with regards to the prescribed stock)
                    # - delta_percent is a % value <= 100%
                    # - correct for outflow and stock in current and future years
                    # - adjust the entire stock AFTER year m as well, stock is lowered in year m, so future cohort
                    #   survival also needs to decrease.

                    # increase outflow according to the lost fraction of the stock, based on Delta_c
                    self.outflow_by_cohort[m, :, ...] = self.outflow_by_cohort[m, :, ...] + (
                        self.stock_by_cohort[m, :, ...] * delta_percent
                    )
                    # shrink future description of stock from previous age-cohorts by factor Delta_percent in current
                    # AND future years.
                    self.stock_by_cohort[m::, 0:m, ...] = (
                        self.stock_by_cohort[m::, 0:m, ...] * (1 - delta_percent)
                    )
                else:  # If no negative inflow would occur
                    self.inflow[m, ...] = np.where(
                        self.sf[m, m, ...] != 0,  # Else, inflow is 0.
                        (self.stock[m, ...] - self.stock_by_cohort[m, :, ...].sum(axis=0))
                        / self.sf[m, m, ...],  # allow for outflow during first year by rescaling with 1/sf[m,m]
                        0.0,
                    )
                    # Add new inflow to stock and determine future decay of new age-cohort
                    self.stock_by_cohort[m::, m, ...] = self.inflow[m, ...] * self.sf[m::, m, ...]
                    self.outflow_by_cohort[m, m, ...] = self.inflow[m, ...] * (1 - self.sf[m, m, ...])
                # NOTE: This method of negative inflow correction is only of of many plausible methods of increasing the
                # outflow to keep matching stock levels. It assumes that the surplus stock is removed in the year that
                # it becomes obsolete. Each cohort loses the same fraction. Modellers need to try out whether this
                # method leads to justifiable results. In some situations it is better to change the lifetime assumption
                # than using the NegativeInflowCorrect option.

        return self.stock_by_cohort, self.outflow_by_cohort, self.inflow

    def check_lifetime_consistency(self):
        """Check if lifetime parameters are consistent with the lifetime distribution type."""
        if self.ldf_type == "Fixed":
            assert self.lifetime_mean is not None, "Lifetime mean must be specified."
        elif self.ldf_type in ["Normal", "FoldedNormal", "LogNormal"]:
            assert self.lifetime_mean is not None, "Lifetime mean must be specified."
            assert self.lifetime_std is not None, "Lifetime standard deviation must be specified."
            assert (
                np.min(self.lifetime_mean) > 0.0
            ), "Lifetime mean must be positive for Normal & FoldedNormal distributions. For zero lifetime, use Fixed " \
                "distribution."
        elif self.ldf_type == "Weibull":
            assert self.lifetime_shape is not None, "Lifetime shape must be specified."
            assert self.lifetime_scale is not None, "Lifetime scale must be specified."
            assert np.min(self.lifetime_shape) > 0.0, "Lifetime shape must be positive for Weibull distribution."
        else:
            raise ValueError("Lifetime distribution type not set or invalid.")

    def check_stock_balance(self):
        balance = self.get_stock_balance()
        balance = np.max(np.abs(balance).sum(axis=0))
        if balance > 1:  # 1 tonne accuracy
            raise RuntimeError("Stock balance for dynamic stock model is too high: " + str(balance))
        elif balance > 0.001:
            print("Stock balance for model dynamic stock model is noteworthy: " + str(balance))

    def get_stock_balance(self):
        """Check wether inflow, outflow, and stock are balanced.

        If possible, the method returns the vector 'Balance', where Balance = inflow - outflow - stock_change
        """
        return self.inflow - self.outflow - self.compute_s__2__dsdt()
