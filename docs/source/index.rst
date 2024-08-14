.. SODYM Documentation documentation master file, created by
   sphinx-quickstart on Wed Aug 14 08:41:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SODYM documentation
=================================

The sodym package provides key functionality for material flow analysis, with the class `MFASystem` acting as a template (parent class) for users to create their own material flow models.

.. toctree::
   :maxdepth: 1

   sodym.classes.mfa_definition
   sodym.classes.mfa_system

Model attributes
----------------
.. toctree::
   :maxdepth: 1

   sodym.classes.dimension
   sodym.classes.named_dim_arrays
   sodym.classes.stocks_in_mfa

Data reading and writing
------------------------
.. toctree::
   :maxdepth: 1

   sodym.classes.data_reader
   sodym.export.data_writer
