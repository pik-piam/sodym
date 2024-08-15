.. SODYM Documentation documentation master file, created by
   sphinx-quickstart on Wed Aug 14 08:41:40 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SODYM documentation
=================================

The sodym package provides key functionality for material flow analysis, with the class `MFASystem` acting as a template (parent class) for users to create their own material flow models.

| The concepts behind sodym are based on:
| ODYM
| Copyright (c) 2018 Industrial Ecology
| author: Stefan Pauliuk, Uni Freiburg, Germany
| https://github.com/IndEcol/ODYM

Model components
----------------
.. toctree::
   :maxdepth: 1

   sodym.mfa_definition
   sodym.data_reader
   sodym.dimension
   sodym.named_dim_arrays
   sodym.stocks
   sodym.mfa_system

Data writing and plotting
-------------------------
.. toctree::
   :maxdepth: 1

   sodym.export.data_writer
