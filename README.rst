PandaTorch
==========

A flexible simple library that makes it easier to use the extrememly
popular pandas package with the other extremely popular framework
PyTorch.

Functions
---------

1. Converts a Pandas DataFrame into a usable PyTorch dataset.
2. Allows use of all usual Pandas functions

Usage
-----

::

   import pandas as pd
   from pandatorch import data
   df=pd.read_csv("path_to_dataset")
   torch_df=data.DataFrame(X=df.drop("<Target Column>",axis=1),y=df["<Target Column>"])

**Note:** Check out the notebooks folder for a full end-to-end training
example of a tabular dataset using PandaTorch
