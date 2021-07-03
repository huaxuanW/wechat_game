import pandas as pd

import numpy as np

from tqdm import tqdm

from collections import defaultdict

import gc



def reduce_mem(df, cols):

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in tqdm(cols):

        col_type = df[col].dtypes

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):

                    df[col] = df[col].astype(np.float16)

                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    print(
        "{:.2f} Mb, {:.2f} Mb ({:.2f} %)".format(
            start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem
        )
    )

    gc.collect()

    return df

import pandas as pd
import numpy as np
from tqdm import tqdm

def reduce(df):
	int_list = ['int', 'int32', 'int16']
	float_list = ['float', 'float32']
	for col in tqdm(df.columns):
		col_type = df[col].dtypes
		if col_type in int_list:
			c_min = df[col].min()
			c_max = df[col].max()
			if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
				df[col] = df[col].astype(np.int8)
			elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
				df[col] = df[col].astype(np.int16)
			elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
				df[col] = df[col].astype(np.int32)
		elif col_type in float_list:
			c_min = df[col].min()
			c_max = df[col].max()
			if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
				df[col] = df[col].astype(np.float16)
			elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
				df[col] = df[col].astype(np.float32)
	return df

def reduce_s(df):
	int_list = ['int', 'int32', 'int16']
	float_list = ['float', 'float32']
	col_type = df.dtypes
	if col_type in int_list:
		c_min = df.min()
		c_max = df.max()
		if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
			df = df.astype(np.int8)
		elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
			df = df.astype(np.int16)
		elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
			df = df.astype(np.int32)
	elif col_type in float_list:
		c_min = df.min()
		c_max = df.max()
		if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
			df = df.astype(np.float16)
		elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
			df = df.astype(np.float32)
	return df