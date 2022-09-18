# https://github.com/chilledgeek/blog_dump/blob/main/fictional_scenarios/fictional_sales.ipynb
# https://towardsdatascience.com/a-little-code-optimisation-goes-a-long-way-91f92ff9f468

import numpy

#selling_prices = list(numpy.random.randint(1, 1100, size=20000000))
#costs_per_unit = list(numpy.random.randint(1, 1000, size=20000000))
#units_sold_counts = list(numpy.random.randint(1, 2000, size=20000000))

selling_prices = list(numpy.random.randint(1, 1100, size=2000000))
costs_per_unit = list(numpy.random.randint(1, 1000, size=2000000))
units_sold_counts = list(numpy.random.randint(1, 2000, size=2000000))

import time

# Method 1: "Exploratory" computation
import pandas
start_time1 = time.time() #start time

# This is completely dumb, but just for exaggeration purposes...
df = pandas.DataFrame()
df.loc[:, "selling_prices"]  = selling_prices
df.loc[:, "costs_per_unit"]  = costs_per_unit
df.loc[:, "units_sold_counts"]  = units_sold_counts

df.head()

profit_per_fruit = df.apply(
    lambda x: (x["selling_prices"] - x["costs_per_unit"]) * x["units_sold_counts"],
    axis=1,
)

total_profit = sum(profit_per_fruit)

end_time1 = time.time() #end time
print("Elapsed time without any optimization is  {}".format(end_time1-start_time1))

#print(total_profit)

# Method 2 : now simplify

start_time2 = time.time() #start time

df = pandas.DataFrame(
    {
        "selling_prices": selling_prices,
        "costs_per_unit": costs_per_unit,
        "units_sold_counts": units_sold_counts,
    }
)

# Compute
profit_per_fruit = (df["selling_prices"] - df["costs_per_unit"]) * df["units_sold_counts"]


end_time2 = time.time() #end time
print("Elapsed time using Panda DataFrame  is    {}".format(end_time2-start_time2))

#assert sum(profit_per_fruit) == total_profit  # Just to check for consistency

#print(total_profit)

# Method 3 : with Numpy

# With numpy


start_time3 = time.time() #start time

price_array = numpy.array(selling_prices)
cost_array = numpy.array(costs_per_unit)
units_array = numpy.array(units_sold_counts)

profit_per_fruit = (price_array - cost_array) * units_array

end_time3 = time.time() #end time
print("Elapsed time using builtin numpy util is  {}".format(end_time3-start_time3))

#assert sum(profit_per_fruit) == total_profit  # Just to check for consistency
#print(total_profit)
