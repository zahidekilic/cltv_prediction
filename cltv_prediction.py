##########################
# Required Library and Functions
##########################
# pip install lifetimes

# lifetimes: It is the library where we make bg/nbd and gamma gamma models.

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# outlier: A value that is well outside the general value distribution of a variable. For example, 200 cm is a value for height, but 250 cm is an outlier.
# Because outliers change the probabilistic ratios in the generalization to be made, they are usually removed from the data set.
# Since the models we will establish are probabilistic models, the distributions of the variables that we use while establishing, will directly affect the results.
# Therefore, after creating the variables we have, we need to touch the outliers in the variables.
# For this reason, we will first detect outliers using a method called the boxplot method, or IQR.
# Then, as part of combating outliers, we will replace the outliers we have identified with a certain threshold value.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1 #IQR value
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    #dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit #you may use this for too low values, in this dataset, we dont need this
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#########################
# Read and Understand the Data
#########################

df = pd.read_excel("online_retail_I.xlsx")

df.describe().T
df.head()
df.isnull().sum()

#########################
# Preparing data
#########################
df.dropna(inplace=True)
df.describe().T

df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]
df.describe().T

#change the outliers
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11) #get the maximum date of from dataset, add 2 days


#########################
# Preparing Lifetime Data Structure
#########################
# frequency represents the number of repeat purchases the customer has made.It must be greater than 1 because it must repeat.
# T represents the age of the customer in whatever time units chosen (weekly, in the above dataset).
# This is equal to the duration between a customer’s first purchase and the end of the period under study. (today_date - first purchase)
# recency represents the age of the customer when they made their most recent purchases. (last purchase - first purchase , recency>1)
# monetary_value represents the average value of a given customer’s purchases. (totalprice /number of purchase)

cltv_df = df.groupby("Customer ID").agg({'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days, # recency
                                                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days], # T
                                         'Invoice': lambda Invoice: Invoice.nunique(), # frequency
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()}) #monetary

cltv_df.columns = cltv_df.columns.droplevel(0)


cltv_df.columns = ["recency","T","frequency","monetary"]
cltv_df = cltv_df[cltv_df["recency"]>1]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"] #average monetary per all purchases

cltv_df.describe().T

cltv_df = cltv_df[cltv_df["frequency"]>1]

cltv_df["recency"] = cltv_df["recency"] / 7 #convert daily to weekly
cltv_df["T"] = cltv_df["T"] / 7 #convert daily to weekly

cltv_df.describe().T

##############################################################
# Setting Up BG-NBD Model
##############################################################
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

plot_period_transactions(bgf)
plt.show()

# Who are the 10 customers we expect the most to purchase in a week?
bgf.predict(1, #number of weeks
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)




# Who are the 10 customers we expect the most to purchase in a month?
bgf.predict(4, #4 weeks=1 month
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)


#total purchase prediction for 1 month
bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()


##############################################################
# Setting Up GAMMA-GAMMA (GG) Model
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

#prediction
ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"]).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

##############################################################
# Calculating CLTV with BG-NBD and GG models
##############################################################

# CLTV = BG/NBD Model * GammaGamma Model

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time = 3, #For 3 months
                                   freq="W", #Weekly
                                   discount_rate=0.01
                                   )
cltv.head()

cltv = cltv.reset_index()

#Merge 2 datasets on Customer ID
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.sort_values(by = "clv", ascending=False).head(10)

##############################################################
# Creating Segments with respect to CLTV
##############################################################

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels= ["D", "C", "B", "A"])
cltv_final.head(50)

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg({"count", "mean", "sum"})
