# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np

#for pandas groupby class transform method
def share(col):
    return col/col.sum()

#making certain that the data is on a consistent basis
#should consider making into a class...
#SHOULD MODIFY TO RETURN A GROUP PANDAS OBJECT' IF ANY BY VARS
def aggregate(data,
              prod_id,
              by_vars=[],
              qty_var="qty",
              sales_var="sales",
              period_var="period",
              first_period=None,
              last_period=None):

        #takes data frame and aggregates, keeping only interesting variables
        #assuming that period_var is in date_time variable
        #but want it as a period index
        df = data.loc[:, by_vars + [period_var, qty_var, sales_var, prod_id]].copy()
        #rename
        df = df.rename(columns={period_var:"period",
                                qty_var:"qty",
                                sales_var:"sales",
                                prod_id:"prod_id"})

        #to make certain that we just have month begin
        #will be updated to acoocamde quarters and weeks
        df.loc[:,"period"] = df.loc[:,"period"] + pd.tseries.offsets.MonthBegin(n=0)

        #SHOULD PUT IN A WARNING IF NO PRODUCTS IN A CERTAIN PERIOD

        #AGGREGATE based on provided values
        df = df.groupby(by_vars + ["prod_id", "period"]).sum().reset_index()
        #remove non positive sales and quanitites
        df = df.loc[(df["qty"] > 0) & (df["sales"]> 0)]
        #prices
        df["price"] = df["sales"]/df["qty"]
        #log price is useful for many indexes
        df["ln_price"] = np.log(df["price"])
        #df = df.sort_values("period")

        #return a grouped pandas object if passed group variables
        if len(by_vars) > 0:
            df = df.groupby(by_vars)

        return df

#a function for extracting unique periods from dataframe
#returns unique periods, sorted, a well as last and first period
def extract_periods(periods, periods_unique):
    if periods_unique is None:
        periods_unique = np.unique(periods) #will be sorted
    return periods_unique, periods_unique[0], periods_unique[-1]

#this function is intended for the renaming of the index column to somthing else
#it is a regular expression, since can multiple index columns with different suffixes
def rename_index(df, index_name):
    if index_name is not None:
        def rename_index_col(col):
            return re.sub(r"index", index_name, col)
        df = df.rename(columns=rename_index_col)
    return df

def clean_up(df, index_name):
    df = rename_index(df, index_name)
    #make period into an index
    df = df.set_index("period")
    return df

#BELOW IS DEPRECATED
#NOT THAT IS WAS EVER PRECATED
#WILL NEED NEW FUNCTIONS FOR CHURN
def calc_churn(df):
    #where is there no valid backprice? (i.e. new product/entry)
    df["entry"] = pd.isnull(df.sales_b)
    df["entry_wghtd"] = df["entry"].astype(np.int) * df["sales_t"]

    #where is there no valid current price (i.e. disappearing product/exit)
    df["exit"] = pd.isnull(df.sales_t)
    df["exit_wghtd"] = df["exit"].astype(np.int) * df["sales_b"]

    #for easier shares
    df["count"] = 1

    #track by class and period
    churn = df.loc[:,["period","entry","exit","count",
                      "entry_wghtd", "exit_wghtd", "sales_t", "sales_b"]]
    churn = churn.groupby("period").sum().reset_index()

    #share
    churn["exit_wghtd_share"] = churn["exit_wghtd"]/churn["sales_t"]
    churn["exit_share"] = churn["exit"]/churn["count"]

    churn["entry_wghtd_share"] = churn["entry_wghtd"]/churn["sales_b"]
    churn["entry_share"] = churn["entry"]/churn["count"]

    churn = churn.loc[:, ["period","exit_wghtd_share","exit_share","entry_wghtd_share",
                         "entry_share"]]

    churn = churn.sort_values("period")

    return churn, df

#calculates a chained bilateral index
#require specification of a bilateral index function
#currently the following are implemented: laspeyres_index, paasche_index, fisher_index, torqnvist_index
def chained_bilateral(df, index_function, index_name=None, periods=None):
    #avoid modifying orignal
    #there might be a better way
    df = df.copy()
    periods, first_period, last_period = extract_periods(df["period"],periods)

    #increment using pandas datetime and Pandas offsets MonthBegin
    #it seems to be the fastest way
    #will need to update to handle other time periods
    df["period_prev"] = df["period"] + pd.tseries.offsets.MonthBegin(n=-1)

    #now merge
    #right table is merged using previous month, so its prices are current and receive the "_t" suffix
    #par contre, the left table is merged with month matching month_prev on right table, so are back prices
    df = df.drop(["period_prev"],axis=1).merge(df,how='inner',
                                               left_on=["period","prod_id"],
                                               right_on=["period_prev","prod_id"],
                                               suffixes=("_b", "_t"))
    #period_t is current period
    df["period"] = df["period_t"]

    #for first period, set current and past values to be the same
    df.loc[df.period == first_period,
           ["sales_b", "qty_b", "price_b"]] = df.loc[df.period == first_period,
                                                    ["sales_t", "qty_t", "price_t"]].values
    movements = index_function(df)
    indexes = movements["index"].cumprod()
    #above results in a series with no period information
    indexes = pd.DataFrame({"period":movements["period"], "index":indexes})
    #need to add back first period
    indexes = pd.concat([indexes, pd.DataFrame({"period":first_period, "index":1},index=[1])],
                        axis=0).sort_values("period")

    #should make periods into the index rather than a column
    #eaier joining?
    indexes = clean_up(indexes, index_name)

    #add a chained suffix
    indexes = indexes.add_suffix("_chained")

    return indexes


#calculates a direct bilateral index
#require specification of a bilateral index function
#currently the following are implemented: laspeyres_index, paasche_index, fisher_index, torqnvist_index
#also takes base_period as an optional keyword arguement
#if not specified, uses first period as base period
def direct_bilateral(df, index_function, index_name=None, base_period = None, periods=None):
    #avoid modifying orignal
    #there might be a better way
    df = df.copy()
    periods, first_period, last_period = extract_periods(df["period"],periods)

    #if base_period is not provided, make it
    if base_period is None:
        base_period = first_period

    #Start by creating dataframe of base period observations
    df_base = df.loc[df.period == base_period, :]

    df = df.merge(df_base, how = 'inner',
                    on="prod_id",
                    suffixes =('_t','_b'))

    #rename
    df = df.rename(columns={"period_t":"period"})

    indexes = index_function(df)
    #should make periods into the index rather than a column
    #easier joining?
    indexes = clean_up(indexes, index_name)

    #add a direct suffix
    #add a chained suffix
    indexes = indexes.add_suffix("_direct")

    return indexes

def laspeyres_index(df):
    #start by renaming current and previous sales
    #should move to earlier ...or not do....
    df = df.rename(columns = {'sales_t': 'ptqt', 'sales_b' : 'pbqb'})

    #calc Laspeyres numerator
    df["ptqb"] = df["price_t"] * df["qty_b"]

    df = df.groupby("period", as_index=False)["ptqb", "pbqb"].sum()

    #now calc indices
    df["index"] = df["ptqb"] / df["pbqb"]

    #now keep only what's neceasry
    df = df.loc[:, ["period","index"]]
    #df = df.sort_values("period")

    return df

def paasche_index(df):
    #start by renaming current and previous sales
    #should move to earlier ...or not do....
    df = df.rename(columns = {'sales_t': 'ptqt', 'sales_b' : 'pbqb'})

    #calc Paasche denominator
    df["pbqt"] = df["price_b"] * df["qty_t"]

    df = df.groupby("period", as_index=False)["pbqt", "ptqt"].sum()

    #now calc indices
    df["index"] = df["ptqt"] / df["pbqt"]

    #now keep only what's neceasry
    df = df.loc[:, ["period","index"]]
    #df = df.sort_values("period")

    return df

def jevons_index(df):
    #start by renaming current and previous sales
    #should move to earlier ...or not do....
    df = df.rename(columns = {'sales_t': 'ptqt', 'sales_b' : 'pbqb'})

    #log relatives
    df["log_rel"] = np.log(df["price_t"] / df["price_b"])

    #mean of log relatives by period
    df = df.groupby("period", as_index=False)["log_rel"].mean()

    #now rename log_rel to index
    df = df.rename(columns = {'log_rel': 'index'})
    #need xponted
    df["index"] = np.exp(df["index"] )

    return df

def tornqvist_index(df):
    #start by renaming current and previous sales
    #should move to earlier ...or not do....
    df = df.rename(columns = {'sales_t': 'ptqt', 'sales_b' : 'pbqb'})

    #calc Paasche denominator
    df["pbqt"] = df["price_b"] * df["qty_t"]

    #relatives (log) for Tornqivst
    df["log_rel"] = np.log(df["price_t"] / df["price_b"])
    #now want shares
    df["st"] = df.groupby("period")["ptqt"].transform(share)
    df["sb"] = df.groupby("period")["pbqb"].transform(share)
    df["st_sb"] = (df["st"] + df["sb"]) / 2
    df["log_rel_wghtd"] = df["st_sb"] * df["log_rel"]

    df = df.groupby("period", as_index=False)["log_rel_wghtd"].sum()

    #tornqvist (needs to epxontiated)
    df["index"] = np.exp(df.log_rel_wghtd)
    #now keep only what's neceasry
    df = df.loc[:, ["period","index"]]
    return df

def fisher_index(df):
    laspeyres_indexes = laspeyres_index(df)
    paasche_indexes = paasche_index(df)

    #calcl fisher as geo mean
    fisher_indexes=(laspeyres_indexes["index"] * paasche_indexes["index"])**0.5
    df = pd.DataFrame({"period":laspeyres_indexes["period"], "index":fisher_indexes})
    return df


#calculates Fixed effects index for whole window
#if doing a window, need to subset first
#there is a rolling_multilateral function which takes a calleable (such as this function) and calculates rolling window indexes
#which are then linked together
def FE_index(df, periods=None):
    #copy to avoid modifying original (not very efificent....)
    #maybe this copy is bad...
    df=df.copy()

    #can either pass the periods or have it calculated
    #need some work
    periods, first_period, last_period = extract_periods(df["period"],periods)

    #calc shares
    #MIGHT WANT TO CONSIDER MOVING THIS TO THE AGGREGATE FUNCTION
    #I think that this causes things to tak a lot longer (double?)
    df["share"] = df.groupby("period")["sales"].transform(share)

    #will "sweep" out product effects, and then just need to do regresion on dummies
    #pandas get_dummies will coerce period to string
    #better to do it explicitly and control the format.
    month_dummies = pd.get_dummies(df["period"].dt.strftime("%Y-%m"))
    #print(month_dummies)

    demean_variables = list(month_dummies) + ["ln_price"]
    #print(demean_variables)

    #Now demean
    #first, isolate
    df = pd.concat([month_dummies, df], axis = 1)
    #return df
    #print(df)
    reg_variables = df.loc[:, ["prod_id","share"] + demean_variables ]
    #now multiply everything (variables to demean) by share
    reg_variables.loc[:, demean_variables] = np.multiply(reg_variables.loc[:,demean_variables],
                                                         np.expand_dims(reg_variables.share,1))
    #sum demeaned variables (by identifers)
    reg_variables = reg_variables.groupby("prod_id", sort = False).transform("sum")
    #divide by summed share
    reg_variables.loc[:,:] = np.divide(reg_variables,np.expand_dims(reg_variables["share"],1))

    #now actually demean (subtract weighted mean from original)
    reg_variables.loc[:, demean_variables] = df.loc[:,demean_variables] - reg_variables.loc[:,demean_variables]

    #make dummy for first period euql to 1
    #otherwise, precision issues
    reg_variables.loc[:, demean_variables[0]] = 1

    #return df
    #create weighted least square matrices
    Aw = reg_variables.loc[:,list(month_dummies)].values * np.sqrt(df.share.values[:,np.newaxis])
    Bw = reg_variables["ln_price"].values * np.sqrt(df["share"].values)


    coefs, _, _, _  = np.linalg.lstsq(Aw, Bw)
    indexes = pd.DataFrame({'index':coefs.flatten(), 'period':periods})
    #simplest just to rebase here (before expontiating to avoid overflow)
    #using values for automatic numpy broadcasting
    indexes.loc[:,"index"] = indexes["index"] - indexes.loc[indexes["period"] == first_period,"index"].values
    indexes["index"] = np.exp(indexes["index"])

    return indexes


#lehr index
def lehr_index(df, periods=None):
    #the keyword argumentts can be provided, or will be calculated if not provided
    #wiill not implement calculation at this timje...

    #I really want to avoid modifying original
    #will figure this out later...
    df = df.copy()

    periods, first_period, last_period = extract_periods(df["period"],periods)

    #first step: get average unit price
    df["avg_price"] =  df.groupby("prod_id")["sales"].transform("sum") / df.groupby("prod_id")["qty"].transform("sum")

    #now multily by quantity to get pbqt (base period is average/whole winodw)
    df["pbqt"] = df["avg_price"] * df["qty"]
    df = df.rename(columns={"sales":"ptqt"})[["ptqt","period","pbqt"]]
    df = df.groupby("period", as_index = False).sum()
    df["level"] = df["ptqt"] / df["pbqt"]
    df["index"] = df["level"].values / df.loc[df.period == first_period,"level"].values

    return df.loc[:,["period","index"]]

def gearykhamis_index(df, periods=None, tolerance = 1e-6, max_iter=100):

    #I really want to avoid modifying original
    #will figure out if there is way to reduce the amount of copying later
    df = df.copy()

     #the keyword arguments can be provided, or will be calculated if not provided
    periods, first_period, last_period = extract_periods(df["period"],periods)

    #now calc quantity share for each product (quanity within product, aross periods)
    #share is labeled at that weird w greek letter in paper
    df["share"] =  df.groupby("prod_id")["qty"].transform(share)
    #renaming for convenience
    df = df.rename(columns={"sales":"ptqt"})

    #initial variables for loop
    #initiliaze mse at infinite
    mse = np.inf
    i = 0
    #initialize indexes, all at 0...
    indexes = pd.DataFrame({"index":1.,"period":periods})
    #stop looping once minimal change in indexes (less than specified MSE) or max iterations reached
    while (mse > tolerance) & (i < max_iter):
        #since updating indexes, reassign previous
        prev_indexes = indexes

        df = df.loc[:,["period","price","ptqt","share","prod_id","qty"]]
        #now merge indexes
        df = df.merge(indexes)
        #now adjusted price
        #ajsuted price called v in paper....
        df["adjstd_price"] = (df["price"] * df["share"])/ df["index"]
        df["adjstd_price"] = df.groupby("prod_id",sort=False)["adjstd_price"].transform("sum")


        #now multily by quantity to get pbqt (base period is average/whole winodw)
        df["pbqt"] = df["adjstd_price"] * df["qty"]

        indexes = df.groupby("period", as_index = False).sum()
        indexes["level"] = indexes["ptqt"] / indexes["pbqt"]
        indexes["index"] = indexes["level"].values / indexes.loc[indexes.period == first_period,"level"].values

        indexes = indexes.loc[:,["period","index"]]
        indexes["prev_index"] = prev_indexes["index"]
        indexes["deviation"] = indexes["index"] - indexes["prev_index"]

        #MSE for stopping criteria
        #not actually mean, just sum
        #sum means that there is less deviation/chage, this is good.
        mse = (indexes["deviation"] ** 2).sum()
        i+=1
    indexes = indexes.loc[:,["period","index"]]

    return indexes

#GEKS index
def GEKS_index(df, periods=None, bilateral_index="tornqvist"):
    #I really want to avoid modifying original
    #will figure this out later...
    df = df.copy()

    #if "tornqvist" specified (default), call the correct index
    #otherwise bilateral_index is expected to be a calleable
    if bilateral_index=="tornqvist":
        bilateral_index=tornqvist_index

    #will not extract unique if not necessary (can save a little time..)
    periods, first_period, last_period = extract_periods(df["period"],periods)

    #merge, just on prod_id
    df = df.merge(df, how='inner', on='prod_id', suffixes=('_t','_b'))
    #rename period_t to period
    df.rename(columns={'period_t':'period'},inplace=True)


    #can use bilateral....
    df = df.groupby("period_b").apply(bilateral_index).reset_index()

    df = df.loc[:,["period","period_b","index"]]
    #log to calc geo mean
    df["index"] = np.log(df["index"])
    df = df.groupby("period", as_index=False)["index"].mean()
    df["index"] = np.exp(df["index"])

    #now rebase
    #using values to get broadcasting
    df["index"] = df["index"] / df.loc[df["period"]==first_period,"index"].values

    return df

#Calculates rolling window mulilateral index
#this is a kind of wrapper
#it takes the same arguments as the other index function so far
#in addiiton, it takes the index_function and window_length variables
#index_function should be a calleable, (index function obvs), that takes the same arguments
#should return a dataframe, consisting of period, the by vrables (if any) and a SINGLE index column (can be named whatever)
#kwargs are passed to the index_function
def rolling_multilateral(df,
             index_function, index_name=None,
             periods=None,
             window_length=None,
             **kwargs):

    #SHOULD MAKE IT SO THAT IF WINDOW LENGHT IS NONE, WHOLE WINDOW IS USED
    #AND SPLICING IS NOT APPLIED?
    #AND CHANGE NAME FROM rolling_multilateral TO

    #can either pass the periods or have it calculated
    #need some work
    periods, first_period, last_period = extract_periods(df["period"],periods)


    #if window_length is not supplied or is equal to number of periods, just do muliteral, no window
    #need to raise an error if window_length > periods.shape[0]
    if window_length is None or window_length == periods.shape[0]:
        indexes = index_function(df, **kwargs)
        indexes = clean_up(indexes, index_name)
        return indexes

    #possible start of windows, exclude last 12 months (can't get a thirteen month window)
    #might be better to define a window by its end rather than beginning
    #assuming that periods is sorted
    first_periods = periods[0:(periods.shape[0]-window_length+1)]
    for period_indx, first_period in enumerate(first_periods):
        #since python slices are closed on right, but open on left
        #period indx is included (first peiriod)
        window_periods = periods[period_indx:(period_indx+window_length)]
        last_period = window_periods[-1]

        window_indexes = index_function(df.loc[df["period"].isin(window_periods),:],
                                              window_periods, **kwargs)
        #add period index, is better
        window_indexes["period_indx"] = np.arange(0, window_length, dtype=int)

        if period_indx == 0: #if first window, want indexes for whole window
            first_window = window_indexes.copy()

            #get names of index
            #ASSUMING THAT THERE iS ONLY A SINLGE COLUMN NAMED index
            #SO WILL FAIL WITH CURRENT IMPLEMENTATIONNS OF DIRECT AND CHAINED INDEXES...
            #need to consider if that matters

            #for first windows, all three indexes the same (since nothing spliced yet)
            first_window["index_windowSplice"] = first_window["index"]
            first_window["index_latestSplice"] = first_window["index"]
            first_window["index_averageSplice"] = first_window["index"]
            first_window = first_window.drop(["index","period_indx"], axis = 1)

            results = [first_window]

            #add a period_indx column, going from 0 to window_len-1
            #makes things easier
            #might want to consider something similar for chained indexes
            #(dict mapping from date_time to period index might be faster) (maybe)

            #make first window levels into splicing factor
            #just need to remove first (since first is equal to one and does not overlap with next window)
            splices = window_indexes.loc[window_indexes["period_indx"] != 0,
                                            ["period_indx", "index"]]
            #shift down
            splices["period_indx"] = splices["period_indx"] -1
            #now rename
            splices = splices.rename(columns={'index':'splice_factor'})
            splices["splice_factor"] = np.log(splices["splice_factor"])
            splice_list = [splices]
        else:
            #dropping period since will use period_indx
            #I wonder if period_indx should be used in the chained index
            #might speed things up slightly...
            window_indexes = window_indexes.drop("period", axis = 1)

            #going with log levels (and movements)
            window_indexes["index"] = np.log(window_indexes["index"])

            #need movements
            #want movement of next period to be associated with current period
            #this is because next window's splice factor is calculatd using this movement
            window_indexes["movement"] = -window_indexes["index"].diff(-1)

            #this way it broadcasts
            #need botht the loc asignment statement on LSH and the values on RHS
            window_indexes.loc[:,"index_num"] = window_indexes.loc[window_indexes.period_indx == window_length-1, "index"].values
            #don't need to keep latest period
            window_indexes = window_indexes[window_indexes.period_indx != window_length-1]

            #now rebase
            #however should already be rebased due to way i set bias feature...
            window_indexes["index"] = window_indexes["index_num"] - window_indexes["index"]

            #now merge on splices
            indexes = window_indexes.merge(splices, on="period_indx")

            #now apply splice log level
            indexes["index_spliced"] = indexes["index"] + indexes["splice_factor"]

            #now extract different links....
            #might need to take geomean here....
            #not sure...
            #if so......................Dunno
            index_average = np.exp(indexes["index_spliced"].mean())
            #must be comparing to second level from last period
            index_window = np.exp(indexes[indexes["period_indx"] == 0])
            #this is splice of latest movement
            index_latest = np.exp(indexes[indexes["period_indx"]== window_length-2])


            inter = pd.DataFrame({"index_averageSplice":index_average,
                                 "index_windowSplice":index_window["index_spliced"].values,
                                 "index_latestSplice":index_latest["index_spliced"].values})
            inter["period"] = last_period
            #for some reason not getting name correct
            results.append(inter)

            #now new splices
            #this is lmovement + splice
            indexes["splice_factor"] = indexes["movement"] + indexes["splice_factor"]
            splices = indexes.loc[:,["splice_factor", "period_indx"]]

            #for checking
            splice_list += [splices]


    indexes = pd.concat(results,axis = 0)
    #should make periods into the index rather than a column
    #easier joining?
    indexes = clean_up(indexes, index_name)
    return indexes

#for working with grouped
def index_func(df, compare_method, index_function, index_name, **kwargs):
    return df.apply(func=compare_method, index_function=index_function, index_name = index_name, **kwargs)