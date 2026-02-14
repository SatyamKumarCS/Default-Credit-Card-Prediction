def add_feature(df):
    bill_cols=[f'BILL_AMT{i}' for i in range(1,7)]
    df['AVG_BILL_AMT']=df[bill_cols].mean(axis=1)

    df['CREDIT_UTILITY']=df['AVG_BILL_AMT']/df['LIMIT_BAL']

    pay_col=[f'PAY_AMT{i}' for i in range(1,7)]
    df['AVG_PAY_AMT']=df[pay_col].mean(axis=1)
    
    pay_delay_col=["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df['AVG_PAY_DELAY']=df[pay_delay_col].clip(lower=0).mean(axis=1)
    
    df['PAYMENT_TO_BILL']=(df['AVG_PAY_AMT']/(df['AVG_BILL_AMT']+1)).fillna(0)

    pay_delay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df["MAX_PAY_DELAY"] = df[pay_delay_cols].clip(lower=0).max(axis=1)  
    
    df["NUM_LATE_MONTHS"] = (df[pay_delay_cols] > 0).sum(axis=1)

    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    df["PAYMENT_STD"] = df[pay_amt_cols].std(axis=1)

    df["SEVERE_DELAY_FLAG"] = (df["MAX_PAY_DELAY"] >= 3).astype(int)

    return df