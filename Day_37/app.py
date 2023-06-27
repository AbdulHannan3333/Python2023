import pandas as pd
import streamlit as st
import pandas_profiling
import seaborn as sns

from streamlit_pandas_profiling import st_profile_report

df = sns.load_dataset('titanic')
pr = df.profile_report()
st_profile_report(pr) # streamlit k app k andar run kar dia