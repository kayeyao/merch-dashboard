import warnings
import pandas as pd
import streamlit as st
from streamlit import caching

import appbody as body

#st.set_page_config(layout="wide")
st.sidebar.title('Merch Dashboard')

add_selectbox = st.sidebar.radio('Select Dashboard',
    ("Sales","Inventory","Inventory Aging","Prices")
)

year = 2021

if add_selectbox == 'Sales':
    st.title("Sales Dashboard")
    st.markdown('<div style="color: #efede8;">.</div>',unsafe_allow_html=True) # space #

    body.weekly_nmv_plot(year)
    body.sales_plot(year)
    body.sales_plot2(year)
    body.top_brands_w_avenmv(year)
    body.top_brands(year)
    body.top_cat_perbrand(year)
    
if add_selectbox == 'Inventory':
    st.title("Inventory Dashboard")
    st.markdown('<div style="color: #efede8;">.</div>',unsafe_allow_html=True) # space #

    body.inv_topbrands(year)
    body.inv_topbrands_percent(year)
    body.inv_percat(year)
    body.inv_percat_percent(year)

if add_selectbox == 'Inventory Aging':
    st.title("Inventory Aging Dashboard")
    st.markdown('<div style="color: #efede8;">.</div>',unsafe_allow_html=True) # space #    

    body.inv_agebands_cat(year)
    body.inv_agebands_cat_percent(year)
    
if add_selectbox == 'Prices':
    st.title("Prices Dashboard")
    st.markdown('<div style="color: #efede8;">.</div>',unsafe_allow_html=True) # space #

    body.price_bands(year)
    body.price_bands_nmv(year)