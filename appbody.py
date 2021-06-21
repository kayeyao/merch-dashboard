import warnings
import pandas as pd
import streamlit as st
import numpy as np
from streamlit import caching
from streamlit import caching

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

sales_df = pd.read_csv('Sales_210621.csv')
inventory_df = pd.read_csv('Inventory_210621.csv')
prices_df = pd.read_csv('Pricing_210621.csv')

fill_values = {'Department Code': '','Brand Name':'','Buying Planning Cat Type':'','Sub Category Type':'', 'Spot Aging Band':0,
    'GMV (€) static':0, 'NMV after Provision (€)':0, 'PC1 provisioned static fx':0, 'Discount (€)':0, 'Markdown (€)':0, 
    'Promo (€)':0, 'Cart Rule Discount (€)':0,'Spot Age w Threshold (bin)':0, 'items sold':0, 'cost of sales':0, 
    'soh cost':0, 'nmv':0, 'soh units': 0, 'Width (Visible)':0, 'inbounded stock units':0, 'inbounded cost':0,  'Discount':0, 
    'Sales Markdown':0, 'Promo':0, 'gmv':0, 'Dynamic Online Age w threshold (bin)':0, 'Price':0, 'Current Price':0,'nmv_eur_after_return':0,     'items_sold':0}

sales_df = sales_df.fillna(value = fill_values)
inventory_df = inventory_df.fillna(value = fill_values)
prices_df = prices_df.fillna(value = fill_values)

sales_df['GMV'] = sales_df['GMV (€) static'].str.replace(',','').astype(float)
sales_df['NMV'] = sales_df['NMV after Provision (€)'].str.replace(',','').astype(float)
sales_df['PC1'] = sales_df['PC1 provisioned static fx'].str.replace(',','').astype(float)
sales_df['Discount'] = sales_df['Discount (€)'].str.replace(',','').astype(float)
sales_df['Markdown'] = sales_df['Markdown (€)'].str.replace(',','').astype(float)
sales_df['Promo'] = sales_df['Promo (€)'].str.replace(',','').astype(float)
sales_df['Cart Rule'] = sales_df['Cart Rule Discount (€)'].str.replace(',','').astype(float)
sales_df['Spot Age'] = sales_df['Spot Aging Band'].astype(str)
sales_df['Brand Name'] = sales_df['Brand Name'].str.upper() 

inventory_df['Items Sold'] = inventory_df['items sold'].astype(str).str.replace(',','').astype(float)
inventory_df['Cost of Sales'] = inventory_df['cost of sales'].str.replace(',','').astype(float)
inventory_df['SOH Cost'] = inventory_df['soh cost'].str.replace(',','').astype(float)
inventory_df['SOH Units'] = inventory_df['soh units'].str.replace(',','').astype(float)
inventory_df['NMV'] = inventory_df['nmv'].str.replace(',','').astype(float)
inventory_df['GMV'] = inventory_df['gmv'].str.replace(',','').astype(float)
inventory_df['Spot Age'] = inventory_df['Spot Age w Threshold (bin)'].astype(float).astype(int).astype(str)
inventory_df['Width'] = inventory_df['Width (Visible)'].str.replace(',','').astype(float)
inventory_df['Inbound Units'] = inventory_df['inbounded stock units'].str.replace(',','').astype(float)
inventory_df['Discount'] = inventory_df['Discount'].astype(str).str.replace(',','').astype(float)
inventory_df['Markdown'] = inventory_df['Sales Markdown'].str.replace(',','').astype(float)
inventory_df['Promo'] = inventory_df['Promo'].astype(str).str.replace(',','').astype(float)
inventory_df['Retail Year'] = 2021
inventory_df['Brand Name'] = inventory_df['brand_name'].str.upper() 

prices_df['Spot Age'] = prices_df['Dynamic Online Age w threshold (bin)'].astype(float).astype(int).astype(str)
prices_df['Price'] = prices_df['Price'].astype(float)
prices_df['Current Price'] = prices_df['Current Price'].astype(float)
prices_df['Brand Name'] = prices_df['Brand Name'].str.upper() 
prices_df['Items Sold'] = prices_df['items_sold'].str.replace(',','').astype(float)
prices_df['NMV'] = prices_df['nmv_eur_after_return'].str.replace(',','').astype(float)

sales_df = sales_df.drop(columns = ['GMV (€) static',
       'NMV after Provision (€)', 'PC1 provisioned static fx', 'Discount (€)',
       'Markdown (€)', 'Promo (€)', 'Cart Rule Discount (€)'])
inventory_df = inventory_df.drop(columns = ['brand_name', 'items sold', 'cost of sales', 'soh cost', 'soh units', 'nmv', 
            'Width (Visible)', 'inbounded stock units', 'Sales Markdown', 'gmv'])
inventory_df = inventory_df.rename(columns = {'retail_week': 'Retail Week', 'department_code': 'Department Code', 
            'buying_planning_cat_type': 'Buying Planning Cat Type'})
prices_df = prices_df.drop(columns = ['Dynamic Online Age w threshold (bin)', 'nmv_eur_after_return', 'items_sold'])
prices_df = prices_df.rename(columns = {'department_code':'Department Code', 'sub_category_type':'Sub Category Type'})

departments = np.append('All', sales_df['Department Code'].unique())
brands = np.append('All',sales_df['Brand Name'].unique())
categories = np.append('All',sales_df['Buying Planning Cat Type'].unique())
subcategories = np.append('All',sales_df['Sub Category Type'].unique())

def weekly_nmv_plot(year):
    st.subheader(f"{year - 1} VS {year} Weekly NMV, PC1, Discounts")
    
    weekly_sales = sales_df.groupby(['Retail Year','Retail Week'])[['GMV','NMV','PC1','Discount','Markdown','Promo']].sum().reset_index()

    weekly_sales['PC1%'] = weekly_sales['PC1'] / weekly_sales['NMV']
    weekly_sales['Discount%'] = weekly_sales['Discount'] / weekly_sales['GMV']
    weekly_sales['Markdown%'] = weekly_sales['Markdown'] / weekly_sales['GMV']
    weekly_sales['Promo%'] = weekly_sales['Promo'] / weekly_sales['GMV']

    weekly_sales_0 = weekly_sales[weekly_sales['Retail Year'] == (year - 1)]
    weekly_sales_1 = weekly_sales[weekly_sales['Retail Year'] == year]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=weekly_sales_0['Retail Week'], y=weekly_sales_0['NMV'], name=f"NMV {year - 1}"), secondary_y=False)
    fig.add_trace(go.Bar(x=weekly_sales_1['Retail Week'], y=weekly_sales_1['NMV'], name=f"NMV {year}"), secondary_y=False)
    fig.add_trace(go.Scatter(x=weekly_sales_0['Retail Week'], y=weekly_sales_0['PC1%'], name=f"PC1% {year - 1}"), secondary_y=True)
    fig.add_trace(go.Scatter(x=weekly_sales_1['Retail Week'], y=weekly_sales_1['PC1%'], name=f"PC1% {year}"), secondary_y=True)
    fig.add_trace(go.Scatter(x=weekly_sales_0['Retail Week'], y=weekly_sales_0['Discount%'], name=f"Discount% {year - 1}"), secondary_y=True)     
    fig.add_trace(go.Scatter(x=weekly_sales_1['Retail Week'], y=weekly_sales_1['Discount%'], name=f"Discount% {year}"), secondary_y=True)
    
    fig.update_layout(width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))

    fig.update_xaxes(title_text="Retail Week")
    fig.update_yaxes(title_text="Amount in EUR", secondary_y=False, tickformat=".3s")
    fig.update_yaxes(title_text="Percent", secondary_y=True, tickformat = '%', showgrid=False, range = [0,0.6])

    st.plotly_chart(fig)
    
    
def weekly_breakdown(year, brand, cat, subcat):
    variable_list = ['GMV', 'NMV', 'Discount', 'Markdown', 'Promo', 'PC1']
    df = sales_df[sales_df['Retail Year'] == year]
    
    if brand != 'All':
        df = df[df['Brand Name'] == brand]
    if cat != 'All':
        df = df[df['Buying Planning Cat Type'] == cat]
    if subcat != 'All':
        df = df[df['Sub Category Type'] == subcat]
    
    weekly_df = df.groupby('Retail Week')[variable_list].sum().reset_index()
    weekly_df['PC1%'] = weekly_df['PC1'] / weekly_df['NMV']
    weekly_df['Discount%'] = weekly_df['Discount'] / weekly_df['GMV']
    weekly_df['Markdown%'] = weekly_df['Markdown'] / weekly_df['GMV']
    weekly_df['Promo%'] = weekly_df['Promo'] / weekly_df['GMV']

    weekly_nmv = sales_df[sales_df['Retail Year'] == year].groupby('Retail Week')['NMV'].sum()
    
    weekly_df = weekly_df.merge(weekly_nmv, on = 'Retail Week', how = 'left')
    weekly_df['% Weekly NMV'] = weekly_df['NMV_x'] / weekly_df['NMV_y']
    weekly_df = weekly_df.rename(columns = {'NMV_x': 'NMV'})
        
    return weekly_df 


def sales_plot(year):
    st.subheader('Weekly NMV, PC1, Discounts, Markdowns, and Promos')
    
    col1, col2, col3 = st.beta_columns(3)
    
    brand = col1.selectbox('Brand',brands, index = 0, key=0)
    cat = col2.selectbox('Category',categories, index = 0, key=0)
    subcat = col3.selectbox('Subcategory',subcategories, index = 0, key=0)
    
    df = weekly_breakdown(year, brand, cat, subcat)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=df['Retail Week'], y=df['NMV'], name="NMV"), secondary_y=False)
    fig.add_trace(go.Bar(x=df['Retail Week'], y=df['PC1'], name="PC1"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Retail Week'], y=df['Discount%'], name="Discount%"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['Retail Week'], y=df['Markdown%'], name="Markdown%"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['Retail Week'], y=df['Promo%'], name="Promo%"), secondary_y=True)

    fig.update_layout(width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))
    fig.update_xaxes(title_text="Retail Week")
    fig.update_yaxes(title_text="Amount in EUR", secondary_y=False, tickformat=".3s")
    fig.update_yaxes(title_text="Percent", secondary_y=True, tickformat = '%', showgrid=False, range = [0,0.6])

    st.plotly_chart(fig)
    
def sales_plot2(year):
    st.subheader('Weekly NMV, PC1, Discounts')
    
    col1, col2, col3 = st.beta_columns(3)
    
    brand = col1.selectbox('Brand',brands, index = 0, key=1)
    cat = col2.selectbox('Category',categories, index = 0, key=1)
    subcat = col3.selectbox('Subcategory',subcategories, index = 0, key=1)
    
    df = weekly_breakdown(year, brand, cat, subcat)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=df['Retail Week'], y=df['NMV'], name="NMV"), secondary_y=False)
    fig.add_trace(go.Bar(x=df['Retail Week'], y=df['PC1'], name="PC1"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['Retail Week'], y=df['Discount%'], name="Discount%"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['Retail Week'], y=df['PC1%'], name="PC1%"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df['Retail Week'], y=df['% Weekly NMV'], name="% Weekly NMV"), secondary_y=True)

    fig.update_layout(width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))
    
    fig.update_xaxes(title_text="Retail Week")
    fig.update_yaxes(title_text="Amount in EUR", secondary_y=False, tickformat=".3s")
    fig.update_yaxes(title_text="Percent", secondary_y=True, tickformat = '%', showgrid=False, range = [0,0.6])

    st.plotly_chart(fig)
    
def top_brands_w_avenmv(year):
    df = sales_df[sales_df['Retail Year'] == year]
    df_0 = sales_df[sales_df['Retail Year'] == year - 1]
    latest_week = df['Retail Week'].max()
    latest_month = [latest_week, latest_week - 1, latest_week - 2, latest_week - 3]
    
    st.subheader(f'Top Brands - Week {latest_week}')
    
    col1, col2, col3 = st.beta_columns(3)
    
    ave_duration = col1.selectbox('Duration for Weekly Average', ('MTD','YTD'), index = 0, key=2)
    
    brand_df = df[df['Retail Week'] == latest_week].groupby('Brand Name')['NMV'].sum().reset_index()  
    brand_df = brand_df.sort_values('NMV', ascending=True).tail(15).reset_index()        
        
    brand_df[f'Average Weekly NMV {year}'] = 0
    brand_df[f'Average Weekly NMV {year - 1}'] = 0
    
    if ave_duration == 'YTD':
        for i in range(len(brand_df)):
            brand_df[f'Average Weekly NMV {year}'][i] = df[df['Brand Name'] == brand_df['Brand Name'][i]]['NMV'].sum()/latest_week
            brand_df_0 = df_0[df_0['Brand Name'] == brand_df['Brand Name'][i]]
            brand_df[f'Average Weekly NMV {year - 1}'][i] = brand_df_0['NMV'].sum()/(brand_df_0['Retail Week'].max()-brand_df_0['Retail Week'].min())
    
    elif ave_duration == 'MTD':
        for i in range(len(brand_df)):
            brand_df[f'Average Weekly NMV {year}'][i] = df[(df['Brand Name'] == brand_df['Brand Name'][i]) & (df['Retail Week'].isin(latest_month))]['NMV'].sum()/4
            brand_df[f'Average Weekly NMV {year - 1}'][i] = df_0[(df_0['Brand Name'] == brand_df['Brand Name'][i]) & (df_0['Retail Week'].isin(latest_month))]['NMV'].sum()/4
                 
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    fig.add_trace(go.Bar(y=brand_df['Brand Name'], x=brand_df[f'Average Weekly NMV {year - 1}'], name=f"NMV Weekly Average ({year-1} {ave_duration})", orientation='h'))  
    fig.add_trace(go.Bar(y=brand_df['Brand Name'], x=brand_df[f'Average Weekly NMV {year}'], name=f"NMV Weekly Average ({year} {ave_duration})", orientation='h'))
    fig.add_trace(go.Bar(y=brand_df['Brand Name'], x=brand_df['NMV'], name=f"NMV Week {latest_week}", orientation='h'))

    fig.update_xaxes(title_text="NMV in EUR")
    fig.update_yaxes(title_text = "Brand Name", tickfont={'size':12}, tickformat=".3s")
    fig.update_layout(legend_traceorder="reversed", width=900, height=400, margin=dict(l=0, r=0, t=20, b=0))

    st.plotly_chart(fig)
    
def top_brands(year):
    df = sales_df[sales_df['Retail Year'] == year]
    df_0 = sales_df[sales_df['Retail Year'] == year - 1]
    latest_week = df['Retail Week'].max()
    latest_month = [latest_week, latest_week - 1, latest_week - 2, latest_week - 3]  
    
    st.subheader(f'Top Brands')
    
    col1, col2, col3 = st.beta_columns(3)
    
    duration = col1.selectbox('Duration', ('Past Week','Past Month','YTD'), index = 0, key=3)
    
    if duration == 'Past Week':
        brand_df = df[df['Retail Week'] == latest_week]
        brand_df_0 = df_0[(df_0['Retail Week'] == latest_week)]
        label = f"Week {latest_week}"
        
    elif duration == 'Past Month':
        brand_df = df[df['Retail Week'].isin(latest_month)]
        brand_df_0 = df_0[(df_0['Retail Week'].isin(latest_month))]
        label = f"Weeks {latest_week - 3} - {latest_week}"

    elif duration == 'YTD':
        brand_df = df[df['Retail Week'] <= latest_week]
        brand_df_0 = df_0[df_0['Retail Week'] <= latest_week]
        label = duration
    
    brand_df = brand_df.groupby('Brand Name')['NMV'].sum().reset_index()
    brand_df_0 = brand_df_0.groupby('Brand Name')['NMV'].sum().reset_index()
    brand_df = brand_df.sort_values('NMV', ascending=True).tail(15)
    brand_df = brand_df.merge(brand_df_0, on = 'Brand Name', how = 'left')

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    fig.add_trace(go.Bar(y=brand_df['Brand Name'], x=brand_df['NMV_y'], name=f"NMV {label} ({year-1})", orientation='h'))
    fig.add_trace(go.Bar(y=brand_df['Brand Name'], x=brand_df['NMV_x'], name=f"NMV {label} ({year})", orientation='h'))  

    fig.update_xaxes(title_text="NMV in EUR", tickformat=".3s")
    fig.update_yaxes(title_text = "Brand Name", tickfont={'size':12})
    fig.update_layout(legend_traceorder="reversed",  width=900, height=400, margin=dict(l=0, r=0, t=20, b=0))

    st.plotly_chart(fig)

def top_cat_perbrand(year):
    st.subheader(f'Top Categories per Brand')
    
    col1, col2,col3 = st.beta_columns(3)
    
    brand = col1.selectbox('Brand', brands, index=0, key=4)
    subtype = col2.selectbox('Subtype', ('Category','Subcategory'), index=0, key=4)
    duration = col3.selectbox('Duration', ('Past Week','Past Month','YTD'), index = 0, key=4)
    
    if brand == "All":
        df = sales_df[(sales_df['Retail Year'] == year)]
        df_0 = sales_df[(sales_df['Retail Year'] == (year - 1))]
    else:    
        df = sales_df[(sales_df['Retail Year'] == year) & (sales_df['Brand Name'] == brand)]
        df_0 = sales_df[(sales_df['Retail Year'] == (year -1)) & (sales_df['Brand Name'] == brand)]
    
    latest_week = df['Retail Week'].max()
    latest_month = [latest_week, latest_week - 1, latest_week - 2, latest_week - 3]
    
    if subtype == 'Category':
        category = 'Buying Planning Cat Type'
    elif subtype == 'Subcategory':
        category = "Sub Category Type"
        
    if duration == 'Past Week':
        brand_df = df[(df['Retail Week'] == latest_week)]
        brand_df_0 = df_0[(df_0['Retail Week'] == latest_week)]
        label = f"Week {latest_week}"
    elif duration == 'Past Month':
        brand_df = df[(df['Retail Week'].isin(latest_month))]
        brand_df_0 = df_0[(df_0['Retail Week'].isin(latest_month))]
        label = f"Weeks {latest_week - 3} - {latest_week}"
    elif duration == 'YTD':
        brand_df = df[df['Retail Week'] <= latest_week]
        brand_df_0 = df_0[df_0['Retail Week'] <= latest_week]
        label = duration
    
    brand_df = brand_df.groupby(category)['NMV'].sum().reset_index()
    brand_df_0 = brand_df_0.groupby(category)['NMV'].sum().reset_index()
    brand_df = brand_df.sort_values('NMV', ascending=True).tail(15)
    brand_df = brand_df.merge(brand_df_0, on = category, how = 'left')
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])
    
    fig.add_trace(go.Bar(y=brand_df[category], x=brand_df['NMV_y'], name=f"NMV {label} ({year-1})", orientation='h'))
    fig.add_trace(go.Bar(y=brand_df[category], x=brand_df['NMV_x'], name=f"NMV {label} ({year})", orientation='h'))  

    fig.update_xaxes(title_text="NMV in EUR", tickformat=".3s")
    fig.update_yaxes(title_text = f"{subtype}", tickfont={'size':12})
    fig.update_layout(legend_traceorder="reversed", width=900, height=400, margin=dict(l=0, r=0, t=20, b=0))

    st.plotly_chart(fig)


def inv_topbrands(year):
    latest_week = sales_df[sales_df['Retail Year'] == 2021]['Retail Week'].max()
    
    st.subheader(f'Top Brands - NMV, SOH Units, Cost of Sales - Week {latest_week}')
    
    col1, col2, col3 = st.beta_columns(3)
    
    dept = col1.selectbox('Department', departments, index = 0, key=5)
    
    if dept == "All":
        df = inventory_df
    else:
        df = inventory_df[inventory_df['Department Code'] == dept]
    
    brand_df = df[(df['Retail Week'] == latest_week)].groupby('Brand Name')[['NMV', 'Cost of Sales','SOH Units', 'SOH Cost']].sum().reset_index()
    brand_df['STR%'] = brand_df['Cost of Sales']/(brand_df['Cost of Sales'] + brand_df['SOH Cost'])
    brand_df = brand_df.sort_values('NMV', ascending=False).head(15)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=brand_df['Brand Name'], y=brand_df['SOH Units'], name="SOH Units"), secondary_y = False)
    fig.add_trace(go.Bar(x=brand_df['Brand Name'], y=brand_df['NMV'], name="NMV"), secondary_y = False)
    fig.add_trace(go.Bar(x=brand_df['Brand Name'], y=brand_df['Cost of Sales'], name="Cost of Sales"), secondary_y = False)
    fig.add_trace(go.Scatter(x=brand_df['Brand Name'], y=brand_df['STR%'], name="STR%"), secondary_y = True)
        
    fig.update_yaxes(title_text = "Amount in EUR", secondary_y = False, rangemode = 'tozero', tickformat=".3s")
    fig.update_yaxes(title_text = "Percent", secondary_y = True, tickformat = '%', showgrid=False, rangemode = 'tozero')
    fig.update_xaxes(title_text = f"Brand Name", tickfont={'size':10})
    fig.update_layout(width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))
    
    st.plotly_chart(fig)
    
def inv_topbrands_percent(year):
    latest_week = sales_df[sales_df['Retail Year'] == 2021]['Retail Week'].max()
 
    st.subheader(f'Top Brands - % Contribution of NMV, SOH Units, Items Sold - Week {latest_week}')
    
    col1, col2, col3 = st.beta_columns(3)
    
    dept = col1.selectbox('Department', departments, index = 0, key=6)   
    
    if dept == "All":
        df = inventory_df
    else:
        df = inventory_df[inventory_df['Department Code'] == dept]
        
    brand_df = df[(df['Retail Week'] == latest_week)].groupby('Brand Name')[['NMV','SOH Units','Items Sold','Width']].sum().reset_index()
    brand_df['%NMV'] = brand_df['NMV']/brand_df['NMV'].sum()
    brand_df['%SOH Units'] = brand_df['SOH Units']/brand_df['SOH Units'].sum()
    brand_df['%Items Sold'] = brand_df['Items Sold']/brand_df['Items Sold'].sum()
    brand_df['%Width'] = brand_df['Width']/brand_df['Width'].sum()

    brand_df = brand_df.sort_values('%NMV', ascending=False).head(15)
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(x=brand_df['Brand Name'], y=brand_df['%SOH Units'], name="SOH Units"), secondary_y = False)
    fig.add_trace(go.Scatter(x=brand_df['Brand Name'], y=brand_df['%NMV'], name="NMV"), secondary_y = False)
    fig.add_trace(go.Scatter(x=brand_df['Brand Name'], y=brand_df['%Items Sold'], name="Items Sold"), secondary_y = False)
    fig.add_trace(go.Scatter(x=brand_df['Brand Name'], y=brand_df['%Width'], name="Width"), secondary_y = False)
                      
    fig.update_yaxes(title_text = "Percent", secondary_y = False, tickformat = '%', rangemode = 'tozero')
    fig.update_xaxes(title_text = f"Brand Name", tickfont={'size':10}, showgrid=False)
    fig.update_layout(width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))
    
    st.plotly_chart(fig)
    
def inv_percat(year):
    latest_week = sales_df[sales_df['Retail Year'] == year]['Retail Week'].max()
    
    st.subheader(f'Top Categories - NMV, SOH Units, Items Sold - Week {latest_week}')
    
    col1, col2, col3 = st.beta_columns(3)
    
    dept = col1.selectbox('Department', departments, index = 0, key=7)
    brand = col2.selectbox('Brand', brands, index = 0, key=7)   
    subtype = col3.selectbox('Subtype', ('Category', 'Subcategory'), index = 0, key=7)   
     
    df = inventory_df
    
    if dept != 'All':
        df = df[df['Department Code'] == dept]
    if brand != "All":
        df = inventory_df[(inventory_df['Brand Name'] == brand)]
    
    if subtype == 'Category':
        category = 'Buying Planning Cat Type'
    elif subtype == 'Subcategory':
        category = "Sub Category Type"

    brand_df = df[(df['Retail Week'] == latest_week)].groupby(category)[['NMV', 'Cost of Sales','SOH Units', 'SOH Cost', 'Discount','Markdown','Promo']].sum().reset_index()
    brand_df['STR%'] = brand_df['Cost of Sales']/(brand_df['Cost of Sales'] + brand_df['SOH Cost'])

    brand_df = brand_df.sort_values('NMV', ascending=False).head(15)
    
    inv_df_2021_start = df[df['Retail Week']==latest_week].groupby(category)['SOH Cost'].sum()
    inv_df_2021 = df[df['Retail Week'].isin(range(latest_week))].groupby(category)[['NMV', 'Cost of Sales']].sum().reset_index()
    inv_df_2021 = inv_df_2021.merge(inv_df_2021_start, on = category, how = 'left')
    inv_df_2021['STR% 2021'] = inv_df_2021['Cost of Sales']/(inv_df_2021['Cost of Sales'] + inv_df_2021['SOH Cost'])
                                                                                                                                                    
    brand_df = brand_df.merge(inv_df_2021[[category, "STR% 2021"]], on = category, how='left')    
        
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=brand_df[category], y=brand_df['SOH Units'], name="SOH Units"), secondary_y = False)
    fig.add_trace(go.Bar(x=brand_df[category], y=brand_df['NMV'], name="NMV"), secondary_y = False)
    fig.add_trace(go.Bar(x=brand_df[category], y=brand_df['Cost of Sales'], name="Cost of Sales"), secondary_y = False)
    fig.add_trace(go.Scatter(x=brand_df[category], y=brand_df['STR%'], name=f"STR% Week {latest_week}"), secondary_y = True)
    fig.add_trace(go.Scatter(x=brand_df[category], y=brand_df['STR% 2021'], name="STR% 2021"), secondary_y = True)
                      
    fig.update_yaxes(title_text = "Amount in EUR", secondary_y = False, tickformat=".3s")
    fig.update_yaxes(title_text = "Percent", showgrid = False, secondary_y = True, tickformat = '%', range = [0,1])
    fig.update_xaxes(title_text = f"{subtype}", tickfont={'size':10})
    fig.update_layout(width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))
    
    st.plotly_chart(fig)
                             
def inv_percat_percent(year):
    latest_week = sales_df[sales_df['Retail Year'] == 2021]['Retail Week'].max()
    
    st.subheader(f'Top Categories - % Contribution of NMV, SOH Units, Items Sold - Week {latest_week}')
    
    col1, col2, col3 = st.beta_columns(3)
    
    dept = col1.selectbox('Department', departments, index = 0, key=8)
    brand = col2.selectbox('Brand', brands, index = 0, key=8)   
    subtype = col3.selectbox('Subtype', ('Category', 'Subcategory'), index = 0, key=8)  
                             
    df = inventory_df    
    
    if dept != "All":
        df = df[df['Department Code'] == dept]
    if brand != "All":
        df = df[(df['Brand Name'] == brand)]
        
    if subtype == 'Category':
        category = 'Buying Planning Cat Type'
    elif subtype == 'Subcategory':
        category = "Sub Category Type"

    brand_df = df[(df['Retail Week'] == latest_week)].groupby(category)[['NMV','SOH Units','Items Sold','Width']].sum().reset_index()
    brand_df['%NMV'] = brand_df['NMV']/brand_df['NMV'].sum()
    brand_df['%SOH Units'] = brand_df['SOH Units']/brand_df['SOH Units'].sum()
    brand_df['%Items Sold'] = brand_df['Items Sold']/brand_df['Items Sold'].sum()
    brand_df['%Width'] = brand_df['Width']/brand_df['Width'].sum()
    
    brand_df = brand_df.sort_values('%NMV', ascending=False).head(15)
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(x=brand_df[category], y=brand_df['%SOH Units'], name="SOH Units"), secondary_y = False)
    fig.add_trace(go.Scatter(x=brand_df[category], y=brand_df['%NMV'], name="NMV"), secondary_y = False)
    fig.add_trace(go.Scatter(x=brand_df[category], y=brand_df['%Items Sold'], name="Items Sold"), secondary_y = False)
    fig.add_trace(go.Scatter(x=brand_df[category], y=brand_df['%Width'], name="Width"), secondary_y = False)
                      
    fig.update_yaxes(title_text = "Percent", secondary_y = False, tickformat = '%', rangemode = 'tozero')
    fig.update_xaxes(title_text = f"{subtype}", tickfont={'size':10})
    fig.update_layout(width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))
    
    st.plotly_chart(fig)

    
def inv_agebands_cat(year):
    latest_week = sales_df[sales_df['Retail Year'] == year]['Retail Week'].max()
    st.subheader(f'NMV, SOH, STR%, Discount% by Age Band')
    
    col1, col2, col3, col4 = st.beta_columns(4)
    
    dept = col1.selectbox('Department', departments, index = 0, key=9)
    brand = col2.selectbox('Brand', brands, index = 0, key=9)   
    cat = col3.selectbox('Category', categories, index = 0, key=9)  
    subcat = col4.selectbox('Subcategory', subcategories, index = 0, key=9)  
    
    df = inventory_df[inventory_df['Retail Year'] == year]
    
    if dept != 'All':
        df = df[df['Department Code'] == dept]
    if brand != 'All':
        df = df[df['Brand Name'] == brand]
    if cat != 'All':
        df = df[df['Buying Planning Cat Type'] == cat]
    if subcat != 'All':
        df = df[df['Sub Category Type'] == subcat]
    
    custom_dict = {'0':0, '30':1, '60':2,'90':3,'120':4,'150':5,'180':6} 

    brand_df = df[(df['Retail Week'] == latest_week)].groupby('Spot Age')[['GMV', 'NMV', 'Cost of Sales','SOH Cost','SOH Units', 'Discount', 'Markdown','Promo']].sum().reset_index()
    brand_df['STR%'] = brand_df['Cost of Sales']/(brand_df['Cost of Sales'] + brand_df['SOH Cost'])
    brand_df['Discount%'] = brand_df['Discount']/brand_df['GMV']
    brand_df['Markdown%'] = brand_df['Markdown']/brand_df['GMV']
    brand_df['Promo%'] = brand_df['Promo']/brand_df['GMV']
    brand_df = brand_df.sort_values(by=['Spot Age'], key=lambda x: x.map(custom_dict))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(x=brand_df['Spot Age'], y=brand_df['SOH Units'], name="SOH Units"))
    fig.add_trace(go.Bar(x=brand_df['Spot Age'], y=brand_df['NMV'], name="NMV"))
    fig.add_trace(go.Bar(x=brand_df['Spot Age'], y=brand_df['Cost of Sales'], name="Cost of Sales"))
    fig.add_trace(go.Scatter(x=brand_df['Spot Age'], y=brand_df['STR%'], name="STR%"), secondary_y = True)
    fig.add_trace(go.Scatter(x=brand_df['Spot Age'], y=brand_df['Discount%'], name="Discount%"), secondary_y = True)
    fig.add_trace(go.Scatter(x=brand_df['Spot Age'], y=brand_df['Markdown%'], name="Markdown%"), secondary_y = True)
    fig.add_trace(go.Scatter(x=brand_df['Spot Age'], y=brand_df['Promo%'], name="Promo%"), secondary_y = True)
        
    fig.update_xaxes(title_text="Age Band", categoryarray = ['0','30','60','90','120','150','180'])
    fig.update_yaxes(title_text = "Amount in EUR", tickfont={'size':10}, tickformat=".3s")
    fig.update_yaxes(title_text = "Percent", showgrid = False, secondary_y = True, tickformat = '%', range = [0,1])
    fig.update_layout(width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))

    st.plotly_chart(fig)
    
def inv_agebands_cat_percent(year):
    st.subheader(f'% Contribution of NMV, SOH, Items Sold, Width by Age Band')
    
    col1, col2, col3, col4 = st.beta_columns(4)
    
    dept = col1.selectbox('Department', departments, index = 0, key=10)
    brand = col2.selectbox('Brand', brands, index = 0, key=10)   
    cat = col3.selectbox('Category', categories, index = 0, key=10)  
    subcat = col4.selectbox('Subcategory', subcategories, index = 0, key=10)  
    
    df = inventory_df[inventory_df['Retail Year'] == year]
    
    if dept != 'All':
        df = df[df['Department Code'] == dept]
    if brand != 'All':
        df = df[df['Brand Name'] == brand]
    if cat != 'All':
        df = df[df['Buying Planning Cat Type'] == cat]
    if subcat != 'All':
        df = df[df['Sub Category Type'] == subcat]
    
    latest_week = sales_df[sales_df['Retail Year'] == year]['Retail Week'].max()
    custom_dict = {'0':0, '30':1, '60':2,'90':3,'120':4,'150':5,'180':6} 

    brand_df = df[(df['Retail Week'] == latest_week)].groupby('Spot Age')[['NMV','SOH Units','Items Sold','Width']].sum().reset_index()
    brand_df['%NMV'] = brand_df['NMV']/brand_df['NMV'].sum()
    brand_df['%SOH Units'] = brand_df['SOH Units']/brand_df['SOH Units'].sum()
    brand_df['%Items Sold'] = brand_df['Items Sold']/brand_df['Items Sold'].sum()
    brand_df['%Width'] = brand_df['Width']/brand_df['Width'].sum()
    
    brand_df = brand_df.sort_values(by=['Spot Age'], key=lambda x: x.map(custom_dict))
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    fig.add_trace(go.Scatter(x=brand_df['Spot Age'], y=brand_df['%SOH Units'], name="SOH Units"))
    fig.add_trace(go.Scatter(x=brand_df['Spot Age'], y=brand_df['%NMV'], name="NMV"))
    fig.add_trace(go.Scatter(x=brand_df['Spot Age'], y=brand_df['%Items Sold'], name="Items Sold"))
    fig.add_trace(go.Scatter(x=brand_df['Spot Age'], y=brand_df['%Width'], name="Width"))

    fig.update_xaxes(title_text="Age Band", categoryarray = ['0','30','60','90','120','150','180'])
    fig.update_yaxes(title_text = "Percent", tickfont={'size':10}, tickformat = '%', rangemode = 'tozero')
    fig.update_layout(width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))

    st.plotly_chart(fig)
    
def price_bands(year):
    st.subheader(f'Price Distribution')
    
    col1, col2, col3 = st.beta_columns(3)
    
    brand = col1.selectbox('Brand', brands, index = 0, key=11)
    
    if brand == 'All':
        df = prices_df
    else:
        df = prices_df[prices_df['Brand Name'] == brand]
    
    latest_week = sales_df[sales_df['Retail Year']==2021]['Retail Week'].max()
#     df = df[df['Retail Week'] == latest_week]


    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['Current Price'], xbins=dict(size=500), name = 'Current Price'))
    fig.add_trace(go.Histogram(x=df['Price'], xbins=dict(size=500), name = 'Original Price'))
                                
    fig.update_layout(xaxis_title_text='Price in PHP', yaxis_title_text='Count', bargap=0.2, width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))

    st.plotly_chart(fig)
    
def price_bands_nmv(year):
    st.subheader(f'Price Distribution of NMV, Items Sold')
    
    col1, col2, col3 = st.beta_columns(3)
    
    brand = col1.selectbox('Brand', brands, index = 0, key=12)
    
    if brand == 'All':
        df = prices_df
    else:
        df = prices_df[prices_df['Brand Name'] == brand]
    
    df['%NMV'] = df['NMV']/df['NMV'].sum()
    df['%Items Sold'] = df['NMV']/df['NMV'].sum()

    latest_week = sales_df[sales_df['Retail Year']==2021]['Retail Week'].max()

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df['Current Price'], y=df['%NMV'], xbins=dict(size=500), name = '%NMV', histfunc="sum"))
    fig.add_trace(go.Histogram(x=df['Current Price'], y=df['%Items Sold'], xbins=dict(size=500), name = '%Items Sold', histfunc="sum"))
    fig.add_trace(go.Histogram(x=df['Current Price'], xbins=dict(size=500), name = '%Sku Count', histnorm="probability"))
                                 
    fig.update_layout(xaxis_title_text='Price in PHP', bargap=0.2, width=850, height=400, margin=dict(l=0, r=0, t=20, b=0))
    fig.update_yaxes(title_text='Percent',tickformat = '%')
    
    st.plotly_chart(fig)