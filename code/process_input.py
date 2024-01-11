import pandas as pd
import streamlit as st
def clean_data(input_set):
    data = pd.DataFrame()
    try:
        if ',' in input_set:
            input_set = [i.strip() for i in input_set.split(',')]
            initial_len = len(input_set)
            
            for i in input_set:
                data = data.append(pd.Series([j.strip() for j in i.split('-')]), ignore_index=True)
            data.columns = ['uniprotID', 'wt', 'pos', 'mut']
            data = data[((~data.uniprotID.isna()) & (~data.wt.isna()) & (~data.pos.isna()) & (~data.mut.isna()))]
            if initial_len != len(data):
                st.write(f'{initial_len- len(data)} of {initial_len} datapoints is omitted. Check your input.')
        elif '\t' in input_set:
            input_set = [i.strip() for i in input_set.split('\t')]
            initial_len = len(input_set)
            
            for i in input_set:
                data = data.append(pd.Series([j.strip() for j in i.split('-')]), ignore_index=True)
            data.columns = ['uniprotID', 'wt', 'pos', 'mut']
            data = data[((~data.uniprotID.isna()) & (~data.wt.isna()) & (~data.pos.isna()) & (~data.mut.isna()))]
            if initial_len != len(data):
                st.write(f'{initial_len- len(data)} of {initial_len} datapoints is omitted. Check your input.')
        elif '-' in input_set:
            data = data.append(pd.Series([j.strip() for j in input_set.split('-')]), ignore_index=True)
            data.columns = ['uniprotID', 'wt', 'pos', 'mut']

        elif '.txt' in input_set:
            data = pd.read_csv(input_set, sep='\t', names=['uniprotID', 'wt', 'pos', 'mut'])

        else:
            data =  pd.DataFrame(columns = ['uniprotID', 'wt', 'pos', 'mut'])
        data = data[['uniprotID', 'wt', 'pos', 'mut']]

        # Exclude termination codons, synonymous mutations and any non-standard residues such as Sec, 4 or 6.
        aa_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        data.wt = data.wt.str.strip()
        data.mut = data.mut.str.strip()
        data = data[data.wt.isin(aa_list)]
        data = data[data.mut.isin(aa_list)]

        for i in data.index:
            data.at[i, 'datapoint'] = data.at[i, 'uniprotID'] + data.at[i, 'wt'] + str(data.at[i, 'pos']) + data.at[i, 'mut']

        data = data.astype(str)
        
        return data
    except ValueError:
        st.write('Your input is in the wrong format. Please see the example.')
        return pd.DataFrame()
