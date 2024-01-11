import numpy as np

def standardize(df, get_columns):
    cols_to_change = ['sasa', 'domaindistance3D', 'disulfide', 'intMet', 'intramembrane',
                      'naturalVariant', 'dnaBinding', 'activeSite', 'nucleotideBinding',
                      'lipidation', 'site', 'transmembrane', 'crosslink', 'mutagenesis',
                      'strand', 'helix', 'turn', 'metalBinding', 'repeat', 'caBinding',
                      'topologicalDomain', 'bindingSite', 'region', 'signalPeptide',
                      'modifiedResidue', 'zincFinger', 'motif', 'coiledCoil', 'peptide',
                      'transitPeptide', 'glycosylation', 'propeptide']
    for col in cols_to_change:  # because in the other ones, they are 3D distance. Here, no distance calculated.
        df[col] = 'nan'
    df = df[get_columns.columns]

    return df


def finalTouch(data):
    for i in data.index:
        if '*' in data.at[i, 'sasa']:
            data.at[i, 'sasa'] = data.at[i, 'sasa'].split('*')[0]
    data.sasa = data.sasa.replace({'N/A': np.NaN})
    data.replace({'   N/A': np.NaN}, inplace=True)
    data.replace({'None': np.NaN, '':np.NaN}, inplace=True)
    data.sasa = data.sasa.astype(float)
    data = data.astype(str)
    for i in data.index:
        if float(data.at[i, 'sasa']) < 5:
            data.at[i, 'trsh4'] = 'core'
        elif float(data.at[i, 'sasa']) >= 5:
            data.at[i, 'trsh4'] = 'surface'
        elif data.at[i, 'sasa'] == 'nan':
            data.at[i, 'trsh4'] = 'nan'


    return data
