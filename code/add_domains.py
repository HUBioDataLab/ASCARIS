from collections import Counter
import pandas as pd
import numpy as np

def add_domains(data, path_to_domains):
    DOMAINS = pd.read_csv(path_to_domains, delimiter=' ')
    data = data.merge(DOMAINS, right_on='proteinID', left_on='uniprotID', how='left')
    data.domStart = data.domStart.astype('Int64')
    data.domEnd = data.domEnd.astype('Int64')
    data = data.drop(['proteinID'], axis=1)
    data['distance'] = np.NaN
    zeroDistanceDomains = []
    for i in data.index:
        if pd.isna(data.at[i, 'domain']):
            data.at[i, 'distance'] = np.NaN
        else:
            if int(data.at[i, 'domStart']) <= int(data.at[i, 'pos']) <= int(data.at[i, 'domEnd']):
                data.at[i, 'distance'] = 0
                DOMAIN_NAME = data.at[i, 'domain']
                zeroDistanceDomains.append(DOMAIN_NAME)
    data = data.sort_values(by=['datapoint', 'distance']).reset_index(drop=True)  # Distances will be sorted.

    ZeroDistance = data[data.distance == 0.0]
    NotZeroDistance = data[data.distance != 0.0]
    NotZeroDistance.distance = -1000

    NotZeroDistance = NotZeroDistance[~NotZeroDistance.datapoint.isin(ZeroDistance.datapoint.to_list())]

    data = pd.concat([ZeroDistance, NotZeroDistance], sort=False)
    data.reset_index(drop=True, inplace=True)
    data.fillna(-1, inplace=True)
    return data
