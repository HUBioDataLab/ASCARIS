import pandas as pd
import numpy as np
from add_sequence import *


def create_isoforms(uniprot_id, isoform_fasta):
    if uniprot_id not in isoform_fasta.uniprotID.to_list():
        isoform_current = pd.DataFrame(get_isoforms(uniprot_id).items(), columns=['uniprotID', 'isoformSequence'])
        isoform_current['whichIsoform'] = isoform_current['uniprotID'].apply(lambda x: x[7:10].strip())
        isoform_current['uniprotID'] = isoform_current['uniprotID'].apply(lambda x: x[0:6])
        isoform_fasta = pd.concat([isoform_fasta, isoform_current], axis=0)
    return isoform_fasta


def add_isoform(isoform_fasta, uniprot_id, variation_position, wild_type):
    if len(isoform_fasta) != 0:
        isoList = isoform_fasta[isoform_fasta['uniprotID'] == uniprot_id].isoformSequence.to_list()
        for k in isoList:
            if len(k) >= int(variation_position):
                resInIso = k[int(int(variation_position) - 1)]
                if wild_type == resInIso:
                    whichIsoform_ = isoform_fasta[isoform_fasta.isoformSequence == k].whichIsoform.to_list()[0]
                    wt_sequence_match = 'i'
                    break
            else:
                whichIsoform_ = np.NaN
                wt_sequence_match = np.NaN
    else:
        whichIsoform_ = np.NaN
        wt_sequence_match = np.NaN
    return whichIsoform_, wt_sequence_match


def add_uniprot_sequence(DATAFRAME):
    CANONICAL_FASTA = pd.DataFrame(columns=['uniprotID', 'uniprotSequence'])
    ISOFORM_FASTA = pd.DataFrame(columns=['uniprotID', 'isoformSequence'])

    UNIPROT_ID_LIST = list(set(DATAFRAME['uniprotID'].to_list()))
    for i in range(len(UNIPROT_ID_LIST)):
        CANONICAL_FASTA.at[i, 'uniprotSequence'] = get_uniprot_seq(UNIPROT_ID_LIST[i])
        CANONICAL_FASTA.at[i, 'uniprotID'] = UNIPROT_ID_LIST[i]

    canonical_fasta = CANONICAL_FASTA.drop_duplicates()
    DATAFRAME = DATAFRAME.merge(canonical_fasta, on='uniprotID', how='left')
    DATAFRAME['uniprotSequence'].replace({'': np.NaN}, inplace=True)

    for i in DATAFRAME.index:
        UNIPROT_ID = DATAFRAME.at[i, 'uniprotID']
        VARIATION_POSITION = DATAFRAME.at[i, 'pos']
        WILDTYPE = DATAFRAME.at[i, 'wt']

        if len(DATAFRAME.loc[i, 'uniprotSequence']) >= int(VARIATION_POSITION):
            can = str(DATAFRAME.at[i, 'uniprotSequence'])[int(VARIATION_POSITION) - 1]
            if WILDTYPE == can:
                DATAFRAME.loc[i, 'wt_sequence_match'] = 'm'
            elif WILDTYPE != can:
                ISOFORM_FASTA = create_isoforms(UNIPROT_ID, ISOFORM_FASTA)
                ISOFORM_NUM, MATCH_STAT = add_isoform(ISOFORM_FASTA, UNIPROT_ID, VARIATION_POSITION, WILDTYPE)
                ISOFORM_SEQUENCE = ISOFORM_FASTA[(ISOFORM_FASTA['uniprotID'] == UNIPROT_ID) &
                                                 (ISOFORM_FASTA['whichIsoform'] == ISOFORM_NUM)].isoformSequence.values


                DATAFRAME.at[i,'whichIsoform'] = ISOFORM_NUM
                DATAFRAME.at[i, 'wt_sequence_match'] = MATCH_STAT
                DATAFRAME.at[i, 'uniprotSequence'] = ISOFORM_SEQUENCE

        elif len(DATAFRAME.at[i, 'uniprotSequence']) < int(VARIATION_POSITION):
            ISOFORM_FASTA = create_isoforms(ISOFORM_FASTA, UNIPROT_ID)
            ISOFORM_NUM, MATCH_STAT = add_isoform(ISOFORM_FASTA, UNIPROT_ID, VARIATION_POSITION, WILDTYPE)
            ISOFORM_SEQUENCE = ISOFORM_FASTA[(ISOFORM_FASTA['uniprotID'] == UNIPROT_ID) &
                                             ( ISOFORM_FASTA['whichIsoform'] == ISOFORM_NUM)].isoformSequence.values

            DATAFRAME.at[i, 'whichIsoform'] = ISOFORM_NUM
            DATAFRAME.at[i, 'wt_sequence_match'] = MATCH_STAT
            DATAFRAME.at[i, 'uniprotSequence'] = ISOFORM_SEQUENCE

    DATAFRAME['uniprotSequence'] = DATAFRAME['uniprotSequence'].apply(lambda x: x[0] if (type(x) != str and len(x)>0) else x)
    DATAFRAME['uniprotSequence'] = DATAFRAME['uniprotSequence'].apply(lambda x: np.NaN if (type(x) != str and len(x) == 0) else x)

    print('>> Sequence files created...\n')
    return DATAFRAME
